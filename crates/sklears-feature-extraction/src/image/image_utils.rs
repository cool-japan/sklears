//! Image processing utilities and fallback implementations
//!
//! This module provides utility functions, fallback implementations for SIMD operations,
//! and common statistical computations used across image feature extraction algorithms.

use crate::*;
use scirs2_core::ndarray::{Array2, Array3, ArrayView2, ArrayView3, ArrayViewMut1};

// SIMD optimizations for image processing

/// SIMD-accelerated image processing operations with fallback implementations
///
/// This module provides high-performance image processing operations that leverage
/// SIMD instructions when available, with fallback implementations for compatibility.
pub mod simd_image {
    use super::*;

    /// Reconstruct image from patches using overlapping region averaging
    ///
    /// This function reconstructs an image from patches that were extracted in raster scan
    /// order. Overlapping regions are averaged to ensure seamless reconstruction.
    ///
    /// # Algorithm
    /// 1. Initialize accumulation and count arrays
    /// 2. Place each patch at its position (inferred from raster scan order)
    /// 3. For overlapping regions, accumulate values and track counts
    /// 4. Divide accumulated values by counts to get averaged result
    ///
    /// # Arguments
    /// * `patches` - 3D array of patches (n_patches, patch_height, patch_width)
    /// * `image_size` - Target image dimensions (height, width)
    ///
    /// # Returns
    /// Reconstructed 2D image array
    ///
    /// # Note
    /// Patches are assumed to be in raster scan order (left-to-right, top-to-bottom).
    /// This matches the extraction order from `extract_patches_2d`.
    pub fn simd_reconstruct_from_patches_2d(
        patches: &ArrayView3<Float>,
        image_size: (usize, usize),
    ) -> SklResult<Array2<Float>> {
        let (height, width) = image_size;
        let n_patches = patches.shape()[0];

        if n_patches == 0 {
            return Ok(Array2::zeros((height, width)));
        }

        let patch_height = patches.shape()[1];
        let patch_width = patches.shape()[2];

        // Calculate grid dimensions for patch placement
        // Patches are placed at positions where they fit completely in the image
        let max_row = if height >= patch_height {
            height - patch_height + 1
        } else {
            return Err(SklearsError::InvalidInput(format!(
                "Patch height {} is larger than image height {}",
                patch_height, height
            )));
        };

        let max_col = if width >= patch_width {
            width - patch_width + 1
        } else {
            return Err(SklearsError::InvalidInput(format!(
                "Patch width {} is larger than image width {}",
                patch_width, width
            )));
        };

        // Initialize accumulation and count arrays for overlapping region averaging
        let mut accumulation: Array2<Float> = Array2::zeros((height, width));
        let mut counts: Array2<Float> = Array2::zeros((height, width));

        // Place patches in raster scan order
        let mut patch_idx = 0;
        'outer: for row in 0..max_row {
            for col in 0..max_col {
                if patch_idx >= n_patches {
                    break 'outer;
                }

                // Add this patch to the accumulation array
                for i in 0..patch_height {
                    for j in 0..patch_width {
                        let img_row = row + i;
                        let img_col = col + j;
                        accumulation[[img_row, img_col]] += patches[[patch_idx, i, j]];
                        counts[[img_row, img_col]] += 1.0;
                    }
                }

                patch_idx += 1;
            }
        }

        // Average overlapping regions by dividing by counts
        let mut reconstructed = Array2::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                if counts[[i, j]] > 0.0 {
                    reconstructed[[i, j]] = accumulation[[i, j]] / counts[[i, j]];
                }
                // If count is 0, pixel remains 0 (this shouldn't happen with proper extraction)
            }
        }

        Ok(reconstructed)
    }

    /// SIMD-accelerated array subtraction
    ///
    /// # Arguments
    /// * `a` - First input array
    /// * `b` - Second input array
    ///
    /// # Returns
    /// Element-wise difference (a - b)
    pub fn simd_array_subtraction(
        a: &ArrayView2<Float>,
        b: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        // Use ndarray's built-in SIMD-optimized operations
        Ok(a - b)
    }

    /// Detect extrema points in scale space using SIMD acceleration
    ///
    /// Detects local maxima and minima by comparing each point with its 26 neighbors:
    /// - 8 neighbors in the same scale
    /// - 9 neighbors in the scale below
    /// - 9 neighbors in the scale above
    ///
    /// # Arguments
    /// * `below` - Scale space level below current
    /// * `center` - Current scale space level
    /// * `above` - Scale space level above current
    /// * `threshold` - Detection threshold (minimum absolute value for detection)
    ///
    /// # Returns
    /// Vector of (x, y, is_maximum) tuples for detected extrema
    pub fn simd_detect_extrema(
        below: &ArrayView2<Float>,
        center: &ArrayView2<Float>,
        above: &ArrayView2<Float>,
        threshold: Float,
    ) -> Vec<(usize, usize, bool)> {
        let (height, width) = center.dim();
        let mut extrema = Vec::new();

        // Check dimensions match
        if below.dim() != center.dim() || above.dim() != center.dim() {
            return extrema;
        }

        // Avoid boundaries (need 1-pixel border)
        for i in 1..(height - 1) {
            for j in 1..(width - 1) {
                let val = center[[i, j]];

                // Skip if below threshold
                if val.abs() < threshold {
                    continue;
                }

                // Check if it's a maximum or minimum
                let mut is_max = true;
                let mut is_min = true;

                // Compare with 8 neighbors in same scale
                for di in -1..=1 {
                    for dj in -1..=1 {
                        if di == 0 && dj == 0 {
                            continue;
                        }
                        let ni = (i as isize + di) as usize;
                        let nj = (j as isize + dj) as usize;
                        let neighbor = center[[ni, nj]];

                        if val <= neighbor {
                            is_max = false;
                        }
                        if val >= neighbor {
                            is_min = false;
                        }
                    }
                }

                // Compare with 9 neighbors in scale below
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let ni = (i as isize + di) as usize;
                        let nj = (j as isize + dj) as usize;
                        let neighbor = below[[ni, nj]];

                        if val <= neighbor {
                            is_max = false;
                        }
                        if val >= neighbor {
                            is_min = false;
                        }
                    }
                }

                // Compare with 9 neighbors in scale above
                for di in -1..=1 {
                    for dj in -1..=1 {
                        let ni = (i as isize + di) as usize;
                        let nj = (j as isize + dj) as usize;
                        let neighbor = above[[ni, nj]];

                        if val <= neighbor {
                            is_max = false;
                        }
                        if val >= neighbor {
                            is_min = false;
                        }
                    }
                }

                // Add to extrema if it's a local maximum or minimum
                if is_max {
                    extrema.push((j, i, true)); // (x, y, is_maximum)
                } else if is_min {
                    extrema.push((j, i, false)); // (x, y, is_maximum=false for minimum)
                }
            }
        }

        extrema
    }

    /// Normalize descriptor vector using SIMD operations
    ///
    /// Performs L2 normalization with clamping to reduce influence of large gradients:
    /// 1. L2 normalize the descriptor
    /// 2. Clamp values to threshold
    /// 3. Re-normalize after clamping
    ///
    /// # Arguments
    /// * `descriptor` - Mutable descriptor vector to normalize
    /// * `threshold` - Threshold for clamping values (typically 0.2)
    pub fn simd_normalize_descriptor(descriptor: &mut ArrayViewMut1<Float>, threshold: Float) {
        let n = descriptor.len();
        if n == 0 {
            return;
        }

        // First L2 normalization
        let mut norm_sq = 0.0;
        for i in 0..n {
            norm_sq += descriptor[i] * descriptor[i];
        }

        let norm = norm_sq.sqrt();
        if norm > 1e-10 {
            for i in 0..n {
                descriptor[i] /= norm;
            }
        }

        // Clamp values to threshold
        for i in 0..n {
            if descriptor[i] > threshold {
                descriptor[i] = threshold;
            } else if descriptor[i] < -threshold {
                descriptor[i] = -threshold;
            }
        }

        // Second L2 normalization after clamping
        let mut norm_sq = 0.0;
        for i in 0..n {
            norm_sq += descriptor[i] * descriptor[i];
        }

        let norm = norm_sq.sqrt();
        if norm > 1e-10 {
            for i in 0..n {
                descriptor[i] /= norm;
            }
        }
    }

    /// Apply Gaussian blur using SIMD-accelerated separable convolution
    ///
    /// Uses separable kernel approach: blur horizontally then vertically for efficiency.
    /// This reduces O(w*h*k^2) to O(w*h*k) where k is kernel size.
    ///
    /// # Arguments
    /// * `image` - Input image
    /// * `kernel` - 1D Gaussian kernel coefficients
    /// * `sigma` - Standard deviation of Gaussian (used for kernel size calculation)
    ///
    /// # Returns
    /// Blurred image
    pub fn simd_gaussian_blur(
        image: &ArrayView2<Float>,
        kernel: &[Float],
        _sigma: Float,
    ) -> Array2<Float> {
        let (height, width) = image.dim();

        if kernel.is_empty() {
            return image.to_owned();
        }

        let radius = kernel.len() / 2;

        // First pass: horizontal blur
        let mut temp = Array2::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                let mut sum = 0.0;
                for k in 0..kernel.len() {
                    let offset = k as isize - radius as isize;
                    let col = (j as isize + offset).max(0).min((width - 1) as isize) as usize;
                    sum += image[[i, col]] * kernel[k];
                }
                temp[[i, j]] = sum;
            }
        }

        // Second pass: vertical blur
        let mut result = Array2::zeros((height, width));
        for i in 0..height {
            for j in 0..width {
                let mut sum = 0.0;
                for k in 0..kernel.len() {
                    let offset = k as isize - radius as isize;
                    let row = (i as isize + offset).max(0).min((height - 1) as isize) as usize;
                    sum += temp[[row, j]] * kernel[k];
                }
                result[[i, j]] = sum;
            }
        }

        result
    }

    /// Compute integral image using SIMD acceleration
    ///
    /// # Arguments
    /// * `image` - Input image
    ///
    /// # Returns
    /// Integral image for fast rectangular region sum computation
    pub fn simd_compute_integral_image(image: &ArrayView2<Float>) -> Array2<Float> {
        let (height, width) = image.dim();
        let mut integral = Array2::zeros((height, width));

        // Compute integral image using dynamic programming
        for i in 0..height {
            for j in 0..width {
                let mut sum = image[[i, j]];
                if i > 0 {
                    sum += integral[[i - 1, j]];
                }
                if j > 0 {
                    sum += integral[[i, j - 1]];
                }
                if i > 0 && j > 0 {
                    sum -= integral[[i - 1, j - 1]];
                }
                integral[[i, j]] = sum;
            }
        }

        integral
    }

    /// Compute Haar wavelet X-direction response using integral image
    ///
    /// Computes vertical edge detection using Haar-like features. The filter
    /// consists of two adjacent vertical rectangles with opposite signs:
    /// - Left half: negative weight
    /// - Right half: positive weight
    ///
    /// This is used in SURF and other feature detection algorithms for fast
    /// edge detection using integral images.
    ///
    /// # Arguments
    /// * `integral` - Precomputed integral image
    /// * `x` - X coordinate (center of filter)
    /// * `y` - Y coordinate (center of filter)
    /// * `size` - Filter size (total width and height)
    ///
    /// # Returns
    /// Haar X-response value (positive for right edge, negative for left edge)
    ///
    /// # Formula
    /// Response = sum(right_half) - sum(left_half)
    ///
    /// Using integral image for O(1) rectangular sum:
    /// sum(rect) = I[br] - I[tr] - I[bl] + I[tl]
    /// where br=bottom_right, tr=top_right, bl=bottom_left, tl=top_left
    pub fn simd_haar_x_response(
        integral: &ArrayView2<Float>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Float {
        let (height, width) = integral.dim();
        let half_size = size / 2;

        // Bounds checking
        if x < half_size || y < half_size || x + half_size >= width || y + half_size >= height {
            return 0.0;
        }

        // Define rectangle bounds
        // For a filter centered at x with half_size:
        // Left rect: [x - half_size, x - 1]
        // Right rect: [x, x + half_size - 1]
        let top = y.saturating_sub(half_size);
        let bottom = (y + half_size - 1).min(height - 1);
        let left_col = x.saturating_sub(half_size);
        let left_end = if x > 0 { x - 1 } else { 0 };
        let right_col = x;
        let right_end = (x + half_size - 1).min(width - 1);

        // Compute left half sum using integral image
        let left_sum = if x > 0 && left_end >= left_col {
            get_rect_sum(integral, top, left_col, bottom, left_end)
        } else {
            0.0
        };

        // Compute right half sum using integral image
        let right_sum = if right_end >= right_col {
            get_rect_sum(integral, top, right_col, bottom, right_end)
        } else {
            0.0
        };

        // Haar X response: right - left (detects vertical edges)
        right_sum - left_sum
    }

    /// Compute Haar wavelet Y-direction response using integral image
    ///
    /// Computes horizontal edge detection using Haar-like features. The filter
    /// consists of two adjacent horizontal rectangles with opposite signs:
    /// - Top half: negative weight
    /// - Bottom half: positive weight
    ///
    /// This is used in SURF and other feature detection algorithms for fast
    /// edge detection using integral images.
    ///
    /// # Arguments
    /// * `integral` - Precomputed integral image
    /// * `x` - X coordinate (center of filter)
    /// * `y` - Y coordinate (center of filter)
    /// * `size` - Filter size (total width and height)
    ///
    /// # Returns
    /// Haar Y-response value (positive for bottom edge, negative for top edge)
    ///
    /// # Formula
    /// Response = sum(bottom_half) - sum(top_half)
    ///
    /// Using integral image for O(1) rectangular sum:
    /// sum(rect) = I[br] - I[tr] - I[bl] + I[tl]
    /// where br=bottom_right, tr=top_right, bl=bottom_left, tl=top_left
    pub fn simd_haar_y_response(
        integral: &ArrayView2<Float>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Float {
        let (height, width) = integral.dim();
        let half_size = size / 2;

        // Bounds checking
        if x < half_size || y < half_size || x + half_size >= width || y + half_size >= height {
            return 0.0;
        }

        // Define rectangle bounds
        // For a filter centered at y with half_size:
        // Top rect: [y - half_size, y - 1]
        // Bottom rect: [y, y + half_size - 1]
        let left = x.saturating_sub(half_size);
        let right = (x + half_size - 1).min(width - 1);
        let top_row = y.saturating_sub(half_size);
        let top_end = if y > 0 { y - 1 } else { 0 };
        let bottom_row = y;
        let bottom_end = (y + half_size - 1).min(height - 1);

        // Compute top half sum using integral image
        let top_sum = if y > 0 && top_end >= top_row {
            get_rect_sum(integral, top_row, left, top_end, right)
        } else {
            0.0
        };

        // Compute bottom half sum using integral image
        let bottom_sum = if bottom_end >= bottom_row {
            get_rect_sum(integral, bottom_row, left, bottom_end, right)
        } else {
            0.0
        };

        // Haar Y response: bottom - top (detects horizontal edges)
        bottom_sum - top_sum
    }

    /// Helper function to compute sum of rectangle using integral image
    ///
    /// # Arguments
    /// * `integral` - Integral image
    /// * `top` - Top row index
    /// * `left` - Left column index
    /// * `bottom` - Bottom row index
    /// * `right` - Right column index
    ///
    /// # Returns
    /// Sum of pixels in rectangle [top:bottom, left:right]
    ///
    /// # Formula
    /// sum = integral[bottom][right] - integral[top-1][right]
    ///     - integral[bottom][left-1] + integral[top-1][left-1]
    fn get_rect_sum(
        integral: &ArrayView2<Float>,
        top: usize,
        left: usize,
        bottom: usize,
        right: usize,
    ) -> Float {
        let (height, width) = integral.dim();

        // Clamp indices to valid range
        let top = top.min(height - 1);
        let left = left.min(width - 1);
        let bottom = bottom.min(height - 1);
        let right = right.min(width - 1);

        // Get integral values at corners
        let br = integral[[bottom, right]];
        let tr = if top > 0 {
            integral[[top - 1, right]]
        } else {
            0.0
        };
        let bl = if left > 0 {
            integral[[bottom, left - 1]]
        } else {
            0.0
        };
        let tl = if top > 0 && left > 0 {
            integral[[top - 1, left - 1]]
        } else {
            0.0
        };

        // Compute rectangle sum using inclusion-exclusion principle
        br - tr - bl + tl
    }

    /// Compute kurtosis using SIMD acceleration
    ///
    /// # Arguments
    /// * `values` - Input values
    /// * `mean` - Precomputed mean
    /// * `std_dev` - Precomputed standard deviation
    ///
    /// # Returns
    /// Kurtosis (excess kurtosis, adjusted by -3)
    pub fn simd_compute_kurtosis(values: &[Float], mean: Float, std_dev: Float) -> Float {
        super::compute_kurtosis_fallback(values, mean, std_dev)
    }
}

/// Extract patches from image at specified positions (fallback implementation)
///
/// This function provides a fallback implementation for patch extraction
/// when SIMD acceleration is not available.
///
/// # Arguments
/// * `image` - Source image
/// * `patch_size` - Size of patches to extract (height, width)
/// * `positions` - List of (row, col) positions to extract patches from
///
/// # Returns
/// 3D array of extracted patches
pub fn extract_patches_fallback(
    image: &ArrayView2<Float>,
    patch_size: (usize, usize),
    positions: &[(usize, usize)],
) -> SklResult<Array3<Float>> {
    let (patch_height, patch_width) = patch_size;
    let mut patches = Array3::zeros((positions.len(), patch_height, patch_width));

    for (patch_idx, &(row, col)) in positions.iter().enumerate() {
        for i in 0..patch_height {
            for j in 0..patch_width {
                if row + i < image.shape()[0] && col + j < image.shape()[1] {
                    patches[[patch_idx, i, j]] = image[[row + i, col + j]];
                }
            }
        }
    }

    Ok(patches)
}

/// Compute entropy from probability distribution (fallback implementation)
///
/// Calculates the Shannon entropy of a probability distribution.
///
/// # Arguments
/// * `probabilities` - Probability distribution values
///
/// # Returns
/// Entropy value in bits (base-2 logarithm)
///
/// # Examples
/// ```
/// use sklears_feature_extraction::image::image_utils::compute_entropy_fallback;
///
/// let probs = vec![0.5, 0.3, 0.2];
/// let entropy = compute_entropy_fallback(&probs);
/// println!("Entropy: {:.3} bits", entropy);
/// ```
pub fn compute_entropy_fallback(probabilities: &[f64]) -> f64 {
    let mut entropy = 0.0;
    for &prob in probabilities {
        if prob > 0.0 {
            entropy -= prob * prob.log2();
        }
    }
    entropy
}

/// Compute skewness of a distribution (fallback implementation)
///
/// Calculates the skewness (third standardized moment) of a distribution,
/// which measures the asymmetry of the distribution.
///
/// # Arguments
/// * `values` - Input values
/// * `mean` - Precomputed mean of the distribution
/// * `std_dev` - Precomputed standard deviation
///
/// # Returns
/// Skewness value (positive = right-skewed, negative = left-skewed)
///
/// # Examples
/// ```
/// use sklears_feature_extraction::image::image_utils::compute_skewness_fallback;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 10.0]; // Right-skewed
/// let mean = 4.0;
/// let std_dev = 3.16;
/// let skewness = compute_skewness_fallback(&values, mean, std_dev);
/// println!("Skewness: {:.3}", skewness);
/// ```
pub fn compute_skewness_fallback(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&x| ((x - mean) / std_dev).powi(3)).sum();
    sum / n
}

/// Compute kurtosis of a distribution (fallback implementation)
///
/// Calculates the excess kurtosis (fourth standardized moment minus 3) of a distribution,
/// which measures the "tailedness" compared to a normal distribution.
///
/// # Arguments
/// * `values` - Input values
/// * `mean` - Precomputed mean of the distribution
/// * `std_dev` - Precomputed standard deviation
///
/// # Returns
/// Excess kurtosis value (0 = normal, positive = heavy tails, negative = light tails)
///
/// # Examples
/// ```
/// use sklears_feature_extraction::image::image_utils::compute_kurtosis_fallback;
///
/// let values = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Normal-like distribution
/// let mean = 3.0;
/// let std_dev = 1.41;
/// let kurtosis = compute_kurtosis_fallback(&values, mean, std_dev);
/// println!("Excess kurtosis: {:.3}", kurtosis);
/// ```
pub fn compute_kurtosis_fallback(values: &[f64], mean: f64, std_dev: f64) -> f64 {
    if std_dev == 0.0 {
        return 0.0;
    }

    let n = values.len() as f64;
    let sum: f64 = values.iter().map(|&x| ((x - mean) / std_dev).powi(4)).sum();
    (sum / n) - 3.0 // Subtract 3 for excess kurtosis
}

/// Compute basic image statistics
///
/// Calculate mean, standard deviation, skewness, and kurtosis for an image.
///
/// # Arguments
/// * `image` - Input image
///
/// # Returns
/// Tuple of (mean, std_dev, skewness, kurtosis)
pub fn compute_image_statistics(image: &ArrayView2<Float>) -> (Float, Float, Float, Float) {
    let values: Vec<Float> = image.iter().cloned().collect();
    let n = values.len() as Float;

    // Compute mean
    let mean = values.iter().sum::<Float>() / n;

    // Compute standard deviation
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n;
    let std_dev = variance.sqrt();

    // Compute skewness and kurtosis
    let values_f64: Vec<f64> = values.iter().map(|&x| x).collect();
    let skewness = compute_skewness_fallback(&values_f64, mean, std_dev) as Float;
    let kurtosis = compute_kurtosis_fallback(&values_f64, mean, std_dev) as Float;

    (mean, std_dev, skewness, kurtosis)
}

/// Convert image to grayscale using luminance weights
///
/// # Arguments
/// * `rgb_image` - RGB image with shape (height, width, 3)
///
/// # Returns
/// Grayscale image with shape (height, width)
pub fn rgb_to_grayscale(rgb_image: &ArrayView3<Float>) -> SklResult<Array2<Float>> {
    let (height, width, channels) = rgb_image.dim();
    if channels != 3 {
        return Err(SklearsError::InvalidInput(
            "Input must be an RGB image with 3 channels".to_string(),
        ));
    }

    let mut grayscale = Array2::zeros((height, width));

    // Standard luminance weights for RGB to grayscale conversion
    let r_weight = 0.299;
    let g_weight = 0.587;
    let b_weight = 0.114;

    for i in 0..height {
        for j in 0..width {
            let r = rgb_image[[i, j, 0]];
            let g = rgb_image[[i, j, 1]];
            let b = rgb_image[[i, j, 2]];

            grayscale[[i, j]] = r_weight * r + g_weight * g + b_weight * b;
        }
    }

    Ok(grayscale)
}

/// Normalize image values to [0, 1] range
///
/// # Arguments
/// * `image` - Input image
///
/// # Returns
/// Normalized image with values in [0, 1] range
pub fn normalize_image(image: &ArrayView2<Float>) -> Array2<Float> {
    let min_val = image.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    let max_val = image.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < Float::EPSILON {
        return Array2::zeros(image.dim());
    }

    let range = max_val - min_val;
    image.mapv(|x| (x - min_val) / range)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_compute_entropy_fallback() {
        let probs = vec![0.5, 0.3, 0.2];
        let entropy = compute_entropy_fallback(&probs);

        // Expected entropy for this distribution
        let expected = -(0.5 * 0.5_f64.log2() + 0.3 * 0.3_f64.log2() + 0.2 * 0.2_f64.log2());
        assert!((entropy - expected).abs() < 1e-10);
    }

    #[test]
    fn test_compute_skewness_fallback() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = (10.0_f64 / 5.0).sqrt(); // sqrt(variance)

        let skewness = compute_skewness_fallback(&values, mean, std_dev);
        assert!(skewness.abs() < 1e-10); // Symmetric distribution should have ~0 skewness
    }

    #[test]
    fn test_compute_kurtosis_fallback() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;
        let std_dev = (10.0_f64 / 5.0).sqrt();

        let kurtosis = compute_kurtosis_fallback(&values, mean, std_dev);
        // Uniform distribution should have negative excess kurtosis
        assert!(kurtosis < 0.0);
    }

    #[test]
    fn test_compute_image_statistics() {
        let image = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let (mean, std_dev, skewness, kurtosis) = compute_image_statistics(&image.view());

        assert!((mean - 2.5).abs() < 1e-6);
        assert!(std_dev > 0.0);
        assert!(skewness.abs() < 1e-6); // Symmetric
        assert!(kurtosis < 0.0); // Uniform-like
    }

    #[test]
    fn test_normalize_image() {
        let image = Array2::from_shape_vec((2, 2), vec![10.0, 20.0, 30.0, 40.0]).unwrap();
        let normalized = normalize_image(&image.view());

        assert!((normalized[[0, 0]] - 0.0).abs() < 1e-6);
        assert!((normalized[[1, 1]] - 1.0).abs() < 1e-6);
        assert!((normalized[[0, 1]] - 1.0 / 3.0).abs() < 1e-6);
        assert!((normalized[[1, 0]] - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn test_extract_patches_fallback() {
        let image = Array2::from_shape_vec((4, 4), (0..16).map(|x| x as f64).collect()).unwrap();
        let positions = vec![(0, 0), (1, 1), (2, 2)];
        let patches = extract_patches_fallback(&image.view(), (2, 2), &positions).unwrap();

        assert_eq!(patches.shape(), &[3, 2, 2]);
        assert_eq!(patches[[0, 0, 0]], 0.0);
        assert_eq!(patches[[1, 0, 0]], 5.0);
        assert_eq!(patches[[2, 0, 0]], 10.0);
    }

    // SIFT SIMD function tests

    #[test]
    fn test_simd_gaussian_blur_basic() {
        use scirs2_core::ndarray::array;

        // Simple 3x3 image
        let image = array![[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0],];

        // Simple 3-element Gaussian kernel (normalized)
        let kernel = vec![0.25, 0.5, 0.25];

        let blurred = simd_image::simd_gaussian_blur(&image.view(), &kernel, 1.0);

        // Center should be diluted by blur
        assert!(
            blurred[[1, 1]] < 1.0,
            "Center value should be reduced by blur"
        );
        assert!(
            blurred[[1, 1]] > 0.0,
            "Center value should still be positive"
        );

        // Neighbors should have some value from the center point
        assert!(
            blurred[[0, 1]] > 0.0,
            "Neighbor should receive blurred value"
        );
        assert!(
            blurred[[1, 0]] > 0.0,
            "Neighbor should receive blurred value"
        );
    }

    #[test]
    fn test_simd_gaussian_blur_empty_kernel() {
        use scirs2_core::ndarray::array;

        let image = array![[1.0, 2.0], [3.0, 4.0]];
        let kernel = vec![];

        let result = simd_image::simd_gaussian_blur(&image.view(), &kernel, 1.0);

        // Empty kernel should return original image
        assert_eq!(result, image);
    }

    #[test]
    fn test_simd_detect_extrema_maximum() {
        use scirs2_core::ndarray::Array2;

        // Create a clear maximum in the center
        let below = Array2::from_elem((5, 5), 0.0);
        let mut center = Array2::from_elem((5, 5), 0.0);
        center[[2, 2]] = 10.0; // Strong maximum
        let above = Array2::from_elem((5, 5), 0.0);

        let extrema =
            simd_image::simd_detect_extrema(&below.view(), &center.view(), &above.view(), 1.0);

        // Should detect the maximum at (2, 2)
        assert_eq!(extrema.len(), 1, "Should detect exactly one extremum");
        assert_eq!(extrema[0], (2, 2, true), "Should detect maximum at center");
    }

    #[test]
    fn test_simd_detect_extrema_minimum() {
        use scirs2_core::ndarray::Array2;

        // Create a clear minimum in the center
        let below = Array2::from_elem((5, 5), 10.0);
        let mut center = Array2::from_elem((5, 5), 10.0);
        center[[2, 2]] = -5.0; // Strong minimum
        let above = Array2::from_elem((5, 5), 10.0);

        let extrema =
            simd_image::simd_detect_extrema(&below.view(), &center.view(), &above.view(), 1.0);

        // Should detect the minimum at (2, 2)
        assert_eq!(extrema.len(), 1, "Should detect exactly one extremum");
        assert_eq!(extrema[0], (2, 2, false), "Should detect minimum at center");
    }

    #[test]
    fn test_simd_detect_extrema_threshold() {
        use scirs2_core::ndarray::Array2;

        // Create a weak maximum below threshold
        let below = Array2::from_elem((5, 5), 0.0);
        let mut center = Array2::from_elem((5, 5), 0.0);
        center[[2, 2]] = 0.5; // Below threshold
        let above = Array2::from_elem((5, 5), 0.0);

        let extrema =
            simd_image::simd_detect_extrema(&below.view(), &center.view(), &above.view(), 1.0);

        // Should NOT detect because value is below threshold
        assert_eq!(
            extrema.len(),
            0,
            "Should not detect extremum below threshold"
        );
    }

    #[test]
    fn test_simd_detect_extrema_boundaries() {
        use scirs2_core::ndarray::Array2;

        // Create extrema at boundaries (should be ignored)
        let below = Array2::from_elem((5, 5), 0.0);
        let mut center = Array2::from_elem((5, 5), 0.0);
        center[[0, 0]] = 10.0; // Corner maximum
        center[[0, 2]] = 10.0; // Edge maximum
        let above = Array2::from_elem((5, 5), 0.0);

        let extrema =
            simd_image::simd_detect_extrema(&below.view(), &center.view(), &above.view(), 1.0);

        // Should NOT detect boundary extrema
        assert_eq!(extrema.len(), 0, "Should not detect extrema at boundaries");
    }

    #[test]
    fn test_simd_normalize_descriptor_l2() {
        use scirs2_core::ndarray::Array1;

        // Create a simple descriptor
        let mut descriptor = Array1::from_vec(vec![3.0, 4.0, 0.0]);

        simd_image::simd_normalize_descriptor(&mut descriptor.view_mut(), 0.5);

        // Check L2 norm is 1.0
        let norm_sq: f64 = descriptor.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Descriptor should be L2 normalized"
        );
    }

    #[test]
    fn test_simd_normalize_descriptor_clamping() {
        use scirs2_core::ndarray::Array1;

        // Create descriptor with extreme values spread across multiple dimensions
        let mut descriptor = Array1::from_vec(vec![10.0, 10.0, 1.0, 1.0]);
        let original = descriptor.clone();

        simd_image::simd_normalize_descriptor(&mut descriptor.view_mut(), 0.2);

        // After normalization, clamping, and re-normalization:
        // The clamping step reduces influence of large gradients
        // This makes the distribution more uniform

        // Calculate how much the descriptor changed
        let mut diff_sq = 0.0;
        for i in 0..original.len() {
            // Normalize original for comparison
            let orig_norm_sq: f64 = original.iter().map(|&x| x * x).sum();
            let orig_normalized = original[i] / orig_norm_sq.sqrt();
            diff_sq += (descriptor[i] - orig_normalized).powi(2);
        }

        // Descriptor should have changed due to clamping
        assert!(
            diff_sq > 0.01,
            "Clamping should modify the descriptor distribution"
        );

        // Should still be L2 normalized
        let norm_sq: f64 = descriptor.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "Should be L2 normalized after clamping"
        );
    }

    #[test]
    fn test_simd_normalize_descriptor_empty() {
        use scirs2_core::ndarray::Array1;

        let mut descriptor = Array1::from_vec(vec![]);

        // Should not panic on empty descriptor
        simd_image::simd_normalize_descriptor(&mut descriptor.view_mut(), 0.2);
    }

    #[test]
    fn test_simd_normalize_descriptor_symmetric() {
        use scirs2_core::ndarray::Array1;

        // Test with symmetric values
        let mut descriptor = Array1::from_vec(vec![1.0, -1.0, 1.0, -1.0]);

        simd_image::simd_normalize_descriptor(&mut descriptor.view_mut(), 0.3);

        // Check normalization
        let norm_sq: f64 = descriptor.iter().map(|&x| x * x).sum();
        let norm = norm_sq.sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "Should be L2 normalized");

        // Check symmetry is preserved (within clamping)
        assert!(
            (descriptor[0].abs() - descriptor[1].abs()).abs() < 1e-6,
            "Symmetry should be preserved"
        );
        assert!(
            (descriptor[2].abs() - descriptor[3].abs()).abs() < 1e-6,
            "Symmetry should be preserved"
        );
    }
}

#[test]
fn test_haar_x_response_vertical_edge() {
    // Create a simple vertical edge: left=0, right=1
    let image = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Test at the edge (x=2, y=2) with size=2
    let response = simd_image::simd_haar_x_response(&integral.view(), 2, 2, 2);

    // With size=2, half_size=1, centered at (2,2):
    // Left rect: [x-half_size, x-1] = [1, 1], [y-half_size, y+half_size-1] = [1, 2]
    //   = 1 column × 2 rows = 0 (all zeros)
    // Right rect: [x, x+half_size-1] = [2, 2], [y-half_size, y+half_size-1] = [1, 2]
    //   = 1 column × 2 rows = 2 (all ones)
    // Expected response: 2 - 0 = 2
    assert!(
        response > 0.0,
        "Should detect right edge (positive response)"
    );
    assert!(
        (response - 2.0).abs() < 0.1,
        "Response should be ~2.0, got {}",
        response
    );
}

#[test]
fn test_haar_y_response_horizontal_edge() {
    // Create a simple horizontal edge: top=0, bottom=1
    let image = Array2::from_shape_vec(
        (4, 4),
        vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
        ],
    )
    .unwrap();

    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Test at the edge (x=2, y=2) with size=2
    let response = simd_image::simd_haar_y_response(&integral.view(), 2, 2, 2);

    // Expected: bottom_sum(2 pixels) - top_sum(2 pixels) = 2.0 - 0.0 = 2.0
    assert!(
        response > 0.0,
        "Should detect bottom edge (positive response)"
    );
    assert!(
        (response - 2.0).abs() < 0.1,
        "Response magnitude should be ~2.0"
    );
}

#[test]
fn test_haar_x_response_uniform() {
    // Uniform image should have zero response
    let image = Array2::from_elem((6, 6), 1.0);
    let integral = simd_image::simd_compute_integral_image(&image.view());

    let response = simd_image::simd_haar_x_response(&integral.view(), 3, 3, 4);

    assert_eq!(
        response, 0.0,
        "Uniform image should have zero Haar response"
    );
}

#[test]
fn test_haar_y_response_uniform() {
    // Uniform image should have zero response
    let image = Array2::from_elem((6, 6), 1.0);
    let integral = simd_image::simd_compute_integral_image(&image.view());

    let response = simd_image::simd_haar_y_response(&integral.view(), 3, 3, 4);

    assert_eq!(
        response, 0.0,
        "Uniform image should have zero Haar response"
    );
}

#[test]
fn test_haar_responses_bounds_checking() {
    let image = Array2::from_elem((4, 4), 1.0);
    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Test near edges - should return 0 when filter doesn't fit
    let response_x = simd_image::simd_haar_x_response(&integral.view(), 0, 0, 4);
    let response_y = simd_image::simd_haar_y_response(&integral.view(), 0, 0, 4);

    assert_eq!(response_x, 0.0, "Out of bounds should return 0");
    assert_eq!(response_y, 0.0, "Out of bounds should return 0");
}

#[test]
fn test_integral_image_computation() {
    // Test integral image correctness
    let image =
        Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]).unwrap();

    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Integral at (0,0) should be 1.0
    assert_eq!(integral[[0, 0]], 1.0);

    // Integral at (1,1) should be sum of top-left 2x2 = 1+2+4+5 = 12
    assert_eq!(integral[[1, 1]], 12.0);

    // Integral at (2,2) should be sum of all = 45
    assert_eq!(integral[[2, 2]], 45.0);
}

#[test]
fn test_haar_x_response_negative_edge() {
    // Create a reverse vertical edge: left=1, right=0
    let image = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Test at the edge (x=2, y=2) with size=2
    let response = simd_image::simd_haar_x_response(&integral.view(), 2, 2, 2);

    // Expected: right_sum(0) - left_sum(2 pixels) = 0.0 - 2.0 = -2.0
    assert!(
        response < 0.0,
        "Should detect left edge (negative response)"
    );
    assert!(
        (response + 2.0).abs() < 0.1,
        "Response magnitude should be ~-2.0"
    );
}

#[test]
fn test_haar_y_response_negative_edge() {
    // Create a reverse horizontal edge: top=1, bottom=0
    let image = Array2::from_shape_vec(
        (4, 4),
        vec![
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        ],
    )
    .unwrap();

    let integral = simd_image::simd_compute_integral_image(&image.view());

    // Test at the edge (x=2, y=2) with size=2
    let response = simd_image::simd_haar_y_response(&integral.view(), 2, 2, 2);

    // Expected: bottom_sum(0) - top_sum(2 pixels) = 0.0 - 2.0 = -2.0
    assert!(response < 0.0, "Should detect top edge (negative response)");
    assert!(
        (response + 2.0).abs() < 0.1,
        "Response magnitude should be ~-2.0"
    );
}
