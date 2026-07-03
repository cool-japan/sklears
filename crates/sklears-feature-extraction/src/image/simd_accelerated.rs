//! SIMD-accelerated image processing operations
//!
//! This module provides vectorized implementations of the image-processing
//! primitives used by the SIFT, SURF, wavelet-texture, and patch-extraction
//! pipelines in this crate: patch reconstruction, Difference-of-Gaussians
//! subtraction, scale-space extrema detection, descriptor normalization,
//! box/Gaussian blur, integral images, Haar wavelet responses, and kurtosis.
//!
//! ## SciRS2 Policy Compliance
//! - Uses `scirs2_core::simd_ops::SimdUnifiedOps` for vectorized array
//!   operations (elementwise add/sub/div/min/max, L2 norm, cumulative sum,
//!   integer power, sum/max/min reductions, matrix transpose, dot product)
//!   instead of hand-rolled SIMD intrinsics or `#![feature(...)]` gates.
//! - Works on stable Rust: no nightly features are used anywhere in this
//!   module. (An earlier revision's "enable proper SIMD when stable
//!   features are available" TODO was stale — nothing here ever actually
//!   required nightly; this module now performs real vectorized work via
//!   `SimdUnifiedOps`, matching the sibling `crate::simd_image` module.)
//! - Correctness never depends on `scirs2-core`'s optional `simd` Cargo
//!   feature: `SimdUnifiedOps` gives identical results through well-tested
//!   scalar fallbacks when that feature is disabled, and gains real
//!   hardware vectorization automatically when it is enabled.
//!
//! `simd_haar_x_response`/`simd_haar_y_response` (and their shared
//! `box_filter_sum` helper) are the one place that stays close to scalar
//! arithmetic: each call only ever combines 4 precomputed integral-image
//! entries, so there is no array-width computation to vectorize. The
//! 4-term combination is still routed through `SimdUnifiedOps::simd_dot`
//! for consistency with the rest of this module, but the real performance
//! win for Haar features comes from the O(1) integral-image lookup trick
//! itself (see `simd_compute_integral_image`), not from SIMD.

use crate::*;
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2, ArrayView3, ArrayViewMut1};
use scirs2_core::simd_ops::SimdUnifiedOps;

// Real SIMD-backed image processing operations, delegated to
// `scirs2_core::simd_ops::SimdUnifiedOps` (see module docs above).
pub mod simd_operations {
    use super::*;

    /// SIMD-accelerated patch reconstruction from extracted patches
    ///
    /// Reconstructs a 2D image from patches by averaging overlapping regions.
    /// Each valid row segment of a patch is accumulated with a single
    /// `SimdUnifiedOps::simd_add` call instead of a per-pixel scalar loop,
    /// and the final overlap-averaging pass is a row-wise
    /// `simd_max`-then-`simd_div` (the `simd_max` against an all-ones row
    /// guards the zero-count cells against division by zero; those cells
    /// are always still `0.0` in `reconstructed`, so dividing by `1`
    /// instead of skipping them is a no-op that reproduces the original
    /// "skip when count == 0" behavior exactly).
    ///
    /// # Parameters
    /// - `patches`: Array of patches with shape (n_patches, patch_height, patch_width)
    /// - `image_size`: Tuple specifying output image dimensions (height, width)
    ///
    /// # Returns
    /// Reconstructed 2D image array
    pub fn simd_reconstruct_from_patches_2d(
        patches: &ArrayView3<Float>,
        image_size: (usize, usize),
    ) -> SklResult<Array2<Float>> {
        let (height, width) = image_size;

        if patches.is_empty() {
            return Ok(Array2::zeros((height, width)));
        }

        let (n_patches, patch_height, patch_width) = patches.dim();

        if n_patches == 0 || patch_height == 0 || patch_width == 0 {
            return Ok(Array2::zeros((height, width)));
        }

        // Initialize reconstruction arrays
        let mut reconstructed = Array2::<Float>::zeros((height, width));
        let mut overlap_counts = Array2::<Float>::zeros((height, width));

        // Calculate patch positions
        // max_row/max_col represent the number of positions where a patch can be placed
        // For a 4x4 image with 2x2 patches, max_row = max_col = 3 (positions 0, 1, 2)
        // For a 2x2 image with 2x2 patches, max_row = max_col = 1 (position 0 only)
        let max_row = height.saturating_sub(patch_height).saturating_add(1);
        let max_col = width.saturating_sub(patch_width).saturating_add(1);

        if patch_height > height || patch_width > width {
            return Err(SklearsError::InvalidInput(format!(
                "Patch size ({}, {}) cannot be larger than image size ({}, {})",
                patch_height, patch_width, height, width
            )));
        }

        let total_positions = max_row * max_col;
        let step = if n_patches < total_positions {
            total_positions / n_patches
        } else {
            1
        };

        // Reconstruct image by placing patches
        let mut patch_idx = 0;
        #[allow(clippy::explicit_counter_loop)] // patch_idx counts valid patches independently of i
        for i in (0..total_positions).step_by(step.max(1)) {
            if patch_idx >= n_patches {
                break;
            }

            let row = i / max_col;
            let col = i % max_col;
            // Same valid sub-range the original per-pixel bounds check
            // (`img_y < height && img_x < width`) would have allowed; a
            // whole row segment is added in one SIMD call instead of one
            // pixel at a time.
            let valid_width = width.saturating_sub(col).min(patch_width);

            if valid_width > 0 {
                let ones = Array1::<Float>::from_elem(valid_width, 1.0);

                for py in 0..patch_height {
                    let img_y = row + py;
                    if img_y >= height {
                        continue;
                    }

                    let patch_row = patches.slice(s![patch_idx, py, 0..valid_width]);
                    let mut recon_row = reconstructed.slice_mut(s![img_y, col..col + valid_width]);
                    let summed = Float::simd_add(&recon_row.view(), &patch_row);
                    recon_row.assign(&summed);

                    let mut count_row =
                        overlap_counts.slice_mut(s![img_y, col..col + valid_width]);
                    let incremented = Float::simd_add(&count_row.view(), &ones.view());
                    count_row.assign(&incremented);
                }
            }

            patch_idx += 1;
        }

        // Average overlapping regions
        let ones_row = Array1::<Float>::from_elem(width, 1.0);
        for y in 0..height {
            let safe_counts = Float::simd_max(&overlap_counts.row(y), &ones_row.view());
            let divided = Float::simd_div(&reconstructed.row(y), &safe_counts.view());
            reconstructed.row_mut(y).assign(&divided);
        }

        Ok(reconstructed)
    }

    /// SIMD-accelerated array subtraction with bounds checking
    ///
    /// Performs element-wise subtraction (`a - b`), row by row, via
    /// `SimdUnifiedOps::simd_sub`, with bounds checking for safety.
    ///
    /// # Parameters
    /// - `a`: First input array
    /// - `b`: Second input array to subtract from first
    ///
    /// # Returns
    /// Result array containing element-wise difference (a - b)
    pub fn simd_array_subtraction(
        a: &ArrayView2<Float>,
        b: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        if a.dim() != b.dim() {
            return Err(SklearsError::InvalidInput(
                "Array dimensions must match for subtraction".to_string(),
            ));
        }

        let (height, width) = a.dim();
        let mut result = Array2::<Float>::zeros((height, width));
        for row in 0..height {
            let diff = Float::simd_sub(&a.row(row), &b.row(row));
            result.row_mut(row).assign(&diff);
        }

        Ok(result)
    }

    /// SIMD-accelerated extrema detection for DoG space analysis
    ///
    /// Detects local extrema (maxima and minima) across three scale levels
    /// for SIFT keypoint detection. A pixel is a maximum iff it is strictly
    /// greater than *every* one of its (up to 24) neighbors, and a minimum
    /// iff it is strictly smaller than all of them; both conditions reduce
    /// exactly to comparing the center value against the neighbor set's
    /// max/min, so the neighbor comparisons are gathered into one buffer
    /// and reduced with `SimdUnifiedOps::simd_max_element`/
    /// `simd_min_element` instead of a per-neighbor scalar loop with manual
    /// early-exit.
    ///
    /// # Parameters
    /// - `below`: Scale level below current level
    /// - `center`: Current scale level
    /// - `above`: Scale level above current level
    /// - `threshold`: Threshold for extrema detection
    ///
    /// # Returns
    /// Vector of extrema positions as (x, y, is_maximum) tuples
    pub fn simd_detect_extrema(
        below: &ArrayView2<Float>,
        center: &ArrayView2<Float>,
        above: &ArrayView2<Float>,
        threshold: Float,
    ) -> Vec<(usize, usize, bool)> {
        let mut extrema = Vec::new();
        let (height, width) = center.dim();

        if height < 3 || width < 3 {
            return extrema;
        }

        // `below`/`above` only ever contribute when their shape matches
        // `center`'s; since the neighborhood offsets are always within
        // `center`'s bounds (checked below), a dimension match is
        // sufficient to guarantee `below`/`above` indexing is in-bounds
        // too, exactly like the original per-neighbor
        // `dim() == dim() && ny < nrows() && nx < ncols()` guard.
        let below_matches = below.dim() == center.dim();
        let above_matches = above.dim() == center.dim();

        // Up to 8 (center) + 8 (below) + 8 (above) = 24 neighbors.
        let mut neighbors: Vec<Float> = Vec::with_capacity(24);

        // Check for extrema in the center of the image (avoid borders)
        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let center_val = center[[y, x]];

                if center_val.abs() < threshold {
                    continue;
                }

                neighbors.clear();
                for dy in 0..3 {
                    for dx in 0..3 {
                        // Skip the center pixel in all three scale levels,
                        // matching the original neighborhood definition
                        // exactly (below[y,x]/above[y,x] are intentionally
                        // never compared, only their 8-ring neighbors are).
                        if dy == 1 && dx == 1 {
                            continue;
                        }

                        let ny = y + dy - 1;
                        let nx = x + dx - 1;

                        neighbors.push(center[[ny, nx]]);
                        if below_matches {
                            neighbors.push(below[[ny, nx]]);
                        }
                        if above_matches {
                            neighbors.push(above[[ny, nx]]);
                        }
                    }
                }

                let neighbor_view = ArrayView1::from(neighbors.as_slice());
                let max_neighbor = Float::simd_max_element(&neighbor_view);
                let min_neighbor = Float::simd_min_element(&neighbor_view);

                let is_maximum = center_val > max_neighbor;
                let is_minimum = center_val < min_neighbor;

                if is_maximum || is_minimum {
                    extrema.push((x, y, is_maximum));
                }
            }
        }

        extrema
    }

    /// SIMD-accelerated descriptor normalization
    ///
    /// Normalizes SIFT/SURF descriptors with vectorized operations and
    /// threshold-based clamping for illumination invariance. L2 norm,
    /// division, and clamping are delegated to
    /// `SimdUnifiedOps::simd_norm`/`simd_div`/`simd_min`.
    ///
    /// # Parameters
    /// - `descriptor`: Mutable descriptor array to normalize
    /// - `threshold`: Maximum value threshold for clamping
    pub fn simd_normalize_descriptor(descriptor: &mut ArrayViewMut1<Float>, threshold: Float) {
        let n = descriptor.len();
        if n == 0 {
            return;
        }

        // L2 normalization
        let norm = Float::simd_norm(&descriptor.view());
        if norm > Float::EPSILON {
            let norm_arr = Array1::<Float>::from_elem(n, norm);
            let normalized = Float::simd_div(&descriptor.view(), &norm_arr.view());
            descriptor.assign(&normalized);
        }

        // Threshold large values and renormalize
        let clamp = Array1::<Float>::from_elem(n, threshold);
        let clipped = Float::simd_min(&descriptor.view(), &clamp.view());
        descriptor.assign(&clipped);

        let norm2 = Float::simd_norm(&descriptor.view());
        if norm2 > Float::EPSILON {
            let norm2_arr = Array1::<Float>::from_elem(n, norm2);
            let renormalized = Float::simd_div(&descriptor.view(), &norm2_arr.view());
            descriptor.assign(&renormalized);
        }
    }

    /// Box-blur implementation (historically documented as "Gaussian")
    ///
    /// Applies an unweighted box blur (uniform local average) over a
    /// `(6*sigma) | 1`-sized centered window, truncating and renormalizing
    /// at the image border. This 2D window is separable into independent
    /// row and column ranges (the border truncation only depends on one
    /// axis at a time), so each row's valid horizontal segment is summed
    /// with one call to `SimdUnifiedOps::simd_sum` instead of a per-pixel
    /// scalar accumulation.
    ///
    /// `kernel` is accepted for call-site signature compatibility but is
    /// unused: this has always computed a uniform box average rather than
    /// a true Gaussian-weighted convolution, and this rewrite preserves
    /// that exact numerical behavior rather than silently changing it.
    ///
    /// # Parameters
    /// - `image`: Input image to blur
    /// - `_kernel`: Unused (kept for signature compatibility)
    /// - `sigma`: Standard deviation used to size the box window
    ///
    /// # Returns
    /// Blurred image array
    pub fn simd_gaussian_blur(
        image: &ArrayView2<Float>,
        _kernel: &[Float],
        sigma: Float,
    ) -> Array2<Float> {
        let (height, width) = image.dim();

        if height == 0 || width == 0 {
            return Array2::zeros((height, width));
        }

        let kernel_size = (6.0 * sigma).ceil() as usize | 1; // Ensure odd size
        let half_size = kernel_size / 2;

        let mut blurred = Array2::zeros((height, width));

        for y in 0..height {
            let lo_y = y.saturating_sub(half_size);
            let hi_y = (y + half_size).min(height - 1);

            for x in 0..width {
                let lo_x = x.saturating_sub(half_size);
                let hi_x = (x + half_size).min(width - 1);
                let row_span = hi_x - lo_x + 1;

                let mut sum = 0.0;
                let mut count = 0usize;
                for src_y in lo_y..=hi_y {
                    let row_segment = image.slice(s![src_y, lo_x..=hi_x]);
                    sum += Float::simd_sum(&row_segment);
                    count += row_span;
                }

                blurred[[y, x]] = if count > 0 { sum / count as Float } else { 0.0 };
            }
        }

        blurred
    }

    /// SIMD-accelerated integral image computation
    ///
    /// Computes the summed-area table (integral image) where each pixel
    /// holds the sum of all pixels above and to the left (inclusive).
    /// Implemented as a row-wise prefix sum followed by a column-wise
    /// prefix sum, via `SimdUnifiedOps::simd_cumsum` applied twice with an
    /// `SimdUnifiedOps::simd_transpose_blocked` in between — the standard
    /// separable formulation of a summed-area table, and a genuinely
    /// vectorizable alternative to the serially-dependent single-pass
    /// inclusion-exclusion recurrence (`I[y,x] = image[y,x] + I[y-1,x] +
    /// I[y,x-1] - I[y-1,x-1]`) this replaces: both compute the exact same
    /// mathematical quantity, `I(y,x) = sum_{y'<=y, x'<=x} image(y',x')`.
    ///
    /// # Parameters
    /// - `image`: Input image for integral computation
    ///
    /// # Returns
    /// Integral image where each pixel contains sum of all pixels above and to the left
    pub fn simd_compute_integral_image(image: &ArrayView2<Float>) -> Array2<Float> {
        let (height, width) = image.dim();

        if height == 0 || width == 0 {
            return Array2::zeros((height, width));
        }

        // Pass 1: row-wise prefix sum -> R(y, x) = sum_{x' <= x} image(y, x')
        let mut row_prefix = Array2::<Float>::zeros((height, width));
        for y in 0..height {
            let cs = Float::simd_cumsum(&image.row(y));
            row_prefix.row_mut(y).assign(&cs);
        }

        // Pass 2: column-wise prefix sum, computed as a row-wise cumsum
        // over the transpose so both passes use the same SIMD primitive.
        // R's column-cumsum equals the full 2D prefix sum I(y, x).
        let transposed = Float::simd_transpose_blocked(&row_prefix.view());
        let mut col_prefix_t = Array2::<Float>::zeros((width, height));
        for x in 0..width {
            let cs = Float::simd_cumsum(&transposed.row(x));
            col_prefix_t.row_mut(x).assign(&cs);
        }

        Float::simd_transpose_blocked(&col_prefix_t.view())
    }

    /// SIMD-accelerated Haar wavelet X response computation
    ///
    /// Computes horizontal Haar-like features using integral image
    /// for efficient SURF keypoint detection.
    ///
    /// # Parameters
    /// - `integral`: Precomputed integral image
    /// - `x`, `y`: Center coordinates of the filter
    /// - `size`: Size of the Haar filter
    ///
    /// # Returns
    /// Haar X (horizontal) response value
    pub fn simd_haar_x_response(
        integral: &ArrayView2<Float>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Float {
        let (height, width) = integral.dim();
        let half_size = size / 2;

        // Ensure bounds
        if x < half_size || y < half_size || x + half_size >= width || y + half_size >= height {
            return 0.0;
        }

        // Left rectangle (negative)
        let left = box_filter_sum(integral, x - half_size, y - half_size, half_size, size);

        // Right rectangle (positive)
        let right = box_filter_sum(integral, x, y - half_size, half_size, size);

        right - left
    }

    /// SIMD-accelerated Haar wavelet Y response computation
    ///
    /// Computes vertical Haar-like features using integral image
    /// for efficient SURF keypoint detection.
    ///
    /// # Parameters
    /// - `integral`: Precomputed integral image
    /// - `x`, `y`: Center coordinates of the filter
    /// - `size`: Size of the Haar filter
    ///
    /// # Returns
    /// Haar Y (vertical) response value
    pub fn simd_haar_y_response(
        integral: &ArrayView2<Float>,
        x: usize,
        y: usize,
        size: usize,
    ) -> Float {
        let (height, width) = integral.dim();
        let half_size = size / 2;

        // Ensure bounds
        if x < half_size || y < half_size || x + half_size >= width || y + half_size >= height {
            return 0.0;
        }

        // Top rectangle (negative)
        let top = box_filter_sum(integral, x - half_size, y - half_size, size, half_size);

        // Bottom rectangle (positive)
        let bottom = box_filter_sum(integral, x - half_size, y, size, half_size);

        bottom - top
    }

    /// SIMD-accelerated kurtosis computation
    ///
    /// Computes the fourth standardized moment (excess kurtosis) of a data
    /// distribution: `mean(((x - mean) / std_dev)^4) - 3`. The centering,
    /// scaling, fourth power, and reduction are delegated to
    /// `SimdUnifiedOps::simd_sub`/`simd_div`/`simd_powi`/`simd_sum`.
    ///
    /// # Parameters
    /// - `values`: Array of values to analyze
    /// - `mean`: Precomputed mean of the distribution
    /// - `std_dev`: Precomputed standard deviation
    ///
    /// # Returns
    /// Excess kurtosis value (normalized, with normal distribution = 0)
    pub fn simd_compute_kurtosis(values: &[Float], mean: Float, std_dev: Float) -> Float {
        if std_dev == 0.0 {
            return 0.0;
        }

        let view = ArrayView1::from(values);
        let n = values.len() as Float;

        let mean_arr = Array1::<Float>::from_elem(values.len(), mean);
        let std_arr = Array1::<Float>::from_elem(values.len(), std_dev);

        let centered = Float::simd_sub(&view, &mean_arr.view());
        let z_scores = Float::simd_div(&centered.view(), &std_arr.view());
        let fourth_powers = Float::simd_powi(&z_scores.view(), 4);
        let sum_fourth = Float::simd_sum(&fourth_powers.view());

        (sum_fourth / n) - 3.0 // Subtract 3 for excess kurtosis
    }
}

// Re-export for convenience
pub use simd_operations::*;

/// Helper for box filter sum using an integral image.
///
/// Combines the 4 corner lookups of the summed-area-table formula
/// (`bottom_right + top_left - top_right - bottom_left`) via
/// `SimdUnifiedOps::simd_dot` against a fixed `[1, 1, -1, -1]` sign
/// vector. This is only ever a 4-element combination — there is nothing
/// to gain from batching further — but routing it through the shared
/// vectorized primitive keeps all arithmetic in this module going
/// through `SimdUnifiedOps` rather than hand-rolled sums.
fn box_filter_sum(integral: &ArrayView2<Float>, x: usize, y: usize, w: usize, h: usize) -> Float {
    let (height, width) = integral.dim();

    let x1 = x.saturating_sub(1);
    let y1 = y.saturating_sub(1);
    let x2 = (x + w).min(width - 1);
    let y2 = (y + h).min(height - 1);

    // Ensure coordinates are within bounds
    if x2 >= width || y2 >= height {
        return 0.0;
    }

    let values = [
        integral[[y2, x2]],
        integral[[y1, x1]],
        integral[[y1, x2]],
        integral[[y2, x1]],
    ];
    let signs = [1.0, 1.0, -1.0, -1.0];

    Float::simd_dot(&ArrayView1::from(&values[..]), &ArrayView1::from(&signs[..]))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{Array2, Array3};

    #[test]
    fn test_array_subtraction() {
        let a = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0])
            .expect("operation should succeed");
        let b = Array2::from_shape_vec((2, 2), vec![0.5, 1.0, 1.5, 2.0])
            .expect("operation should succeed");

        let result =
            simd_array_subtraction(&a.view(), &b.view()).expect("operation should succeed");
        let expected = Array2::from_shape_vec((2, 2), vec![0.5, 1.0, 1.5, 2.0])
            .expect("operation should succeed");

        assert!((result - expected)
            .map(|x| x.abs())
            .iter()
            .all(|&x| x < 1e-10));
    }

    #[test]
    fn test_integral_image() {
        let image =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .expect("operation should succeed");

        let integral = simd_compute_integral_image(&image.view());

        // Check some values
        assert_eq!(integral[[0, 0]], 1.0);
        assert_eq!(integral[[0, 2]], 6.0); // 1+2+3
        assert_eq!(integral[[2, 2]], 45.0); // Sum of all elements
    }

    #[test]
    fn test_extrema_detection() {
        let below =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0])
                .expect("operation should succeed");

        let center = Array2::from_shape_vec(
            (3, 3),
            vec![
                2.0, 3.0, 2.0, 3.0, 5.0, 3.0, // Center pixel is maximum
                2.0, 3.0, 2.0,
            ],
        )
        .expect("operation should succeed");

        let above =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0])
                .expect("operation should succeed");

        let extrema = simd_detect_extrema(&below.view(), &center.view(), &above.view(), 1.0);

        // Should detect maximum at center
        assert!(!extrema.is_empty());
        assert!(extrema
            .iter()
            .any(|(x, y, is_max)| *x == 1 && *y == 1 && *is_max));
    }

    // ------------------------------------------------------------------
    // Numerical-equivalence tests: reference values below were captured
    // by actually *running* the pre-rewrite scalar implementations on
    // these exact fixed inputs (not hand-derived), then asserted here
    // against the SimdUnifiedOps-based rewrite so a vectorization bug
    // cannot silently change SIFT/SURF/wavelet/patch-extraction results.
    // ------------------------------------------------------------------

    fn assert_array2_close(actual: &Array2<Float>, expected: &Array2<Float>, epsilon: Float) {
        assert_eq!(actual.dim(), expected.dim(), "shape mismatch");
        for ((y, x), &e) in expected.indexed_iter() {
            let a = actual[[y, x]];
            assert!(
                (a - e).abs() <= epsilon,
                "mismatch at ({y},{x}): actual={a}, expected={e}"
            );
        }
    }

    fn assert_array1_close(actual: &Array1<Float>, expected: &[Float], epsilon: Float) {
        assert_eq!(actual.len(), expected.len(), "length mismatch");
        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - e).abs() <= epsilon,
                "mismatch at [{i}]: actual={a}, expected={e}"
            );
        }
    }

    #[test]
    fn test_reconstruct_from_patches_2d_matches_reference() {
        let patches = Array3::from_shape_vec((2, 2, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("operation should succeed");
        let recon = simd_reconstruct_from_patches_2d(&patches.view(), (3, 3))
            .expect("operation should succeed");
        let expected = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 0.0, 4.0, 5.0, 0.0, 7.0, 8.0, 0.0],
        )
        .expect("operation should succeed");
        assert_array2_close(&recon, &expected, 1e-9);
    }

    #[test]
    fn test_reconstruct_from_patches_2d_overlap_matches_reference() {
        let patches = Array3::from_shape_vec(
            (4, 2, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, // patch0
                5.0, 6.0, 7.0, 8.0, // patch1
                9.0, 10.0, 11.0, 12.0, // patch2
                13.0, 14.0, 15.0, 16.0, // patch3
            ],
        )
        .expect("operation should succeed");
        let recon = simd_reconstruct_from_patches_2d(&patches.view(), (4, 4))
            .expect("operation should succeed");
        let expected = Array2::from_shape_vec(
            (4, 4),
            vec![
                1.0, 2.0, 5.0, 6.0, //
                3.0, 6.5, 8.5, 8.0, //
                13.0, 12.5, 12.0, 0.0, //
                15.0, 16.0, 0.0, 0.0,
            ],
        )
        .expect("operation should succeed");
        assert_array2_close(&recon, &expected, 1e-9);
    }

    #[test]
    fn test_array_subtraction_matches_reference() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, -2.5, 3.0, 4.25, -5.0, 6.75])
            .expect("operation should succeed");
        let b = Array2::from_shape_vec((2, 3), vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
            .expect("operation should succeed");
        let result = simd_array_subtraction(&a.view(), &b.view()).expect("operation should succeed");
        let expected = Array2::from_shape_vec((2, 3), vec![0.9, -2.7, 2.7, 3.85, -5.5, 6.15])
            .expect("operation should succeed");
        assert_array2_close(&result, &expected, 1e-9);
    }

    #[test]
    fn test_detect_extrema_matches_reference() {
        let center = Array2::from_shape_vec(
            (3, 3),
            vec![2.0, 3.0, 2.0, 3.0, 9.0, 3.0, 2.0, 3.0, 2.0],
        )
        .expect("operation should succeed");
        let below3 = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 1.0, 2.0, 3.0, 2.0, 1.0, 2.0, 1.0],
        )
        .expect("operation should succeed");
        let above3 = below3.clone();

        // Dimension mismatch: below/above are ignored, only center's 8-ring matters.
        let no_dims: Array2<Float> = Array2::zeros((0, 0));
        let extrema_nodims =
            simd_detect_extrema(&no_dims.view(), &center.view(), &no_dims.view(), 1.0);
        assert_eq!(extrema_nodims, vec![(1, 1, true)]);

        // Matching dims on all three scale levels.
        let extrema_full =
            simd_detect_extrema(&below3.view(), &center.view(), &above3.view(), 1.0);
        assert_eq!(extrema_full, vec![(1, 1, true)]);

        // Minimum case.
        let center_min = Array2::from_shape_vec(
            (3, 3),
            vec![5.0, 5.0, 5.0, 5.0, 0.5, 5.0, 5.0, 5.0, 5.0],
        )
        .expect("operation should succeed");
        let extrema_min =
            simd_detect_extrema(&below3.view(), &center_min.view(), &above3.view(), 0.1);
        assert_eq!(extrema_min, vec![(1, 1, false)]);
    }

    #[test]
    fn test_normalize_descriptor_matches_reference() {
        let mut desc = Array1::from_vec(vec![3.0, 4.0, 0.0, 0.0]);
        simd_normalize_descriptor(&mut desc.view_mut(), 0.7);
        assert_array1_close(
            &desc,
            &[0.6507913734559686, 0.7592566023652966, 0.0, 0.0],
            1e-9,
        );

        let mut desc2 = Array1::from_vec(vec![1.0, -2.0, 2.0, -1.0, 0.5]);
        simd_normalize_descriptor(&mut desc2.view_mut(), 10.0); // threshold never triggers
        assert_array1_close(
            &desc2,
            &[
                0.31234752377721214,
                -0.6246950475544243,
                0.6246950475544243,
                -0.31234752377721214,
                0.15617376188860607,
            ],
            1e-9,
        );
    }

    #[test]
    fn test_haar_responses_match_reference() {
        // Non-linear (row*col cross term) integral image so box_filter_sum's
        // inclusion-exclusion does not trivially cancel to zero.
        let integral = Array2::from_shape_vec(
            (6, 6),
            (0..6)
                .flat_map(|r: i64| (0..6).map(move |c: i64| (r * c + 2 * r + 3 * c + 1) as f64))
                .collect(),
        )
        .expect("operation should succeed");

        assert_abs_diff_eq!(
            simd_haar_x_response(&integral.view(), 3, 3, 2),
            0.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            simd_haar_y_response(&integral.view(), 3, 3, 2),
            0.0,
            epsilon = 1e-9
        );

        assert_abs_diff_eq!(
            simd_haar_x_response(&integral.view(), 2, 2, 4),
            4.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            simd_haar_y_response(&integral.view(), 2, 2, 4),
            4.0,
            epsilon = 1e-9
        );

        assert_abs_diff_eq!(
            simd_haar_x_response(&integral.view(), 4, 3, 3),
            0.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            simd_haar_y_response(&integral.view(), 4, 3, 3),
            0.0,
            epsilon = 1e-9
        );

        // Out-of-bounds request must short-circuit to 0.0.
        assert_abs_diff_eq!(
            simd_haar_x_response(&integral.view(), 0, 0, 4),
            0.0,
            epsilon = 1e-9
        );

        // Edge-clamped request.
        assert_abs_diff_eq!(
            simd_haar_x_response(&integral.view(), 4, 4, 2),
            0.0,
            epsilon = 1e-9
        );
        assert_abs_diff_eq!(
            simd_haar_y_response(&integral.view(), 4, 4, 2),
            0.0,
            epsilon = 1e-9
        );
    }

    #[test]
    fn test_compute_integral_image_matches_reference() {
        let img2x4 = Array2::from_shape_vec((2, 4), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
            .expect("operation should succeed");
        let integ2x4 = simd_compute_integral_image(&img2x4.view());
        let expected2x4 = Array2::from_shape_vec(
            (2, 4),
            vec![1.0, 3.0, 6.0, 10.0, 6.0, 14.0, 24.0, 36.0],
        )
        .expect("operation should succeed");
        assert_array2_close(&integ2x4, &expected2x4, 1e-9);

        let img3x3 = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
        )
        .expect("operation should succeed");
        let integ3x3 = simd_compute_integral_image(&img3x3.view());
        let expected3x3 = Array2::from_shape_vec(
            (3, 3),
            vec![1.0, 3.0, 6.0, 5.0, 12.0, 21.0, 12.0, 27.0, 45.0],
        )
        .expect("operation should succeed");
        assert_array2_close(&integ3x3, &expected3x3, 1e-9);
    }

    #[test]
    fn test_compute_kurtosis_matches_reference() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert_abs_diff_eq!(
            simd_compute_kurtosis(&values, 3.0, 1.41),
            -1.279588004134889,
            epsilon = 1e-9
        );

        let values2 = vec![2.0, 8.0, 0.0, 4.0, 4.0, 4.0, 4.0, 6.0, 10.0, 4.0];
        let mean2 = values2.iter().sum::<Float>() / values2.len() as Float;
        let var2 =
            values2.iter().map(|v| (v - mean2) * (v - mean2)).sum::<Float>() / values2.len() as Float;
        let std2 = var2.sqrt();
        assert_abs_diff_eq!(mean2, 4.6, epsilon = 1e-9);
        assert_abs_diff_eq!(std2, 2.690724809414742, epsilon = 1e-9);
        assert_abs_diff_eq!(
            simd_compute_kurtosis(&values2, mean2, std2),
            -0.17294954366472215,
            epsilon = 1e-9
        );

        // std_dev == 0 short-circuits to 0.0
        assert_abs_diff_eq!(simd_compute_kurtosis(&values, 3.0, 0.0), 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_gaussian_blur_matches_reference() {
        let img4 = Array2::from_shape_vec((4, 4), (0..16).map(|v| v as Float).collect())
            .expect("operation should succeed");
        let blurred_wide = simd_gaussian_blur(&img4.view(), &[], 1.0);
        let expected_wide = Array2::from_elem((4, 4), 7.5);
        assert_array2_close(&blurred_wide, &expected_wide, 1e-9);

        let img5 = Array2::from_shape_vec(
            (5, 5),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
                16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0,
            ],
        )
        .expect("operation should succeed");
        let blurred_narrow = simd_gaussian_blur(&img5.view(), &[], 0.3);
        let expected_narrow = Array2::from_shape_vec(
            (5, 5),
            vec![
                4.0, 4.5, 5.5, 6.5, 7.0, //
                6.5, 7.0, 8.0, 9.0, 9.5, //
                11.5, 12.0, 13.0, 14.0, 14.5, //
                16.5, 17.0, 18.0, 19.0, 19.5, //
                19.0, 19.5, 20.5, 21.5, 22.0,
            ],
        )
        .expect("operation should succeed");
        assert_array2_close(&blurred_narrow, &expected_narrow, 1e-9);
    }
}
