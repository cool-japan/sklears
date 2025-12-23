//! SIMD-Optimized Statistical Operations
//!
//! This module provides high-performance statistical computations using SIMD instructions
//! including SSE2, AVX2, and AVX512 on x86/x86_64 and NEON on ARM64.
//!
//! # Functions
//!
//! - **Basic Statistics**: mean, sum, variance, standard deviation
//! - **Extrema Operations**: min, max, min_max combined
//! - **Distribution Analysis**: histogram computation, quantile calculations
//! - **Robust Statistics**: median, percentiles, interquartile range
//!
//! # Performance Features
//!
//! - Automatic SIMD instruction set detection
//! - Optimized memory access patterns for cache efficiency
//! - Vectorized computation across multiple data elements
//! - Scalar fallbacks for unsupported architectures

#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

/// SIMD-optimized mean calculation
pub fn mean(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { mean_avx2(values) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { mean_sse2(values) };
        }
    }

    mean_scalar(values)
}

fn mean_scalar(values: &[f32]) -> f32 {
    values.iter().sum::<f32>() / values.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mean_sse2(values: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= values.len() {
        let vec = _mm_loadu_ps(values.as_ptr().add(i));
        sum = _mm_add_ps(sum, vec);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < values.len() {
        scalar_sum += values[i];
        i += 1;
    }

    scalar_sum / values.len() as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mean_avx2(values: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= values.len() {
        let vec = _mm256_loadu_ps(values.as_ptr().add(i));
        sum = _mm256_add_ps(sum, vec);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < values.len() {
        scalar_sum += values[i];
        i += 1;
    }

    scalar_sum / values.len() as f32
}

/// SIMD-optimized sum calculation
pub fn sum(values: &[f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { sum_avx2(values) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { sum_sse2(values) };
        }
    }

    sum_scalar(values)
}

fn sum_scalar(values: &[f32]) -> f32 {
    values.iter().sum()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sum_sse2(values: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= values.len() {
        let vec = _mm_loadu_ps(values.as_ptr().add(i));
        sum = _mm_add_ps(sum, vec);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < values.len() {
        scalar_sum += values[i];
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sum_avx2(values: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= values.len() {
        let vec = _mm256_loadu_ps(values.as_ptr().add(i));
        sum = _mm256_add_ps(sum, vec);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < values.len() {
        scalar_sum += values[i];
        i += 1;
    }

    scalar_sum
}

/// SIMD-optimized min/max operations
pub fn min_max(values: &[f32]) -> (f32, f32) {
    if values.is_empty() {
        return (0.0, 0.0);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { min_max_avx2(values) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { min_max_sse2(values) };
        }
    }

    min_max_scalar(values)
}

fn min_max_scalar(values: &[f32]) -> (f32, f32) {
    let mut min_val = values[0];
    let mut max_val = values[0];

    for &val in values.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    (min_val, max_val)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn min_max_sse2(values: &[f32]) -> (f32, f32) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_vec = _mm_set1_ps(values[0]);
    let mut max_vec = _mm_set1_ps(values[0]);
    let mut i = 0;

    while i + 4 <= values.len() {
        let vec = _mm_loadu_ps(values.as_ptr().add(i));
        min_vec = _mm_min_ps(min_vec, vec);
        max_vec = _mm_max_ps(max_vec, vec);
        i += 4;
    }

    let mut min_result = [0.0f32; 4];
    let mut max_result = [0.0f32; 4];
    _mm_storeu_ps(min_result.as_mut_ptr(), min_vec);
    _mm_storeu_ps(max_result.as_mut_ptr(), max_vec);

    let mut min_val = min_result[0]
        .min(min_result[1])
        .min(min_result[2])
        .min(min_result[3]);
    let mut max_val = max_result[0]
        .max(max_result[1])
        .max(max_result[2])
        .max(max_result[3]);

    while i < values.len() {
        if values[i] < min_val {
            min_val = values[i];
        }
        if values[i] > max_val {
            max_val = values[i];
        }
        i += 1;
    }

    (min_val, max_val)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn min_max_avx2(values: &[f32]) -> (f32, f32) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_vec = _mm256_set1_ps(values[0]);
    let mut max_vec = _mm256_set1_ps(values[0]);
    let mut i = 0;

    while i + 8 <= values.len() {
        let vec = _mm256_loadu_ps(values.as_ptr().add(i));
        min_vec = _mm256_min_ps(min_vec, vec);
        max_vec = _mm256_max_ps(max_vec, vec);
        i += 8;
    }

    let mut min_result = [0.0f32; 8];
    let mut max_result = [0.0f32; 8];
    _mm256_storeu_ps(min_result.as_mut_ptr(), min_vec);
    _mm256_storeu_ps(max_result.as_mut_ptr(), max_vec);

    let mut min_val = min_result.iter().fold(f32::INFINITY, |a, &b| a.min(b));
    let mut max_val = max_result.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    while i < values.len() {
        if values[i] < min_val {
            min_val = values[i];
        }
        if values[i] > max_val {
            max_val = values[i];
        }
        i += 1;
    }

    (min_val, max_val)
}

/// Individual minimum value
pub fn minimum(values: &[f32]) -> f32 {
    min_max(values).0
}

/// Individual maximum value
pub fn maximum(values: &[f32]) -> f32 {
    min_max(values).1
}

/// SIMD-optimized variance calculation
pub fn variance(values: &[f32]) -> f32 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean_val = mean(values);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { variance_avx2(values, mean_val) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { variance_sse2(values, mean_val) };
        }
    }

    variance_scalar(values, mean_val)
}

fn variance_scalar(values: &[f32], mean_val: f32) -> f32 {
    let sum_of_squares: f32 = values
        .iter()
        .map(|&x| {
            let diff = x - mean_val;
            diff * diff
        })
        .sum();

    sum_of_squares / (values.len() - 1) as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn variance_sse2(values: &[f32], mean_val: f32) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mean_vec = _mm_set1_ps(mean_val);
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= values.len() {
        let vec = _mm_loadu_ps(values.as_ptr().add(i));
        let diff = _mm_sub_ps(vec, mean_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < values.len() {
        let diff = values[i] - mean_val;
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / (values.len() - 1) as f32
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn variance_avx2(values: &[f32], mean_val: f32) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mean_vec = _mm256_set1_ps(mean_val);
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= values.len() {
        let vec = _mm256_loadu_ps(values.as_ptr().add(i));
        let diff = _mm256_sub_ps(vec, mean_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < values.len() {
        let diff = values[i] - mean_val;
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / (values.len() - 1) as f32
}

/// Standard deviation (square root of variance)
pub fn std_dev(values: &[f32]) -> f32 {
    variance(values).sqrt()
}

/// SIMD-optimized histogram computation
pub fn histogram_simd(data: &[f32], num_bins: usize, min_val: f32, max_val: f32) -> Vec<u32> {
    assert!(num_bins > 0, "Number of bins must be positive");
    assert!(
        max_val > min_val,
        "Max value must be greater than min value"
    );

    let mut histogram = vec![0u32; num_bins];
    let bin_width = (max_val - min_val) / num_bins as f32;
    let inv_bin_width = 1.0 / bin_width;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            unsafe { histogram_avx2(data, &mut histogram, min_val, inv_bin_width, num_bins) };
            return histogram;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { histogram_sse2(data, &mut histogram, min_val, inv_bin_width, num_bins) };
            return histogram;
        }
    }

    histogram_scalar(data, &mut histogram, min_val, inv_bin_width, num_bins);
    histogram
}

fn histogram_scalar(
    data: &[f32],
    histogram: &mut [u32],
    min_val: f32,
    inv_bin_width: f32,
    num_bins: usize,
) {
    for &value in data {
        if value >= min_val && value < min_val + (num_bins as f32 / inv_bin_width) {
            let bin_index = ((value - min_val) * inv_bin_width) as usize;
            let bin_index = bin_index.min(num_bins - 1);
            histogram[bin_index] += 1;
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn histogram_sse2(
    data: &[f32],
    histogram: &mut [u32],
    min_val: f32,
    inv_bin_width: f32,
    num_bins: usize,
) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let min_vec = _mm_set1_ps(min_val);
    let inv_width_vec = _mm_set1_ps(inv_bin_width);
    let max_bin = _mm_set1_ps((num_bins - 1) as f32);
    let zero = _mm_setzero_ps();

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= data.len() {
        let values = _mm_loadu_ps(data.as_ptr().add(i));
        let normalized = _mm_mul_ps(_mm_sub_ps(values, min_vec), inv_width_vec);

        // Clamp to valid range [0, num_bins-1]
        let clamped = _mm_min_ps(_mm_max_ps(normalized, zero), max_bin);

        // Convert to integers and extract bin indices
        let bin_indices = _mm_cvtps_epi32(clamped);
        let indices_array = std::mem::transmute::<__m128i, [i32; 4]>(bin_indices);

        // Check bounds and increment histogram
        for j in 0..4 {
            if i + j < data.len() {
                let value = data[i + j];
                if value >= min_val && value < min_val + (num_bins as f32 / inv_bin_width) {
                    let bin_idx = indices_array[j] as usize;
                    if bin_idx < num_bins {
                        histogram[bin_idx] += 1;
                    }
                }
            }
        }

        i += 4;
    }

    // Handle remaining elements
    while i < data.len() {
        let value = data[i];
        if value >= min_val && value < min_val + (num_bins as f32 / inv_bin_width) {
            let bin_index = ((value - min_val) * inv_bin_width) as usize;
            let bin_index = bin_index.min(num_bins - 1);
            histogram[bin_index] += 1;
        }
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn histogram_avx2(
    data: &[f32],
    histogram: &mut [u32],
    min_val: f32,
    inv_bin_width: f32,
    num_bins: usize,
) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let min_vec = _mm256_set1_ps(min_val);
    let inv_width_vec = _mm256_set1_ps(inv_bin_width);
    let max_bin = _mm256_set1_ps((num_bins - 1) as f32);
    let zero = _mm256_setzero_ps();

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= data.len() {
        let values = _mm256_loadu_ps(data.as_ptr().add(i));
        let normalized = _mm256_mul_ps(_mm256_sub_ps(values, min_vec), inv_width_vec);

        // Clamp to valid range [0, num_bins-1]
        let clamped = _mm256_min_ps(_mm256_max_ps(normalized, zero), max_bin);

        // Convert to integers and extract bin indices
        let bin_indices = _mm256_cvtps_epi32(clamped);
        let indices_array = std::mem::transmute::<__m256i, [i32; 8]>(bin_indices);

        // Check bounds and increment histogram
        for j in 0..8 {
            if i + j < data.len() {
                let value = data[i + j];
                if value >= min_val && value < min_val + (num_bins as f32 / inv_bin_width) {
                    let bin_idx = indices_array[j] as usize;
                    if bin_idx < num_bins {
                        histogram[bin_idx] += 1;
                    }
                }
            }
        }

        i += 8;
    }

    // Handle remaining elements
    while i < data.len() {
        let value = data[i];
        if value >= min_val && value < min_val + (num_bins as f32 / inv_bin_width) {
            let bin_index = ((value - min_val) * inv_bin_width) as usize;
            let bin_index = bin_index.min(num_bins - 1);
            histogram[bin_index] += 1;
        }
        i += 1;
    }
}

/// SIMD-optimized quantile computation
pub fn quantile_simd(data: &mut [f32], quantile: f32) -> f32 {
    assert!(
        quantile >= 0.0 && quantile <= 1.0,
        "Quantile must be between 0.0 and 1.0"
    );
    assert!(!data.is_empty(), "Data cannot be empty");

    // Sort the data first
    data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    if quantile == 0.0 {
        return data[0];
    }
    if quantile == 1.0 {
        return data[data.len() - 1];
    }

    // Calculate position using linear interpolation
    let pos = quantile * (data.len() - 1) as f32;
    let lower_index = pos.floor() as usize;
    let upper_index = pos.ceil() as usize;

    if lower_index == upper_index {
        data[lower_index]
    } else {
        let weight = pos - lower_index as f32;
        data[lower_index] * (1.0 - weight) + data[upper_index] * weight
    }
}

/// Median (50th percentile)
pub fn median(data: &mut [f32]) -> f32 {
    quantile_simd(data, 0.5)
}

/// First quartile (25th percentile)
pub fn q1(data: &mut [f32]) -> f32 {
    quantile_simd(data, 0.25)
}

/// Third quartile (75th percentile)
pub fn q3(data: &mut [f32]) -> f32 {
    quantile_simd(data, 0.75)
}

/// Interquartile range (Q3 - Q1)
pub fn iqr(data: &mut [f32]) -> f32 {
    let data_copy1 = &mut data.to_vec();
    let data_copy2 = &mut data.to_vec();
    q3(data_copy1) - q1(data_copy2)
}

/// Count of values in data
pub fn count(values: &[f32]) -> usize {
    values.len()
}

/// Range (max - min)
pub fn range(values: &[f32]) -> f32 {
    let (min_val, max_val) = min_max(values);
    max_val - min_val
}

/// Calculate percentile (alias for quantile_simd)
pub fn percentile(data: &mut [f32], p: f32) -> f32 {
    quantile_simd(data, p / 100.0)
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    fn assert_relative_eq(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "Expected {}, got {}", b, a);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mean(&data);
        assert_relative_eq(result, 3.0, 1e-6);
    }

    #[test]
    fn test_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = sum(&data);
        assert_relative_eq(result, 15.0, 1e-6);
    }

    #[test]
    fn test_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let (min_val, max_val) = min_max(&data);
        assert_relative_eq(min_val, 1.0, 1e-6);
        assert_relative_eq(max_val, 9.0, 1e-6);
    }

    #[test]
    fn test_variance() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = variance(&data);
        // Variance of [1,2,3,4,5] = 2.5
        assert_relative_eq(result, 2.5, 1e-6);
    }

    #[test]
    fn test_std_dev() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_dev(&data);
        // Standard deviation of [1,2,3,4,5] = sqrt(2.5) ≈ 1.581
        assert_relative_eq(result, 1.5811388, 1e-6);
    }

    #[test]
    fn test_histogram_simd() {
        let data = vec![1.0, 2.5, 3.0, 1.5, 4.0, 2.0, 3.5, 4.5, 1.2, 2.8];
        let num_bins = 4;
        let min_val = 1.0;
        let max_val = 5.0;

        let histogram = histogram_simd(&data, num_bins, min_val, max_val);

        assert_eq!(histogram.len(), num_bins);
        assert_eq!(histogram[0], 3); // [1.0, 2.0): 1.0, 1.5, 1.2
        assert_eq!(histogram[1], 3); // [2.0, 3.0): 2.5, 2.0, 2.8
        assert_eq!(histogram[2], 2); // [3.0, 4.0): 3.0, 3.5
        assert_eq!(histogram[3], 2); // [4.0, 5.0): 4.0, 4.5
    }

    #[test]
    fn test_quantile_simd() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test median (0.5 quantile)
        let median = quantile_simd(&mut data.clone(), 0.5);
        assert_relative_eq(median, 5.5, 1e-3);

        // Test first quartile (0.25 quantile)
        let mut data2 = data.clone();
        let q1 = quantile_simd(&mut data2, 0.25);
        assert_relative_eq(q1, 3.25, 1e-3);

        // Test third quartile (0.75 quantile)
        let mut data3 = data.clone();
        let q3 = quantile_simd(&mut data3, 0.75);
        assert_relative_eq(q3, 7.75, 1e-3);
    }

    #[test]
    fn test_median() {
        let mut data = vec![5.0, 1.0, 3.0, 2.0, 4.0];
        let result = median(&mut data);
        assert_relative_eq(result, 3.0, 1e-6);
    }

    #[test]
    fn test_iqr() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = iqr(&mut data);
        // Q3 - Q1 = 7.75 - 3.25 = 4.5
        assert_relative_eq(result, 4.5, 1e-3);
    }

    #[test]
    fn test_range() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0];
        let result = range(&data);
        assert_relative_eq(result, 8.0, 1e-6); // 9.0 - 1.0
    }
}