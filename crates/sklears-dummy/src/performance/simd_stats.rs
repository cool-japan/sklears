//! SIMD-optimized statistical operations for high-performance dummy estimator computations

use super::simd_dummy;

/// Fast SIMD-optimized mean calculation using comprehensive SIMD module
#[inline]
pub fn fast_mean(data: &[f64]) -> f64 {
    simd_dummy::simd_mean_f64(data)
}

/// Fast SIMD-optimized variance calculation using comprehensive SIMD module
#[inline]
pub fn fast_variance(data: &[f64], mean: f64) -> f64 {
    simd_dummy::simd_variance_f64(data, mean)
}

/// Fast SIMD-optimized sum calculation using comprehensive SIMD module
#[inline]
pub fn fast_sum(data: &[f64]) -> f64 {
    simd_dummy::simd_sum_f64(data)
}

/// Fast SIMD-optimized standard deviation calculation
#[inline]
pub fn fast_std_dev(data: &[f64]) -> f64 {
    simd_dummy::simd_std_dev_f64(data)
}

/// Fast SIMD-optimized min calculation
#[inline]
pub fn fast_min(data: &[f64]) -> f64 {
    simd_dummy::simd_min_f64(data)
}

/// Fast SIMD-optimized max calculation
#[inline]
pub fn fast_max(data: &[f64]) -> f64 {
    simd_dummy::simd_max_f64(data)
}

/// Fast SIMD-optimized count above threshold
#[inline]
pub fn fast_count_above_threshold(data: &[f64], threshold: f64) -> usize {
    simd_dummy::simd_count_above_threshold_f64(data, threshold)
}

/// Fast SIMD-optimized weighted sum
#[inline]
pub fn fast_weighted_sum(values: &[f64], weights: &[f64]) -> f64 {
    simd_dummy::simd_weighted_sum_f64(values, weights)
}

/// Fast SIMD-optimized weighted mean
#[inline]
pub fn fast_weighted_mean(values: &[f64], weights: &[f64]) -> f64 {
    simd_dummy::simd_weighted_mean_f64(values, weights)
}

/// Fast SIMD-optimized quantile calculation for sorted data
#[inline]
pub fn fast_quantile_sorted(sorted_data: &[f64], q: f64) -> f64 {
    simd_dummy::simd_quantile_sorted_f64(sorted_data, q)
}

/// Fast SIMD-optimized histogram computation
#[inline]
pub fn fast_histogram(data: &[f64], bins: usize, min_val: f64, max_val: f64) -> Vec<usize> {
    simd_dummy::simd_histogram_f64(data, bins, min_val, max_val)
}

/// Fast SIMD-optimized mode calculation
#[inline]
pub fn fast_mode(data: &[f64], bins: usize) -> f64 {
    simd_dummy::simd_mode_f64(data, bins)
}

/// Scalar fallback mean implementation
#[inline]
fn scalar_mean(data: &[f64]) -> f64 {
    let sum: f64 = data.iter().sum();
    sum / data.len() as f64
}

/// Scalar fallback variance implementation
#[inline]
fn scalar_variance(data: &[f64], mean: f64) -> f64 {
    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (data.len() - 1) as f64
}

/// Scalar fallback sum implementation
#[inline]
fn scalar_sum(data: &[f64]) -> f64 {
    data.iter().sum()
}
