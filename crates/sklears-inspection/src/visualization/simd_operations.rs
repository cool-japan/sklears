//! Optimized Operations for High-Performance Visualization Computations
//!
//! This module provides optimized functions for visualization data processing.
//! SIMD optimizations are available with the nightly-simd feature.

use crate::Float;

/// Grid generation for visualization surfaces
#[inline]
pub fn simd_generate_grid_range(min_val: Float, max_val: Float, steps: usize) -> Vec<Float> {
    if steps == 0 {
        return Vec::new();
    }

    let step_size = (max_val - min_val) / ((steps - 1) as Float);
    (0..steps)
        .map(|i| min_val + (i as Float) * step_size)
        .collect()
}

/// Data normalization for visualization scaling
#[inline]
pub fn simd_normalize_data(data: &[Float], target_min: Float, target_max: Float) -> Vec<Float> {
    if data.is_empty() {
        return Vec::new();
    }

    let min_val = data.iter().copied().fold(Float::INFINITY, Float::min);
    let max_val = data.iter().copied().fold(Float::NEG_INFINITY, Float::max);

    if min_val == max_val {
        return vec![target_min; data.len()];
    }

    let range = max_val - min_val;
    let target_range = target_max - target_min;

    data.iter()
        .map(|&val| target_min + ((val - min_val) / range) * target_range)
        .collect()
}

/// Fast min/max finding for data range detection
#[inline]
pub fn simd_find_min_max(data: &[Float]) -> (Float, Float) {
    if data.is_empty() {
        return (0.0, 0.0);
    }

    let mut min_val = data[0];
    let mut max_val = data[0];

    for &val in data.iter().skip(1) {
        if val < min_val {
            min_val = val;
        }
        if val > max_val {
            max_val = val;
        }
    }

    (min_val, max_val)
}

/// Color value mapping for visualization gradients
#[inline]
pub fn simd_color_mapping(values: &[Float], color_min: Float, color_max: Float) -> Vec<Float> {
    simd_normalize_data(values, color_min, color_max)
}

/// Data smoothing for visualization enhancement
#[inline]
pub fn simd_smooth_data(data: &[Float], window_size: usize) -> Vec<Float> {
    if data.is_empty() || window_size == 0 {
        return data.to_vec();
    }

    let mut result = Vec::with_capacity(data.len());
    let half_window = window_size / 2;

    for i in 0..data.len() {
        let start = i.saturating_sub(half_window);
        let end = (i + half_window + 1).min(data.len());

        let sum: Float = data[start..end].iter().sum();
        let count = (end - start) as Float;
        result.push(sum / count);
    }

    result
}

/// Fast sum computation for aggregation operations
#[inline]
pub fn simd_sum(data: &[Float]) -> Float {
    data.iter().sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simd_generate_grid_range() {
        let result = simd_generate_grid_range(0.0, 10.0, 5);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[4] - 10.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_normalize_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_normalize_data(&data, 0.0, 1.0);
        assert_eq!(result.len(), 5);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[4] - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_simd_find_min_max() {
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let (min_val, max_val) = simd_find_min_max(&data);
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);
    }

    #[test]
    fn test_simd_smooth_data() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_smooth_data(&data, 3);
        assert_eq!(result.len(), 5);
        // Check that smoothing produces reasonable results
        assert!(result[2] > 1.0 && result[2] < 5.0);
    }

    #[test]
    fn test_simd_sum() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = simd_sum(&data);
        assert_eq!(result, 15.0);
    }
}
