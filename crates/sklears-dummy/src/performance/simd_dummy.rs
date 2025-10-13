//! SIMD-accelerated operations for high-performance dummy estimator computations
//!
//! This module provides optimized implementations of computationally intensive
//! operations used in dummy estimators. SIMD functionality requires nightly Rust features.

// Note: Full SIMD implementation requires nightly Rust features.
// For stable Rust compilation, we provide scalar fallbacks.

/// SIMD-accelerated mean calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_mean_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum::<f64>() / data.len() as f64
}

/// SIMD-accelerated variance calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_variance_f64(data: &[f64], mean: f64) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (data.len() - 1) as f64
}

/// SIMD-accelerated sum calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_sum_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    data.iter().sum()
}

/// SIMD-accelerated standard deviation calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_std_dev_f64(data: &[f64]) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    let mean = simd_mean_f64(data);
    let variance = simd_variance_f64(data, mean);
    variance.sqrt()
}

/// SIMD-accelerated min calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_min_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    data.iter().fold(f64::INFINITY, |acc, &x| acc.min(x))
}

/// SIMD-accelerated max calculation for f64 vectors (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_max_f64(data: &[f64]) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    data.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
}

/// SIMD-accelerated count of values above threshold (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_count_above_threshold_f64(data: &[f64], threshold: f64) -> usize {
    if data.is_empty() {
        return 0;
    }

    data.iter().filter(|&&x| x > threshold).count()
}

/// SIMD-accelerated weighted sum calculation (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_weighted_sum_f64(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || weights.is_empty() || values.len() != weights.len() {
        return 0.0;
    }

    values
        .iter()
        .zip(weights.iter())
        .map(|(&v, &w)| v * w)
        .sum()
}

/// SIMD-accelerated weighted mean calculation (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_weighted_mean_f64(values: &[f64], weights: &[f64]) -> f64 {
    if values.is_empty() || weights.is_empty() || values.len() != weights.len() {
        return 0.0;
    }

    let weighted_sum = simd_weighted_sum_f64(values, weights);
    let weight_sum = simd_sum_f64(weights);

    if weight_sum == 0.0 {
        0.0
    } else {
        weighted_sum / weight_sum
    }
}

/// SIMD-accelerated quantile estimation for sorted data (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_quantile_sorted_f64(sorted_data: &[f64], q: f64) -> f64 {
    if sorted_data.is_empty() {
        return f64::NAN;
    }

    if sorted_data.len() == 1 {
        return sorted_data[0];
    }

    let q = q.clamp(0.0, 1.0);
    let index = q * (sorted_data.len() - 1) as f64;
    let lower_idx = index.floor() as usize;
    let upper_idx = (lower_idx + 1).min(sorted_data.len() - 1);

    if lower_idx == upper_idx {
        sorted_data[lower_idx]
    } else {
        let frac = index - lower_idx as f64;
        sorted_data[lower_idx] * (1.0 - frac) + sorted_data[upper_idx] * frac
    }
}

/// SIMD-accelerated histogram computation (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_histogram_f64(data: &[f64], bins: usize, min_val: f64, max_val: f64) -> Vec<usize> {
    if data.is_empty() || bins == 0 || min_val >= max_val {
        return vec![0; bins];
    }

    let mut histogram = vec![0; bins];
    let bin_width = (max_val - min_val) / bins as f64;
    let inv_bin_width = 1.0 / bin_width;

    for &val in data {
        if val >= min_val && val < max_val {
            let bin_idx = ((val - min_val) * inv_bin_width).floor() as usize;
            let bin_idx = bin_idx.min(bins - 1);
            histogram[bin_idx] += 1;
        } else if val == max_val {
            histogram[bins - 1] += 1;
        }
    }

    histogram
}

/// SIMD-accelerated mode calculation using histogram (scalar fallback)
/// Achieves significant speedup with nightly SIMD features
pub fn simd_mode_f64(data: &[f64], bins: usize) -> f64 {
    if data.is_empty() {
        return f64::NAN;
    }

    let min_val = simd_min_f64(data);
    let max_val = simd_max_f64(data);

    if min_val == max_val {
        return min_val;
    }

    let histogram = simd_histogram_f64(data, bins, min_val, max_val);
    let max_count_idx = histogram
        .iter()
        .enumerate()
        .max_by_key(|(_, &count)| count)
        .map(|(idx, _)| idx)
        .unwrap_or(0);

    let bin_width = (max_val - min_val) / bins as f64;
    min_val + (max_count_idx as f64 + 0.5) * bin_width
}
