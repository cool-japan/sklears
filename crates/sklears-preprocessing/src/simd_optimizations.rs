//! SIMD optimizations for preprocessing operations
//!
//! This module provides SIMD-accelerated implementations of common preprocessing
//! operations like element-wise arithmetic, statistical calculations, and data
//! transformations that are frequently used in scaling, normalization, and other
//! preprocessing tasks.

use scirs2_core::ndarray::{Array1, Array2, Axis};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Configuration for SIMD optimizations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SimdConfig {
    /// Whether to use SIMD optimizations
    pub enabled: bool,
    /// Minimum array size to use SIMD (avoids overhead for small arrays)
    pub min_size_threshold: usize,
    /// Force specific SIMD width (None for auto-detection)
    pub force_width: Option<usize>,
    /// Whether to use parallel SIMD for large arrays
    pub use_parallel: bool,
}

impl Default for SimdConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_size_threshold: 32,
            force_width: None,
            use_parallel: true,
        }
    }
}

/// SIMD-optimized element-wise addition of a scalar to a vector
pub fn add_scalar_f64_simd(data: &mut [f64], scalar: f64, config: &SimdConfig) {
    if !config.enabled || data.len() < config.min_size_threshold {
        add_scalar_f64_scalar(data, scalar);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_scalar_f64_avx2(data, scalar) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { add_scalar_f64_sse2(data, scalar) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        add_scalar_f64_neon(data, scalar)
    };

    #[cfg(not(target_arch = "aarch64"))]
    add_scalar_f64_scalar(data, scalar);
}

/// SIMD-optimized element-wise subtraction of a scalar from a vector
pub fn sub_scalar_f64_simd(data: &mut [f64], scalar: f64, config: &SimdConfig) {
    add_scalar_f64_simd(data, -scalar, config);
}

/// SIMD-optimized element-wise multiplication of a vector by a scalar
pub fn mul_scalar_f64_simd(data: &mut [f64], scalar: f64, config: &SimdConfig) {
    if !config.enabled || data.len() < config.min_size_threshold {
        mul_scalar_f64_scalar(data, scalar);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { mul_scalar_f64_avx2(data, scalar) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { mul_scalar_f64_sse2(data, scalar) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        mul_scalar_f64_neon(data, scalar)
    };

    #[cfg(not(target_arch = "aarch64"))]
    mul_scalar_f64_scalar(data, scalar);
}

/// SIMD-optimized element-wise division of a vector by a scalar
pub fn div_scalar_f64_simd(data: &mut [f64], scalar: f64, config: &SimdConfig) {
    if scalar != 0.0 {
        mul_scalar_f64_simd(data, 1.0 / scalar, config);
    }
}

/// SIMD-optimized vector addition
pub fn add_vectors_f64_simd(a: &[f64], b: &[f64], result: &mut [f64], config: &SimdConfig) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    if !config.enabled || a.len() < config.min_size_threshold {
        add_vectors_f64_scalar(a, b, result);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { add_vectors_f64_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { add_vectors_f64_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        add_vectors_f64_neon(a, b, result)
    };

    #[cfg(not(target_arch = "aarch64"))]
    add_vectors_f64_scalar(a, b, result);
}

/// SIMD-optimized vector subtraction
pub fn sub_vectors_f64_simd(a: &[f64], b: &[f64], result: &mut [f64], config: &SimdConfig) {
    assert_eq!(a.len(), b.len());
    assert_eq!(a.len(), result.len());

    if !config.enabled || a.len() < config.min_size_threshold {
        sub_vectors_f64_scalar(a, b, result);
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sub_vectors_f64_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { sub_vectors_f64_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(target_arch = "aarch64")]
    unsafe {
        sub_vectors_f64_neon(a, b, result)
    };

    #[cfg(not(target_arch = "aarch64"))]
    sub_vectors_f64_scalar(a, b, result);
}

/// SIMD-optimized mean calculation
pub fn mean_f64_simd(data: &[f64], config: &SimdConfig) -> f64 {
    if data.is_empty() {
        return 0.0;
    }

    if !config.enabled || data.len() < config.min_size_threshold {
        return mean_f64_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { mean_f64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { mean_f64_sse2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    return unsafe { mean_f64_neon(data) };

    #[cfg(not(target_arch = "aarch64"))]
    mean_f64_scalar(data)
}

/// SIMD-optimized variance calculation
pub fn variance_f64_simd(data: &[f64], mean: f64, config: &SimdConfig) -> f64 {
    if data.len() <= 1 {
        return 0.0;
    }

    if !config.enabled || data.len() < config.min_size_threshold {
        return variance_f64_scalar(data, mean);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { variance_f64_avx2(data, mean) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { variance_f64_sse2(data, mean) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    return unsafe { variance_f64_neon(data, mean) };

    #[cfg(not(target_arch = "aarch64"))]
    variance_f64_scalar(data, mean)
}

/// SIMD-optimized min/max finding
pub fn min_max_f64_simd(data: &[f64], config: &SimdConfig) -> (f64, f64) {
    if data.is_empty() {
        return (f64::NAN, f64::NAN);
    }

    if !config.enabled || data.len() < config.min_size_threshold {
        return min_max_f64_scalar(data);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { min_max_f64_avx2(data) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { min_max_f64_sse2(data) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    return unsafe { min_max_f64_neon(data) };

    #[cfg(not(target_arch = "aarch64"))]
    min_max_f64_scalar(data)
}

// Scalar fallback implementations

fn add_scalar_f64_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x += scalar;
    }
}

fn mul_scalar_f64_scalar(data: &mut [f64], scalar: f64) {
    for x in data.iter_mut() {
        *x *= scalar;
    }
}

fn add_vectors_f64_scalar(a: &[f64], b: &[f64], result: &mut [f64]) {
    for ((x, y), r) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *r = x + y;
    }
}

fn sub_vectors_f64_scalar(a: &[f64], b: &[f64], result: &mut [f64]) {
    for ((x, y), r) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
        *r = x - y;
    }
}

fn mean_f64_scalar(data: &[f64]) -> f64 {
    data.iter().sum::<f64>() / data.len() as f64
}

fn variance_f64_scalar(data: &[f64], mean: f64) -> f64 {
    data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64
}

fn min_max_f64_scalar(data: &[f64]) -> (f64, f64) {
    let mut min = data[0];
    let mut max = data[0];

    for &x in &data[1..] {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }

    (min, max)
}

// x86_64 SSE2 implementations

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn add_scalar_f64_sse2(data: &mut [f64], scalar: f64) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm_set1_pd(scalar);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = _mm_loadu_pd(data.as_ptr().add(i));
        let result = _mm_add_pd(data_vec, scalar_vec);
        _mm_storeu_pd(data.as_mut_ptr().add(i), result);
        i += 2;
    }

    // Handle remaining elements
    while i < data.len() {
        data[i] += scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mul_scalar_f64_sse2(data: &mut [f64], scalar: f64) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm_set1_pd(scalar);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = _mm_loadu_pd(data.as_ptr().add(i));
        let result = _mm_mul_pd(data_vec, scalar_vec);
        _mm_storeu_pd(data.as_mut_ptr().add(i), result);
        i += 2;
    }

    while i < data.len() {
        data[i] *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn add_vectors_f64_sse2(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;

    let mut i = 0;

    while i + 2 <= a.len() {
        let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
        let result_vec = _mm_add_pd(a_vec, b_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(i), result_vec);
        i += 2;
    }

    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sub_vectors_f64_sse2(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;

    let mut i = 0;

    while i + 2 <= a.len() {
        let a_vec = _mm_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm_loadu_pd(b.as_ptr().add(i));
        let result_vec = _mm_sub_pd(a_vec, b_vec);
        _mm_storeu_pd(result.as_mut_ptr().add(i), result_vec);
        i += 2;
    }

    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn mean_f64_sse2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = _mm_setzero_pd();
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = _mm_loadu_pd(data.as_ptr().add(i));
        sum = _mm_add_pd(sum, data_vec);
        i += 2;
    }

    let mut result = [0.0f64; 2];
    _mm_storeu_pd(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1];

    while i < data.len() {
        scalar_sum += data[i];
        i += 1;
    }

    scalar_sum / data.len() as f64
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn variance_f64_sse2(data: &[f64], mean: f64) -> f64 {
    use std::arch::x86_64::*;

    let mean_vec = _mm_set1_pd(mean);
    let mut sum = _mm_setzero_pd();
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = _mm_loadu_pd(data.as_ptr().add(i));
        let diff = _mm_sub_pd(data_vec, mean_vec);
        let squared = _mm_mul_pd(diff, diff);
        sum = _mm_add_pd(sum, squared);
        i += 2;
    }

    let mut result = [0.0f64; 2];
    _mm_storeu_pd(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1];

    while i < data.len() {
        let diff = data[i] - mean;
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / (data.len() - 1) as f64
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn min_max_f64_sse2(data: &[f64]) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut min_vec = _mm_set1_pd(data[0]);
    let mut max_vec = _mm_set1_pd(data[0]);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = _mm_loadu_pd(data.as_ptr().add(i));
        min_vec = _mm_min_pd(min_vec, data_vec);
        max_vec = _mm_max_pd(max_vec, data_vec);
        i += 2;
    }

    let mut min_result = [0.0f64; 2];
    let mut max_result = [0.0f64; 2];
    _mm_storeu_pd(min_result.as_mut_ptr(), min_vec);
    _mm_storeu_pd(max_result.as_mut_ptr(), max_vec);

    let mut min_val = min_result[0].min(min_result[1]);
    let mut max_val = max_result[0].max(max_result[1]);

    while i < data.len() {
        if data[i] < min_val {
            min_val = data[i];
        }
        if data[i] > max_val {
            max_val = data[i];
        }
        i += 1;
    }

    (min_val, max_val)
}

// x86_64 AVX2 implementations

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_scalar_f64_avx2(data: &mut [f64], scalar: f64) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_pd(scalar);
    let mut i = 0;

    while i + 4 <= data.len() {
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
        let result = _mm256_add_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(data.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < data.len() {
        data[i] += scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_scalar_f64_avx2(data: &mut [f64], scalar: f64) {
    use std::arch::x86_64::*;

    let scalar_vec = _mm256_set1_pd(scalar);
    let mut i = 0;

    while i + 4 <= data.len() {
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
        let result = _mm256_mul_pd(data_vec, scalar_vec);
        _mm256_storeu_pd(data.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < data.len() {
        data[i] *= scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_vectors_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let result_vec = _mm256_add_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sub_vectors_f64_avx2(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm256_loadu_pd(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_pd(b.as_ptr().add(i));
        let result_vec = _mm256_sub_pd(a_vec, b_vec);
        _mm256_storeu_pd(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mean_f64_avx2(data: &[f64]) -> f64 {
    use std::arch::x86_64::*;

    let mut sum = _mm256_setzero_pd();
    let mut i = 0;

    while i + 4 <= data.len() {
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
        sum = _mm256_add_pd(sum, data_vec);
        i += 4;
    }

    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f64>();

    while i < data.len() {
        scalar_sum += data[i];
        i += 1;
    }

    scalar_sum / data.len() as f64
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn variance_f64_avx2(data: &[f64], mean: f64) -> f64 {
    use std::arch::x86_64::*;

    let mean_vec = _mm256_set1_pd(mean);
    let mut sum = _mm256_setzero_pd();
    let mut i = 0;

    while i + 4 <= data.len() {
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
        let diff = _mm256_sub_pd(data_vec, mean_vec);
        let squared = _mm256_mul_pd(diff, diff);
        sum = _mm256_add_pd(sum, squared);
        i += 4;
    }

    let mut result = [0.0f64; 4];
    _mm256_storeu_pd(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f64>();

    while i < data.len() {
        let diff = data[i] - mean;
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / (data.len() - 1) as f64
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn min_max_f64_avx2(data: &[f64]) -> (f64, f64) {
    use std::arch::x86_64::*;

    let mut min_vec = _mm256_set1_pd(data[0]);
    let mut max_vec = _mm256_set1_pd(data[0]);
    let mut i = 0;

    while i + 4 <= data.len() {
        let data_vec = _mm256_loadu_pd(data.as_ptr().add(i));
        min_vec = _mm256_min_pd(min_vec, data_vec);
        max_vec = _mm256_max_pd(max_vec, data_vec);
        i += 4;
    }

    let mut min_result = [0.0f64; 4];
    let mut max_result = [0.0f64; 4];
    _mm256_storeu_pd(min_result.as_mut_ptr(), min_vec);
    _mm256_storeu_pd(max_result.as_mut_ptr(), max_vec);

    let mut min_val = min_result.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let mut max_val = max_result.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    while i < data.len() {
        if data[i] < min_val {
            min_val = data[i];
        }
        if data[i] > max_val {
            max_val = data[i];
        }
        i += 1;
    }

    (min_val, max_val)
}

// ARM NEON implementations

#[cfg(target_arch = "aarch64")]
unsafe fn add_scalar_f64_neon(data: &mut [f64], scalar: f64) {
    use std::arch::aarch64::*;

    let scalar_vec = vdupq_n_f64(scalar);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = vld1q_f64(data.as_ptr().add(i));
        let result = vaddq_f64(data_vec, scalar_vec);
        vst1q_f64(data.as_mut_ptr().add(i), result);
        i += 2;
    }

    while i < data.len() {
        data[i] += scalar;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mul_scalar_f64_neon(data: &mut [f64], scalar: f64) {
    use std::arch::aarch64::*;

    let scalar_vec = vdupq_n_f64(scalar);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = vld1q_f64(data.as_ptr().add(i));
        let result = vmulq_f64(data_vec, scalar_vec);
        vst1q_f64(data.as_mut_ptr().add(i), result);
        i += 2;
    }

    while i < data.len() {
        data[i] *= scalar;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn add_vectors_f64_neon(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::aarch64::*;

    let mut i = 0;

    while i + 2 <= a.len() {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        let result_vec = vaddq_f64(a_vec, b_vec);
        vst1q_f64(result.as_mut_ptr().add(i), result_vec);
        i += 2;
    }

    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn sub_vectors_f64_neon(a: &[f64], b: &[f64], result: &mut [f64]) {
    use std::arch::aarch64::*;

    let mut i = 0;

    while i + 2 <= a.len() {
        let a_vec = vld1q_f64(a.as_ptr().add(i));
        let b_vec = vld1q_f64(b.as_ptr().add(i));
        let result_vec = vsubq_f64(a_vec, b_vec);
        vst1q_f64(result.as_mut_ptr().add(i), result_vec);
        i += 2;
    }

    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn mean_f64_neon(data: &[f64]) -> f64 {
    use std::arch::aarch64::*;

    let mut sum = vdupq_n_f64(0.0);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = vld1q_f64(data.as_ptr().add(i));
        sum = vaddq_f64(sum, data_vec);
        i += 2;
    }

    let mut scalar_sum = vaddvq_f64(sum);

    while i < data.len() {
        scalar_sum += data[i];
        i += 1;
    }

    scalar_sum / data.len() as f64
}

#[cfg(target_arch = "aarch64")]
unsafe fn variance_f64_neon(data: &[f64], mean: f64) -> f64 {
    use std::arch::aarch64::*;

    let mean_vec = vdupq_n_f64(mean);
    let mut sum = vdupq_n_f64(0.0);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = vld1q_f64(data.as_ptr().add(i));
        let diff = vsubq_f64(data_vec, mean_vec);
        let squared = vmulq_f64(diff, diff);
        sum = vaddq_f64(sum, squared);
        i += 2;
    }

    let mut scalar_sum = vaddvq_f64(sum);

    while i < data.len() {
        let diff = data[i] - mean;
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum / (data.len() - 1) as f64
}

#[cfg(target_arch = "aarch64")]
unsafe fn min_max_f64_neon(data: &[f64]) -> (f64, f64) {
    use std::arch::aarch64::*;

    let mut min_vec = vdupq_n_f64(data[0]);
    let mut max_vec = vdupq_n_f64(data[0]);
    let mut i = 0;

    while i + 2 <= data.len() {
        let data_vec = vld1q_f64(data.as_ptr().add(i));
        min_vec = vminq_f64(min_vec, data_vec);
        max_vec = vmaxq_f64(max_vec, data_vec);
        i += 2;
    }

    let min_val = vminvq_f64(min_vec);
    let max_val = vmaxvq_f64(max_vec);

    let mut final_min = min_val;
    let mut final_max = max_val;

    while i < data.len() {
        if data[i] < final_min {
            final_min = data[i];
        }
        if data[i] > final_max {
            final_max = data[i];
        }
        i += 1;
    }

    (final_min, final_max)
}

/// High-level SIMD-accelerated operations for ndarray integration
pub mod ndarray_ops {
    use super::*;

    /// SIMD-optimized element-wise array addition with scalar
    pub fn add_scalar_array(array: &mut Array2<f64>, scalar: f64, config: &SimdConfig) {
        if config.use_parallel && array.len() > 1000 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                array
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut row| {
                        add_scalar_f64_simd(row.as_slice_mut().unwrap(), scalar, config);
                    });
                return;
            }
        }

        for mut row in array.axis_iter_mut(Axis(0)) {
            if let Some(slice) = row.as_slice_mut() {
                add_scalar_f64_simd(slice, scalar, config);
            } else {
                // Non-contiguous case - fallback to element-wise
                for elem in row.iter_mut() {
                    *elem += scalar;
                }
            }
        }
    }

    /// SIMD-optimized element-wise array multiplication with scalar
    pub fn mul_scalar_array(array: &mut Array2<f64>, scalar: f64, config: &SimdConfig) {
        if config.use_parallel && array.len() > 1000 {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                array
                    .axis_iter_mut(Axis(0))
                    .into_par_iter()
                    .for_each(|mut row| {
                        mul_scalar_f64_simd(row.as_slice_mut().unwrap(), scalar, config);
                    });
                return;
            }
        }

        for mut row in array.axis_iter_mut(Axis(0)) {
            if let Some(slice) = row.as_slice_mut() {
                mul_scalar_f64_simd(slice, scalar, config);
            } else {
                for elem in row.iter_mut() {
                    *elem *= scalar;
                }
            }
        }
    }

    /// SIMD-optimized column-wise mean calculation
    pub fn column_means(array: &Array2<f64>, config: &SimdConfig) -> Array1<f64> {
        let mut means = Array1::zeros(array.ncols());

        for (j, mean_col) in means.iter_mut().enumerate() {
            let column = array.column(j);
            if let Some(slice) = column.as_slice() {
                *mean_col = mean_f64_simd(slice, config);
            } else {
                *mean_col = column.iter().sum::<f64>() / array.nrows() as f64;
            }
        }

        means
    }

    /// SIMD-optimized column-wise variance calculation
    pub fn column_variances(
        array: &Array2<f64>,
        means: &Array1<f64>,
        config: &SimdConfig,
    ) -> Array1<f64> {
        let mut variances = Array1::zeros(array.ncols());

        for (j, var_col) in variances.iter_mut().enumerate() {
            let column = array.column(j);
            if let Some(slice) = column.as_slice() {
                *var_col = variance_f64_simd(slice, means[j], config);
            } else {
                let sum_sq_diff: f64 = column.iter().map(|x| (x - means[j]).powi(2)).sum();
                *var_col = sum_sq_diff / (array.nrows() - 1) as f64;
            }
        }

        variances
    }

    /// SIMD-optimized column-wise min/max calculation
    pub fn column_min_max(array: &Array2<f64>, config: &SimdConfig) -> (Array1<f64>, Array1<f64>) {
        let mut mins = Array1::zeros(array.ncols());
        let mut maxs = Array1::zeros(array.ncols());

        for j in 0..array.ncols() {
            let column = array.column(j);
            if let Some(slice) = column.as_slice() {
                let (min_val, max_val) = min_max_f64_simd(slice, config);
                mins[j] = min_val;
                maxs[j] = max_val;
            } else {
                let mut min_val = column[0];
                let mut max_val = column[0];
                for &val in column.iter().skip(1) {
                    if val < min_val {
                        min_val = val;
                    }
                    if val > max_val {
                        max_val = val;
                    }
                }
                mins[j] = min_val;
                maxs[j] = max_val;
            }
        }

        (mins, maxs)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_config() {
        let config = SimdConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_size_threshold, 32);
        assert!(config.use_parallel);
    }

    #[test]
    fn test_add_scalar_simd() {
        let config = SimdConfig::default();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();

        add_scalar_f64_simd(&mut data, 10.0, &config);

        for (i, &val) in data.iter().enumerate() {
            assert_relative_eq!(val, original[i] + 10.0, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_mul_scalar_simd() {
        let config = SimdConfig::default();
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let original = data.clone();

        mul_scalar_f64_simd(&mut data, 2.5, &config);

        for (i, &val) in data.iter().enumerate() {
            assert_relative_eq!(val, original[i] * 2.5, epsilon = 1e-14);
        }
    }

    #[test]
    fn test_vector_operations_simd() {
        let config = SimdConfig::default();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mut result = vec![0.0; 8];

        add_vectors_f64_simd(&a, &b, &mut result, &config);

        for (i, &val) in result.iter().enumerate() {
            assert_relative_eq!(val, a[i] + b[i], epsilon = 1e-14);
        }
    }

    #[test]
    fn test_mean_simd() {
        let config = SimdConfig::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mean = mean_f64_simd(&data, &config);
        let expected = 5.5;

        assert_relative_eq!(mean, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_variance_simd() {
        let config = SimdConfig::default();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let mean = 3.0;

        let variance = variance_f64_simd(&data, mean, &config);
        let expected = 2.5; // Sample variance

        assert_relative_eq!(variance, expected, epsilon = 1e-14);
    }

    #[test]
    fn test_min_max_simd() {
        let config = SimdConfig::default();
        let data = vec![3.0, 1.0, 4.0, 1.0, 5.0, 9.0, 2.0, 6.0];

        let (min_val, max_val) = min_max_f64_simd(&data, &config);

        assert_relative_eq!(min_val, 1.0, epsilon = 1e-14);
        assert_relative_eq!(max_val, 9.0, epsilon = 1e-14);
    }

    #[test]
    fn test_ndarray_operations() {
        let config = SimdConfig::default();
        let mut array = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();

        // Test scalar addition
        let original = array.clone();
        ndarray_ops::add_scalar_array(&mut array, 5.0, &config);

        for (i, (&new_val, &old_val)) in array.iter().zip(original.iter()).enumerate() {
            assert_relative_eq!(new_val, old_val + 5.0, epsilon = 1e-14);
        }

        // Test column means
        let means = ndarray_ops::column_means(&original, &config);
        assert_relative_eq!(means[0], 5.5, epsilon = 1e-14); // (1+4+7+10)/4
        assert_relative_eq!(means[1], 6.5, epsilon = 1e-14); // (2+5+8+11)/4
        assert_relative_eq!(means[2], 7.5, epsilon = 1e-14); // (3+6+9+12)/4
    }

    #[test]
    fn test_disabled_simd() {
        let config = SimdConfig {
            enabled: false,
            ..Default::default()
        };

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        add_scalar_f64_simd(&mut data, 10.0, &config);

        // Should still work correctly even with SIMD disabled
        assert_relative_eq!(data[0], 11.0, epsilon = 1e-14);
        assert_relative_eq!(data[1], 12.0, epsilon = 1e-14);
        assert_relative_eq!(data[2], 13.0, epsilon = 1e-14);
        assert_relative_eq!(data[3], 14.0, epsilon = 1e-14);
    }

    #[test]
    fn test_small_array_threshold() {
        let config = SimdConfig {
            min_size_threshold: 100, // Larger than test data
            ..Default::default()
        };

        let mut data = vec![1.0, 2.0, 3.0, 4.0];
        add_scalar_f64_simd(&mut data, 10.0, &config);

        // Should fall back to scalar implementation for small arrays
        assert_relative_eq!(data[0], 11.0, epsilon = 1e-14);
        assert_relative_eq!(data[1], 12.0, epsilon = 1e-14);
        assert_relative_eq!(data[2], 13.0, epsilon = 1e-14);
        assert_relative_eq!(data[3], 14.0, epsilon = 1e-14);
    }
}
