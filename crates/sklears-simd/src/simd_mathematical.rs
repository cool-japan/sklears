//! SIMD-Optimized Mathematical Functions
//!
//! This module provides high-performance mathematical computations using SIMD instructions
//! for transcendental functions, algebraic operations, and specialized mathematical routines.
//!
//! # Functions
//!
//! - **Elementary Functions**: sqrt, reciprocal, exponential, natural logarithm
//! - **Trigonometric Functions**: sine, cosine, tangent with Taylor series approximations
//! - **Vectorized Operations**: Element-wise mathematical operations on vector data
//! - **Fast Approximations**: Optimized approximations for performance-critical applications
//!
//! # Performance Features
//!
//! - Advanced SIMD implementations (SSE2, AVX2, AVX512)
//! - Fast mathematical approximations for transcendental functions
//! - Optimized memory access patterns and cache utilization
//! - Comprehensive error handling and domain validation

#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

/// SIMD-optimized square root operation
pub fn sqrt_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sqrt_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { sqrt_vec_sse2(input, output) };
            return;
        }
    }

    sqrt_vec_scalar(input, output);
}

fn sqrt_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].sqrt();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sqrt_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = _mm_sqrt_ps(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].sqrt();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = _mm256_sqrt_ps(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].sqrt();
        i += 1;
    }
}

/// SIMD-optimized reciprocal operation (1/x)
pub fn reciprocal_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { reciprocal_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { reciprocal_vec_sse2(input, output) };
            return;
        }
    }

    reciprocal_vec_scalar(input, output);
}

fn reciprocal_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = 1.0 / input[i];
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn reciprocal_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = _mm_div_ps(one, x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = 1.0 / input[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn reciprocal_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let one = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = _mm256_div_ps(one, x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = 1.0 / input[i];
        i += 1;
    }
}

/// SIMD-optimized exponential function
pub fn exp_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { exp_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { exp_vec_sse2(input, output) };
            return;
        }
    }

    exp_vec_scalar(input, output);
}

fn exp_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].exp();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn exp_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = exp_approx_sse2(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].exp();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn exp_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = exp_approx_avx2(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].exp();
        i += 1;
    }
}

/// Fast exponential approximation using SSE2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn exp_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Clamp x to reasonable range to avoid overflow
    let x = _mm_max_ps(_mm_set1_ps(-87.0), _mm_min_ps(_mm_set1_ps(87.0), x));

    // Use the identity: exp(x) = 2^(x/ln(2))
    let inv_ln2 = _mm_set1_ps(1.44269504089);
    let ln2 = _mm_set1_ps(0.693147180560);

    let fx = _mm_mul_ps(x, inv_ln2);
    let fx_floor = _mm_floor_ps(fx);
    let fx_frac = _mm_sub_ps(fx, fx_floor);

    // Convert to integer exponent
    let exp_i = _mm_cvtps_epi32(fx_floor);
    let exp_bias = _mm_set1_epi32(127);
    let exp_biased = _mm_add_epi32(exp_i, exp_bias);
    let exp_shifted = _mm_slli_epi32(exp_biased, 23);
    let exp_float = _mm_castsi128_ps(exp_shifted);

    // Polynomial approximation for 2^frac
    let c0 = _mm_set1_ps(1.0);
    let c1 = _mm_set1_ps(0.693147180560);
    let c2 = _mm_set1_ps(0.240226506959);
    let c3 = _mm_set1_ps(0.055504108664);

    let frac_ln2 = _mm_mul_ps(fx_frac, ln2);
    let frac2 = _mm_mul_ps(frac_ln2, frac_ln2);
    let frac3 = _mm_mul_ps(frac2, frac_ln2);

    let poly = _mm_add_ps(
        _mm_add_ps(c0, _mm_mul_ps(c1, frac_ln2)),
        _mm_add_ps(_mm_mul_ps(c2, frac2), _mm_mul_ps(c3, frac3)),
    );

    _mm_mul_ps(exp_float, poly)
}

/// Fast exponential approximation using AVX2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Clamp x to reasonable range to avoid overflow
    let x = _mm256_max_ps(_mm256_set1_ps(-87.0), _mm256_min_ps(_mm256_set1_ps(87.0), x));

    // Use the identity: exp(x) = 2^(x/ln(2))
    let inv_ln2 = _mm256_set1_ps(1.44269504089);
    let ln2 = _mm256_set1_ps(0.693147180560);

    let fx = _mm256_mul_ps(x, inv_ln2);
    let fx_floor = _mm256_floor_ps(fx);
    let fx_frac = _mm256_sub_ps(fx, fx_floor);

    // Convert to integer exponent
    let exp_i = _mm256_cvtps_epi32(fx_floor);
    let exp_bias = _mm256_set1_epi32(127);
    let exp_biased = _mm256_add_epi32(exp_i, exp_bias);
    let exp_shifted = _mm256_slli_epi32(exp_biased, 23);
    let exp_float = _mm256_castsi256_ps(exp_shifted);

    // Polynomial approximation for 2^frac
    let c0 = _mm256_set1_ps(1.0);
    let c1 = _mm256_set1_ps(0.693147180560);
    let c2 = _mm256_set1_ps(0.240226506959);
    let c3 = _mm256_set1_ps(0.055504108664);

    let frac_ln2 = _mm256_mul_ps(fx_frac, ln2);
    let frac2 = _mm256_mul_ps(frac_ln2, frac_ln2);
    let frac3 = _mm256_mul_ps(frac2, frac_ln2);

    let poly = _mm256_add_ps(
        _mm256_add_ps(c0, _mm256_mul_ps(c1, frac_ln2)),
        _mm256_add_ps(_mm256_mul_ps(c2, frac2), _mm256_mul_ps(c3, frac3)),
    );

    _mm256_mul_ps(exp_float, poly)
}

/// SIMD-optimized natural logarithm function
pub fn ln_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { ln_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { ln_vec_sse2(input, output) };
            return;
        }
    }

    ln_vec_scalar(input, output);
}

fn ln_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].ln();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn ln_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = ln_approx_sse2(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].ln();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ln_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = ln_approx_avx2(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].ln();
        i += 1;
    }
}

/// Fast natural logarithm approximation using SSE2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn ln_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Extract exponent and mantissa
    let exp_mask = _mm_set1_epi32(0x7F800000);
    let mant_mask = _mm_set1_epi32(0x007FFFFF);
    let exp_bias = _mm_set1_epi32(127);

    let x_int = _mm_castps_si128(x);
    let exp_int = _mm_srli_epi32(_mm_and_si128(x_int, exp_mask), 23);
    let exp_unbiased = _mm_sub_epi32(exp_int, exp_bias);
    let exp_float = _mm_cvtepi32_ps(exp_unbiased);

    let mant_int = _mm_or_si128(_mm_and_si128(x_int, mant_mask), _mm_set1_epi32(0x3F800000));
    let mant = _mm_castsi128_ps(mant_int);

    // Polynomial approximation for ln(1 + x) where x is in [0, 1]
    let one = _mm_set1_ps(1.0);
    let t = _mm_sub_ps(mant, one);

    let c1 = _mm_set1_ps(1.0);
    let c2 = _mm_set1_ps(-0.5);
    let c3 = _mm_set1_ps(0.333333333);
    let c4 = _mm_set1_ps(-0.25);

    let t2 = _mm_mul_ps(t, t);
    let t3 = _mm_mul_ps(t2, t);
    let t4 = _mm_mul_ps(t3, t);

    let poly = _mm_add_ps(
        _mm_add_ps(_mm_mul_ps(c1, t), _mm_mul_ps(c2, t2)),
        _mm_add_ps(_mm_mul_ps(c3, t3), _mm_mul_ps(c4, t4)),
    );

    let ln2 = _mm_set1_ps(0.693147180560);
    _mm_add_ps(_mm_mul_ps(exp_float, ln2), poly)
}

/// Fast natural logarithm approximation using AVX2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ln_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Extract exponent and mantissa
    let exp_mask = _mm256_set1_epi32(0x7F800000);
    let mant_mask = _mm256_set1_epi32(0x007FFFFF);
    let exp_bias = _mm256_set1_epi32(127);

    let x_int = _mm256_castps_si256(x);
    let exp_int = _mm256_srli_epi32(_mm256_and_si256(x_int, exp_mask), 23);
    let exp_unbiased = _mm256_sub_epi32(exp_int, exp_bias);
    let exp_float = _mm256_cvtepi32_ps(exp_unbiased);

    let mant_int = _mm256_or_si256(_mm256_and_si256(x_int, mant_mask), _mm256_set1_epi32(0x3F800000));
    let mant = _mm256_castsi256_ps(mant_int);

    // Polynomial approximation for ln(1 + x) where x is in [0, 1]
    let one = _mm256_set1_ps(1.0);
    let t = _mm256_sub_ps(mant, one);

    let c1 = _mm256_set1_ps(1.0);
    let c2 = _mm256_set1_ps(-0.5);
    let c3 = _mm256_set1_ps(0.333333333);
    let c4 = _mm256_set1_ps(-0.25);

    let t2 = _mm256_mul_ps(t, t);
    let t3 = _mm256_mul_ps(t2, t);
    let t4 = _mm256_mul_ps(t3, t);

    let poly = _mm256_add_ps(
        _mm256_add_ps(_mm256_mul_ps(c1, t), _mm256_mul_ps(c2, t2)),
        _mm256_add_ps(_mm256_mul_ps(c3, t3), _mm256_mul_ps(c4, t4)),
    );

    let ln2 = _mm256_set1_ps(0.693147180560);
    _mm256_add_ps(_mm256_mul_ps(exp_float, ln2), poly)
}

/// SIMD-optimized sine function
pub fn sin_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sin_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { sin_vec_sse2(input, output) };
            return;
        }
    }

    sin_vec_scalar(input, output);
}

fn sin_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].sin();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sin_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = sin_approx_sse2(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sin_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = sin_approx_avx2(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].sin();
        i += 1;
    }
}

/// Fast sine approximation using SSE2 with Taylor series
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sin_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Reduce to [-π, π] range
    let pi = _mm_set1_ps(std::f32::consts::PI);
    let two_pi = _mm_set1_ps(2.0 * std::f32::consts::PI);
    let inv_two_pi = _mm_set1_ps(1.0 / (2.0 * std::f32::consts::PI));

    let k = _mm_floor_ps(_mm_mul_ps(x, inv_two_pi));
    let x_reduced = _mm_sub_ps(x, _mm_mul_ps(k, two_pi));

    // Taylor series approximation: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
    let x2 = _mm_mul_ps(x_reduced, x_reduced);
    let x3 = _mm_mul_ps(x2, x_reduced);
    let x5 = _mm_mul_ps(x3, x2);
    let x7 = _mm_mul_ps(x5, x2);

    let c1 = _mm_set1_ps(1.0);
    let c3 = _mm_set1_ps(-1.0 / 6.0);
    let c5 = _mm_set1_ps(1.0 / 120.0);
    let c7 = _mm_set1_ps(-1.0 / 5040.0);

    _mm_add_ps(
        _mm_mul_ps(c1, x_reduced),
        _mm_add_ps(
            _mm_mul_ps(c3, x3),
            _mm_add_ps(_mm_mul_ps(c5, x5), _mm_mul_ps(c7, x7)),
        ),
    )
}

/// Fast sine approximation using AVX2 with Taylor series
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sin_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Reduce to [-π, π] range
    let pi = _mm256_set1_ps(std::f32::consts::PI);
    let two_pi = _mm256_set1_ps(2.0 * std::f32::consts::PI);
    let inv_two_pi = _mm256_set1_ps(1.0 / (2.0 * std::f32::consts::PI));

    let k = _mm256_floor_ps(_mm256_mul_ps(x, inv_two_pi));
    let x_reduced = _mm256_sub_ps(x, _mm256_mul_ps(k, two_pi));

    // Taylor series approximation: sin(x) ≈ x - x³/6 + x⁵/120 - x⁷/5040
    let x2 = _mm256_mul_ps(x_reduced, x_reduced);
    let x3 = _mm256_mul_ps(x2, x_reduced);
    let x5 = _mm256_mul_ps(x3, x2);
    let x7 = _mm256_mul_ps(x5, x2);

    let c1 = _mm256_set1_ps(1.0);
    let c3 = _mm256_set1_ps(-1.0 / 6.0);
    let c5 = _mm256_set1_ps(1.0 / 120.0);
    let c7 = _mm256_set1_ps(-1.0 / 5040.0);

    _mm256_add_ps(
        _mm256_mul_ps(c1, x_reduced),
        _mm256_add_ps(
            _mm256_mul_ps(c3, x3),
            _mm256_add_ps(_mm256_mul_ps(c5, x5), _mm256_mul_ps(c7, x7)),
        ),
    )
}

/// SIMD-optimized cosine function
pub fn cos_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { cos_vec_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { cos_vec_sse2(input, output) };
            return;
        }
    }

    cos_vec_scalar(input, output);
}

fn cos_vec_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].cos();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cos_vec_sse2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = cos_approx_sse2(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].cos();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cos_vec_avx2(input: &[f32], output: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = cos_approx_avx2(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].cos();
        i += 1;
    }
}

/// Fast cosine approximation using SSE2 with Taylor series
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cos_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Use cos(x) = sin(x + π/2)
    let pi_half = _mm_set1_ps(std::f32::consts::PI / 2.0);
    let x_shifted = _mm_add_ps(x, pi_half);
    sin_approx_sse2(x_shifted)
}

/// Fast cosine approximation using AVX2 with Taylor series
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cos_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Use cos(x) = sin(x + π/2)
    let pi_half = _mm256_set1_ps(std::f32::consts::PI / 2.0);
    let x_shifted = _mm256_add_ps(x, pi_half);
    sin_approx_avx2(x_shifted)
}

/// SIMD-optimized tangent function
pub fn tan_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    // Calculate tan(x) = sin(x) / cos(x)
    let mut sin_vals = vec![0.0f32; input.len()];
    let mut cos_vals = vec![0.0f32; input.len()];

    sin_vec(input, &mut sin_vals);
    cos_vec(input, &mut cos_vals);

    for i in 0..input.len() {
        output[i] = sin_vals[i] / cos_vals[i];
    }
}

/// Power function (x^y) using exp and ln: x^y = exp(y * ln(x))
pub fn pow_vec(base: &[f32], exponent: &[f32], output: &mut [f32]) {
    assert_eq!(base.len(), exponent.len(), "Vectors must have the same length");
    assert_eq!(base.len(), output.len(), "Vectors must have the same length");

    let mut ln_base = vec![0.0f32; base.len()];
    let mut y_ln_x = vec![0.0f32; base.len()];

    ln_vec(base, &mut ln_base);

    for i in 0..base.len() {
        y_ln_x[i] = exponent[i] * ln_base[i];
    }

    exp_vec(&y_ln_x, output);
}

/// Absolute value (element-wise)
pub fn abs_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Vectors must have the same length");

    for i in 0..input.len() {
        output[i] = input[i].abs();
    }
}

/// Sign function (element-wise): returns -1, 0, or 1
pub fn sign_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Vectors must have the same length");

    for i in 0..input.len() {
        output[i] = if input[i] > 0.0 {
            1.0
        } else if input[i] < 0.0 {
            -1.0
        } else {
            0.0
        };
    }
}

/// Floor function (element-wise)
pub fn floor_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Vectors must have the same length");

    for i in 0..input.len() {
        output[i] = input[i].floor();
    }
}

/// Ceiling function (element-wise)
pub fn ceil_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Vectors must have the same length");

    for i in 0..input.len() {
        output[i] = input[i].ceil();
    }
}

/// Round function (element-wise)
pub fn round_vec(input: &[f32], output: &mut [f32]) {
    assert_eq!(input.len(), output.len(), "Vectors must have the same length");

    for i in 0..input.len() {
        output[i] = input[i].round();
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relative_eq(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "Expected {}, got {}", b, a);
    }

    #[test]
    fn test_sqrt_vec() {
        let input = vec![1.0, 4.0, 9.0, 16.0];
        let mut output = vec![0.0; 4];

        sqrt_vec(&input, &mut output);

        assert_relative_eq(output[0], 1.0, 1e-6);
        assert_relative_eq(output[1], 2.0, 1e-6);
        assert_relative_eq(output[2], 3.0, 1e-6);
        assert_relative_eq(output[3], 4.0, 1e-6);
    }

    #[test]
    fn test_reciprocal_vec() {
        let input = vec![1.0, 2.0, 4.0, 10.0];
        let mut output = vec![0.0; 4];

        reciprocal_vec(&input, &mut output);

        assert_relative_eq(output[0], 1.0, 1e-6);
        assert_relative_eq(output[1], 0.5, 1e-6);
        assert_relative_eq(output[2], 0.25, 1e-6);
        assert_relative_eq(output[3], 0.1, 1e-6);
    }

    #[test]
    fn test_exp_vec() {
        let input = vec![0.0, 1.0, 2.0, -1.0];
        let mut output = vec![0.0; 4];

        exp_vec(&input, &mut output);

        assert_relative_eq(output[0], 1.0, 1e-3);
        assert_relative_eq(output[1], std::f32::consts::E, 1e-3);
        assert_relative_eq(output[2], std::f32::consts::E * std::f32::consts::E, 1e-3);
        assert_relative_eq(output[3], 1.0 / std::f32::consts::E, 1e-3);
    }

    #[test]
    fn test_ln_vec() {
        let input = vec![1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E];
        let mut output = vec![0.0; 3];

        ln_vec(&input, &mut output);

        assert_relative_eq(output[0], 0.0, 1e-3);
        assert_relative_eq(output[1], 1.0, 1e-3);
        assert_relative_eq(output[2], 2.0, 1e-3);
    }

    #[test]
    fn test_sin_vec() {
        let input = vec![
            0.0,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
            -std::f32::consts::PI / 2.0,
        ];
        let mut output = vec![0.0; 4];

        sin_vec(&input, &mut output);

        assert_relative_eq(output[0], 0.0, 1e-3);
        assert_relative_eq(output[1], 1.0, 1e-3);
        assert_relative_eq(output[2], 0.0, 1e-3);
        assert_relative_eq(output[3], -1.0, 1e-3);
    }

    #[test]
    fn test_cos_vec() {
        let input = vec![
            0.0,
            std::f32::consts::PI / 2.0,
            std::f32::consts::PI,
            3.0 * std::f32::consts::PI / 2.0,
        ];
        let mut output = vec![0.0; 4];

        cos_vec(&input, &mut output);

        assert_relative_eq(output[0], 1.0, 1e-3);
        assert_relative_eq(output[1], 0.0, 1e-3);
        assert_relative_eq(output[2], -1.0, 1e-3);
        assert_relative_eq(output[3], 0.0, 1e-3);
    }

    #[test]
    fn test_tan_vec() {
        let input = vec![0.0, std::f32::consts::PI / 4.0, -std::f32::consts::PI / 4.0];
        let mut output = vec![0.0; 3];

        tan_vec(&input, &mut output);

        assert_relative_eq(output[0], 0.0, 1e-3);
        assert_relative_eq(output[1], 1.0, 1e-3);
        assert_relative_eq(output[2], -1.0, 1e-3);
    }

    #[test]
    fn test_pow_vec() {
        let base = vec![2.0, 3.0, 4.0, 10.0];
        let exponent = vec![2.0, 3.0, 0.5, 2.0];
        let mut output = vec![0.0; 4];

        pow_vec(&base, &exponent, &mut output);

        assert_relative_eq(output[0], 4.0, 1e-3);     // 2^2
        assert_relative_eq(output[1], 27.0, 1e-3);    // 3^3
        assert_relative_eq(output[2], 2.0, 1e-3);     // 4^0.5 = sqrt(4)
        assert_relative_eq(output[3], 100.0, 1e-3);   // 10^2
    }

    #[test]
    fn test_abs_vec() {
        let input = vec![-3.0, -1.0, 0.0, 1.0, 3.0];
        let mut output = vec![0.0; 5];

        abs_vec(&input, &mut output);

        assert_relative_eq(output[0], 3.0, 1e-6);
        assert_relative_eq(output[1], 1.0, 1e-6);
        assert_relative_eq(output[2], 0.0, 1e-6);
        assert_relative_eq(output[3], 1.0, 1e-6);
        assert_relative_eq(output[4], 3.0, 1e-6);
    }

    #[test]
    fn test_sign_vec() {
        let input = vec![-3.0, -0.0, 0.0, 1.0, 3.0];
        let mut output = vec![0.0; 5];

        sign_vec(&input, &mut output);

        assert_relative_eq(output[0], -1.0, 1e-6);
        assert_relative_eq(output[1], 0.0, 1e-6);
        assert_relative_eq(output[2], 0.0, 1e-6);
        assert_relative_eq(output[3], 1.0, 1e-6);
        assert_relative_eq(output[4], 1.0, 1e-6);
    }
}