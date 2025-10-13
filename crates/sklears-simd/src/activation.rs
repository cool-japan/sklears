//! SIMD-optimized activation functions for machine learning
//!
//! This module provides high-performance implementations of activation functions
//! commonly used in neural networks, optimized using SIMD instructions for
//! maximum throughput. All functions include both forward and derivative variants
//! for efficient backpropagation.

use crate::vector::sum;
use scirs2_autograd::ndarray::{Array1, Array2};

#[cfg(feature = "no-std")]
use alloc::{format, vec};

/// SIMD-optimized sigmoid activation function
pub fn sigmoid(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { sigmoid_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { sigmoid_sse2(input, output) };
            return;
        }
    }

    sigmoid_scalar(input, output);
}

fn sigmoid_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sigmoid_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let one = _mm_set1_ps(1.0);

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));

        // Approximate exp(-x) using polynomial approximation for better SIMD performance
        let neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
        let exp_neg_x = exp_approx_sse2(neg_x);

        let one_plus_exp = _mm_add_ps(one, exp_neg_x);
        let result = _mm_div_ps(one, one_plus_exp);

        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sigmoid_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let one = _mm256_set1_ps(1.0);

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));

        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg_x = exp_approx_avx2(neg_x);

        let one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        let result = _mm256_div_ps(one, one_plus_exp);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = 1.0 / (1.0 + (-input[i]).exp());
        i += 1;
    }
}

/// Fast exponential approximation for SSE2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn exp_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use core::arch::x86_64::*;

    // Clamp input to reasonable range to avoid overflow
    let min_val = _mm_set1_ps(-10.0);
    let max_val = _mm_set1_ps(10.0);
    let clamped = _mm_max_ps(min_val, _mm_min_ps(max_val, x));

    // Simple polynomial approximation: exp(x) ≈ 1 + x + x²/2 + x³/6
    let one = _mm_set1_ps(1.0);
    let half = _mm_set1_ps(0.5);
    let sixth = _mm_set1_ps(1.0 / 6.0);

    let x2 = _mm_mul_ps(clamped, clamped);
    let x3 = _mm_mul_ps(x2, clamped);

    let term1 = one;
    let term2 = clamped;
    let term3 = _mm_mul_ps(x2, half);
    let term4 = _mm_mul_ps(x3, sixth);

    _mm_add_ps(_mm_add_ps(term1, term2), _mm_add_ps(term3, term4))
}

/// Fast exponential approximation for AVX2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn exp_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;

    let min_val = _mm256_set1_ps(-10.0);
    let max_val = _mm256_set1_ps(10.0);
    let clamped = _mm256_max_ps(min_val, _mm256_min_ps(max_val, x));

    let one = _mm256_set1_ps(1.0);
    let half = _mm256_set1_ps(0.5);
    let sixth = _mm256_set1_ps(1.0 / 6.0);

    let x2 = _mm256_mul_ps(clamped, clamped);
    let x3 = _mm256_mul_ps(x2, clamped);

    let term1 = one;
    let term2 = clamped;
    let term3 = _mm256_mul_ps(x2, half);
    let term4 = _mm256_mul_ps(x3, sixth);

    _mm256_add_ps(_mm256_add_ps(term1, term2), _mm256_add_ps(term3, term4))
}

/// SIMD-optimized ReLU activation function
pub fn relu(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { relu_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { relu_sse2(input, output) };
            return;
        }
    }

    relu_scalar(input, output);
}

fn relu_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].max(0.0);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn relu_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let zero = _mm_setzero_ps();

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = _mm_max_ps(x, zero);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].max(0.0);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let zero = _mm256_setzero_ps();

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = _mm256_max_ps(x, zero);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].max(0.0);
        i += 1;
    }
}

/// SIMD-optimized Leaky ReLU activation function
pub fn leaky_relu(input: &[f32], output: &mut [f32], alpha: f32) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { leaky_relu_avx2(input, output, alpha) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { leaky_relu_sse2(input, output, alpha) };
            return;
        }
    }

    leaky_relu_scalar(input, output, alpha);
}

fn leaky_relu_scalar(input: &[f32], output: &mut [f32], alpha: f32) {
    for i in 0..input.len() {
        output[i] = if input[i] > 0.0 {
            input[i]
        } else {
            alpha * input[i]
        };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn leaky_relu_sse2(input: &[f32], output: &mut [f32], alpha: f32) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let zero = _mm_setzero_ps();
    let alpha_vec = _mm_set1_ps(alpha);

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let mask = _mm_cmpgt_ps(x, zero);
        let positive = x;
        let negative = _mm_mul_ps(x, alpha_vec);
        let result = _mm_blendv_ps(negative, positive, mask);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = if input[i] > 0.0 {
            input[i]
        } else {
            alpha * input[i]
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn leaky_relu_avx2(input: &[f32], output: &mut [f32], alpha: f32) {
    use core::arch::x86_64::*;

    let mut i = 0;
    let zero = _mm256_setzero_ps();
    let alpha_vec = _mm256_set1_ps(alpha);

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
        let positive = x;
        let negative = _mm256_mul_ps(x, alpha_vec);
        let result = _mm256_blendv_ps(negative, positive, mask);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = if input[i] > 0.0 {
            input[i]
        } else {
            alpha * input[i]
        };
        i += 1;
    }
}

/// SIMD-optimized tanh activation function
pub fn tanh_activation(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { tanh_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { tanh_sse2(input, output) };
            return;
        }
    }

    tanh_scalar(input, output);
}

fn tanh_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = input[i].tanh();
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn tanh_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let result = tanh_approx_sse2(x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = input[i].tanh();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn tanh_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let result = tanh_approx_avx2(x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = input[i].tanh();
        i += 1;
    }
}

/// Fast tanh approximation for SSE2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn tanh_approx_sse2(x: std::arch::x86_64::__m128) -> std::arch::x86_64::__m128 {
    use core::arch::x86_64::*;

    // Clamp input
    let min_val = _mm_set1_ps(-5.0);
    let max_val = _mm_set1_ps(5.0);
    let clamped = _mm_max_ps(min_val, _mm_min_ps(max_val, x));

    // Use rational approximation: tanh(x) ≈ x * (1 - x²/3)
    let x2 = _mm_mul_ps(clamped, clamped);
    let third = _mm_set1_ps(1.0 / 3.0);
    let one = _mm_set1_ps(1.0);

    let term = _mm_sub_ps(one, _mm_mul_ps(x2, third));
    _mm_mul_ps(clamped, term)
}

/// Fast tanh approximation for AVX2
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn tanh_approx_avx2(x: std::arch::x86_64::__m256) -> std::arch::x86_64::__m256 {
    use core::arch::x86_64::*;

    let min_val = _mm256_set1_ps(-5.0);
    let max_val = _mm256_set1_ps(5.0);
    let clamped = _mm256_max_ps(min_val, _mm256_min_ps(max_val, x));

    let x2 = _mm256_mul_ps(clamped, clamped);
    let third = _mm256_set1_ps(1.0 / 3.0);
    let one = _mm256_set1_ps(1.0);

    let term = _mm256_sub_ps(one, _mm256_mul_ps(x2, third));
    _mm256_mul_ps(clamped, term)
}

/// SIMD-optimized softmax activation function
pub fn softmax(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    // Find max for numerical stability
    let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp(x - max)
    let mut exp_values = vec![0.0; input.len()];
    for i in 0..input.len() {
        exp_values[i] = (input[i] - max_val).exp();
    }

    // Compute sum of exponentials
    let exp_sum = sum(&exp_values);

    // Normalize
    for i in 0..input.len() {
        output[i] = exp_values[i] / exp_sum;
    }
}

/// SIMD-optimized ELU (Exponential Linear Unit) activation function
pub fn elu(input: &[f32], output: &mut [f32], alpha: f32) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { elu_avx2(input, output, alpha) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { elu_sse2(input, output, alpha) };
            return;
        }
    }

    elu_scalar(input, output, alpha);
}

fn elu_scalar(input: &[f32], output: &mut [f32], alpha: f32) {
    for i in 0..input.len() {
        output[i] = if input[i] >= 0.0 {
            input[i]
        } else {
            alpha * (input[i].exp() - 1.0)
        };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn elu_sse2(input: &[f32], output: &mut [f32], alpha: f32) {
    use core::arch::x86_64::*;

    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let alpha_vec = _mm_set1_ps(alpha);
    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let mask = _mm_cmpge_ps(x, zero);

        let positive = x;
        let exp_x = exp_approx_sse2(x);
        let negative = _mm_mul_ps(alpha_vec, _mm_sub_ps(exp_x, one));

        let result = _mm_blendv_ps(negative, positive, mask);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = if input[i] >= 0.0 {
            input[i]
        } else {
            alpha * (input[i].exp() - 1.0)
        };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn elu_avx2(input: &[f32], output: &mut [f32], alpha: f32) {
    use core::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let alpha_vec = _mm256_set1_ps(alpha);
    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let mask = _mm256_cmp_ps(x, zero, _CMP_GE_OQ);

        let positive = x;
        let exp_x = exp_approx_avx2(x);
        let negative = _mm256_mul_ps(alpha_vec, _mm256_sub_ps(exp_x, one));

        let result = _mm256_blendv_ps(negative, positive, mask);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = if input[i] >= 0.0 {
            input[i]
        } else {
            alpha * (input[i].exp() - 1.0)
        };
        i += 1;
    }
}

/// SIMD-optimized Swish (SiLU) activation function: x * sigmoid(x)
pub fn swish(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { swish_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { swish_sse2(input, output) };
            return;
        }
    }

    swish_scalar(input, output);
}

fn swish_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        let sigmoid_x = 1.0 / (1.0 + (-input[i]).exp());
        output[i] = input[i] * sigmoid_x;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn swish_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));

        let neg_x = _mm_sub_ps(_mm_setzero_ps(), x);
        let exp_neg_x = exp_approx_sse2(neg_x);
        let one_plus_exp = _mm_add_ps(one, exp_neg_x);
        let sigmoid_x = _mm_div_ps(one, one_plus_exp);

        let result = _mm_mul_ps(x, sigmoid_x);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        let sigmoid_x = 1.0 / (1.0 + (-input[i]).exp());
        output[i] = input[i] * sigmoid_x;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn swish_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let one = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));

        let neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
        let exp_neg_x = exp_approx_avx2(neg_x);
        let one_plus_exp = _mm256_add_ps(one, exp_neg_x);
        let sigmoid_x = _mm256_div_ps(one, one_plus_exp);

        let result = _mm256_mul_ps(x, sigmoid_x);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        let sigmoid_x = 1.0 / (1.0 + (-input[i]).exp());
        output[i] = input[i] * sigmoid_x;
        i += 1;
    }
}

/// SIMD-optimized GELU (Gaussian Error Linear Unit) activation function
pub fn gelu(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { gelu_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { gelu_sse2(input, output) };
            return;
        }
    }

    gelu_scalar(input, output);
}

fn gelu_scalar(input: &[f32], output: &mut [f32]) {
    const SQRT_2_PI: f32 = 0.797_884_6; // sqrt(2/π)
    for i in 0..input.len() {
        let x = input[i];
        // GELU approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))
        let x_cubed = x * x * x;
        let inner = SQRT_2_PI * (x + 0.044715 * x_cubed);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn gelu_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let sqrt_2_pi = _mm_set1_ps(0.7978845608028654);
    let coeff = _mm_set1_ps(0.044715);
    let half = _mm_set1_ps(0.5);
    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));

        let x2 = _mm_mul_ps(x, x);
        let x3 = _mm_mul_ps(x2, x);
        let coeff_x3 = _mm_mul_ps(coeff, x3);
        let inner_term = _mm_add_ps(x, coeff_x3);
        let scaled_inner = _mm_mul_ps(sqrt_2_pi, inner_term);

        let tanh_result = tanh_approx_sse2(scaled_inner);
        let one_plus_tanh = _mm_add_ps(one, tanh_result);
        let result = _mm_mul_ps(_mm_mul_ps(half, x), one_plus_tanh);

        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        let x = input[i];
        let x_cubed = x * x * x;
        let inner = 0.7978845608028654 * (x + 0.044715 * x_cubed);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn gelu_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let sqrt_2_pi = _mm256_set1_ps(0.7978845608028654);
    let coeff = _mm256_set1_ps(0.044715);
    let half = _mm256_set1_ps(0.5);
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));

        let x2 = _mm256_mul_ps(x, x);
        let x3 = _mm256_mul_ps(x2, x);
        let coeff_x3 = _mm256_mul_ps(coeff, x3);
        let inner_term = _mm256_add_ps(x, coeff_x3);
        let scaled_inner = _mm256_mul_ps(sqrt_2_pi, inner_term);

        let tanh_result = tanh_approx_avx2(scaled_inner);
        let one_plus_tanh = _mm256_add_ps(one, tanh_result);
        let result = _mm256_mul_ps(_mm256_mul_ps(half, x), one_plus_tanh);

        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        let x = input[i];
        let x_cubed = x * x * x;
        let inner = 0.7978845608028654 * (x + 0.044715 * x_cubed);
        output[i] = 0.5 * x * (1.0 + inner.tanh());
        i += 1;
    }
}

// ===== DERIVATIVE FUNCTIONS FOR BACKPROPAGATION =====

/// SIMD-optimized sigmoid derivative
pub fn sigmoid_derivative(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    // Compute sigmoid first, then derivative: sigmoid(x) * (1 - sigmoid(x))
    let mut sigmoid_vals = vec![0.0; input.len()];
    sigmoid(input, &mut sigmoid_vals);

    for i in 0..input.len() {
        output[i] = sigmoid_vals[i] * (1.0 - sigmoid_vals[i]);
    }
}

/// SIMD-optimized ReLU derivative
pub fn relu_derivative(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe { relu_derivative_avx2(input, output) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { relu_derivative_sse2(input, output) };
            return;
        }
    }

    relu_derivative_scalar(input, output);
}

fn relu_derivative_scalar(input: &[f32], output: &mut [f32]) {
    for i in 0..input.len() {
        output[i] = if input[i] > 0.0 { 1.0 } else { 0.0 };
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn relu_derivative_sse2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let zero = _mm_setzero_ps();
    let one = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let x = _mm_loadu_ps(input.as_ptr().add(i));
        let mask = _mm_cmpgt_ps(x, zero);
        let result = _mm_and_ps(mask, one);
        _mm_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 4;
    }

    while i < input.len() {
        output[i] = if input[i] > 0.0 { 1.0 } else { 0.0 };
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn relu_derivative_avx2(input: &[f32], output: &mut [f32]) {
    use core::arch::x86_64::*;

    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= input.len() {
        let x = _mm256_loadu_ps(input.as_ptr().add(i));
        let mask = _mm256_cmp_ps(x, zero, _CMP_GT_OQ);
        let result = _mm256_and_ps(mask, one);
        _mm256_storeu_ps(output.as_mut_ptr().add(i), result);
        i += 8;
    }

    while i < input.len() {
        output[i] = if input[i] > 0.0 { 1.0 } else { 0.0 };
        i += 1;
    }
}

/// SIMD-optimized tanh derivative: 1 - tanh²(x)
pub fn tanh_derivative(input: &[f32], output: &mut [f32]) {
    assert_eq!(
        input.len(),
        output.len(),
        "Vectors must have the same length"
    );

    // Compute tanh first, then derivative
    let mut tanh_vals = vec![0.0; input.len()];
    tanh_activation(input, &mut tanh_vals);

    for i in 0..input.len() {
        output[i] = 1.0 - tanh_vals[i] * tanh_vals[i];
    }
}

// ===== CONVENIENT NDARRAY INTERFACES =====

/// Apply activation function to ndarray Array1
pub fn apply_activation_1d(
    input: &Array1<f32>,
    activation: ActivationFunction,
    alpha: Option<f32>,
) -> Array1<f32> {
    let mut output = Array1::zeros(input.len());
    apply_activation_slice(
        input.as_slice().unwrap(),
        output.as_slice_mut().unwrap(),
        activation,
        alpha,
    );
    output
}

/// Apply activation function to ndarray Array2 (applies to each element)
pub fn apply_activation_2d(
    input: &Array2<f32>,
    activation: ActivationFunction,
    alpha: Option<f32>,
) -> Array2<f32> {
    let mut output = Array2::zeros(input.dim());
    if let (Some(input_slice), Some(output_slice)) = (input.as_slice(), output.as_slice_mut()) {
        apply_activation_slice(input_slice, output_slice, activation, alpha);
    }
    output
}

/// Apply activation function to raw slice
pub fn apply_activation_slice(
    input: &[f32],
    output: &mut [f32],
    activation: ActivationFunction,
    alpha: Option<f32>,
) {
    match activation {
        ActivationFunction::ReLU => relu(input, output),
        ActivationFunction::Sigmoid => sigmoid(input, output),
        ActivationFunction::Tanh => tanh_activation(input, output),
        ActivationFunction::LeakyReLU => {
            let alpha_val = alpha.unwrap_or(0.01);
            leaky_relu(input, output, alpha_val);
        }
        ActivationFunction::ELU => {
            let alpha_val = alpha.unwrap_or(1.0);
            elu(input, output, alpha_val);
        }
        ActivationFunction::Swish => swish(input, output),
        ActivationFunction::GELU => gelu(input, output),
        ActivationFunction::Softmax => softmax(input, output),
    }
}

/// Enumeration of available activation functions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ActivationFunction {
    ReLU,
    Sigmoid,
    Tanh,
    LeakyReLU,
    ELU,
    Swish,
    GELU,
    Softmax,
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_sigmoid() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = vec![0.0; 5];

        sigmoid(&input, &mut output);

        assert_relative_eq!(output[0], 0.5, epsilon = 1e-3);
        assert!(output[1] > 0.7 && output[1] < 0.8);
        assert!(output[2] > 0.2 && output[2] < 0.3);
        assert!(output[3] > 0.8 && output[3] < 0.9);
        assert!(output[4] > 0.1 && output[4] < 0.2);
    }

    #[test]
    fn test_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        relu(&input, &mut output);

        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_leaky_relu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        let alpha = 0.1;

        leaky_relu(&input, &mut output, alpha);

        assert_relative_eq!(output[0], -0.2, epsilon = 1e-6);
        assert_relative_eq!(output[1], -0.1, epsilon = 1e-6);
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 2.0, epsilon = 1e-6);
    }

    #[test]
    fn test_tanh_activation() {
        let input = vec![0.0, 1.0, -1.0, 2.0, -2.0];
        let mut output = vec![0.0; 5];

        tanh_activation(&input, &mut output);

        assert_relative_eq!(output[0], 0.0, epsilon = 1e-3);
        assert!(output[1] > 0.7 && output[1] < 0.8);
        assert!(output[2] > -0.8 && output[2] < -0.7);
        assert!(output[3] > 0.9);
        assert!(output[4] < -0.9);
    }

    #[test]
    fn test_softmax() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];

        softmax(&input, &mut output);

        // Check that probabilities sum to 1
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6);

        // Check that largest input corresponds to largest output
        assert!(output[2] > output[1]);
        assert!(output[1] > output[0]);
    }

    #[test]
    fn test_elu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];
        let alpha = 1.0;

        elu(&input, &mut output, alpha);

        // For positive values, ELU should equal input
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6);
        assert_relative_eq!(output[4], 2.0, epsilon = 1e-6);

        // For negative values, ELU should be alpha * (exp(x) - 1)
        assert!(output[0] < 0.0 && output[0] > -alpha); // Should approach -alpha
        assert!(output[1] < 0.0 && output[1] > output[0]); // Less negative than output[0]
    }

    #[test]
    fn test_swish() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        swish(&input, &mut output);

        // Swish(0) should be 0
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-3);

        // For positive values, should be positive and increase with input
        assert!(output[3] > 0.0);
        assert!(output[4] > output[3]);

        // For negative values, should be negative but approaching 0
        assert!(output[0] < 0.0);
        assert!(output[1] < 0.0);
        // Note: Swish is not monotonic for very negative values
        // The minimum occurs around x ≈ -1.28, so swish(-1) < swish(-2)
    }

    #[test]
    fn test_gelu() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        gelu(&input, &mut output);

        // GELU(0) should be 0
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-3);

        // For positive values, should be positive and roughly follow input
        assert!(output[3] > 0.0);
        assert!(output[4] > output[3]);

        // GELU should be smooth and differentiable everywhere
        for &val in &output {
            assert!(!val.is_nan());
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_relu_derivative() {
        let input = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut output = vec![0.0; 5];

        relu_derivative(&input, &mut output);

        assert_relative_eq!(output[0], 0.0, epsilon = 1e-6); // negative -> 0
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6); // negative -> 0
        assert_relative_eq!(output[2], 0.0, epsilon = 1e-6); // zero -> 0
        assert_relative_eq!(output[3], 1.0, epsilon = 1e-6); // positive -> 1
        assert_relative_eq!(output[4], 1.0, epsilon = 1e-6); // positive -> 1
    }

    #[test]
    fn test_sigmoid_derivative() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        sigmoid_derivative(&input, &mut output);

        // Sigmoid derivative at 0 should be 0.25
        assert_relative_eq!(output[0], 0.25, epsilon = 1e-3);

        // All derivatives should be positive
        for &val in &output {
            assert!(val > 0.0);
        }
    }

    #[test]
    fn test_tanh_derivative() {
        let input = vec![0.0, 1.0, -1.0];
        let mut output = vec![0.0; 3];

        tanh_derivative(&input, &mut output);

        // Tanh derivative at 0 should be 1.0
        assert_relative_eq!(output[0], 1.0, epsilon = 1e-3);

        // All derivatives should be positive and <= 1
        for &val in &output {
            assert!(val > 0.0 && val <= 1.0);
        }
    }

    #[test]
    fn test_activation_function_enum() {
        let input = vec![1.0, 2.0, 3.0];
        let mut output = vec![0.0; 3];

        // Test all activation functions through the enum interface
        apply_activation_slice(&input, &mut output, ActivationFunction::ReLU, None);
        assert_eq!(output, input); // ReLU for positive inputs

        apply_activation_slice(&input, &mut output, ActivationFunction::Sigmoid, None);
        assert!(output.iter().all(|&x| x > 0.0 && x < 1.0)); // Sigmoid range

        apply_activation_slice(&input, &mut output, ActivationFunction::Softmax, None);
        let sum: f32 = output.iter().sum();
        assert_relative_eq!(sum, 1.0, epsilon = 1e-6); // Softmax sums to 1
    }

    #[test]
    fn test_ndarray_interface() {
        let input_1d = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let output_1d = apply_activation_1d(&input_1d, ActivationFunction::ReLU, None);
        assert_eq!(output_1d.as_slice().unwrap(), &[1.0, 2.0, 3.0]);

        let input_2d = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let output_2d = apply_activation_2d(&input_2d, ActivationFunction::ReLU, None);
        assert_eq!(output_2d.as_slice().unwrap(), &[1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_activation_with_alpha() {
        let input = vec![-1.0, 0.0, 1.0];
        let mut output = vec![0.0; 3];

        // Test LeakyReLU with custom alpha
        apply_activation_slice(
            &input,
            &mut output,
            ActivationFunction::LeakyReLU,
            Some(0.2),
        );
        assert_relative_eq!(output[0], -0.2, epsilon = 1e-6);
        assert_relative_eq!(output[1], 0.0, epsilon = 1e-6);
        assert_relative_eq!(output[2], 1.0, epsilon = 1e-6);

        // Test ELU with custom alpha
        apply_activation_slice(&input, &mut output, ActivationFunction::ELU, Some(2.0));
        assert!(output[0] < 0.0 && output[0] > -2.0); // Should approach -2.0 for negative inputs
    }
}
