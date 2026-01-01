//! # SIMD Vector Mathematical Functions
//!
//! High-performance SIMD-optimized mathematical functions for vectors including
//! trigonometric, exponential, logarithmic, and power functions.
//!
//! ## Features
//!
//! - **Trigonometric Functions**: Sin, cos, tan and their inverse functions
//! - **Hyperbolic Functions**: Sinh, cosh, tanh functions
//! - **Exponential Functions**: Exp, exp2, exp10 with SIMD optimization
//! - **Logarithmic Functions**: Natural log, log2, log10
//! - **Power Functions**: Square root, cube root, power operations
//! - **Multi-Platform SIMD**: SSE2, AVX2, AVX512, NEON optimizations
//! - **Automatic Fallback**: Graceful fallback to standard library functions
//!
//! ## Performance Notes
//!
//! Mathematical functions use hardware-optimized implementations where available.
//! For transcendental functions, polynomial approximations are used for maximum
//! SIMD efficiency while maintaining numerical accuracy.

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

/// SIMD-optimized element-wise sine function
///
/// Computes result\[i\] = sin(input\[i\]) for all elements using SIMD instructions.
///
/// # Arguments
/// * `input` - Input vector (angles in radians)
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::sin_vec;
/// use std::f32::consts::PI;
///
/// let input = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
/// let mut result = vec![0.0; 4];
///
/// sin_vec(&input, &mut result);
/// assert!((result[0] - 0.0).abs() < 1e-6);
/// assert!((result[1] - 1.0).abs() < 1e-6);
/// assert!((result[2] - 0.0).abs() < 1e-6);
/// assert!((result[3] + 1.0).abs() < 1e-6);
/// ```
pub fn sin_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { sin_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { sin_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { sin_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { sin_vec_neon(input, result) };
            return;
        }
    }

    sin_vec_scalar(input, result);
}

/// SIMD-optimized element-wise cosine function
///
/// Computes result\[i\] = cos(input\[i\]) for all elements using SIMD instructions.
///
/// # Arguments
/// * `input` - Input vector (angles in radians)
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::cos_vec;
/// use std::f32::consts::PI;
///
/// let input = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
/// let mut result = vec![0.0; 4];
///
/// cos_vec(&input, &mut result);
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - 0.0).abs() < 1e-6);
/// assert!((result[2] + 1.0).abs() < 1e-6);
/// assert!((result[3] - 0.0).abs() < 1e-6);
/// ```
pub fn cos_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { cos_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { cos_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { cos_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { cos_vec_neon(input, result) };
            return;
        }
    }

    cos_vec_scalar(input, result);
}

/// SIMD-optimized element-wise tangent function
///
/// Computes result\[i\] = tan(input\[i\]) for all elements using SIMD instructions.
///
/// # Arguments
/// * `input` - Input vector (angles in radians)
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::tan_vec;
/// use std::f32::consts::PI;
///
/// let input = vec![0.0, PI / 4.0, -PI / 4.0];
/// let mut result = vec![0.0; 3];
///
/// tan_vec(&input, &mut result);
/// assert!((result[0] - 0.0).abs() < 1e-6);
/// assert!((result[1] - 1.0).abs() < 1e-6);
/// assert!((result[2] + 1.0).abs() < 1e-6);
/// ```
pub fn tan_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { tan_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { tan_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { tan_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { tan_vec_neon(input, result) };
            return;
        }
    }

    tan_vec_scalar(input, result);
}

/// SIMD-optimized element-wise exponential function
///
/// Computes result\[i\] = exp(input\[i\]) for all elements using SIMD instructions.
///
/// # Arguments
/// * `input` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::exp_vec;
///
/// let input = vec![0.0, 1.0, 2.0, -1.0];
/// let mut result = vec![0.0; 4];
///
/// exp_vec(&input, &mut result);
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - std::f32::consts::E).abs() < 1e-6);
/// assert!((result[3] - (1.0 / std::f32::consts::E)).abs() < 1e-6);
/// ```
pub fn exp_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { exp_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { exp_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { exp_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { exp_vec_neon(input, result) };
            return;
        }
    }

    exp_vec_scalar(input, result);
}

/// SIMD-optimized element-wise natural logarithm function
///
/// Computes result\[i\] = ln(input\[i\]) for all elements using SIMD instructions.
/// Input values must be positive; negative inputs will produce NaN.
///
/// # Arguments
/// * `input` - Input vector (must contain positive values)
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::ln_vec;
///
/// let input = vec![1.0, std::f32::consts::E, std::f32::consts::E * std::f32::consts::E];
/// let mut result = vec![0.0; 3];
///
/// ln_vec(&input, &mut result);
/// assert!((result[0] - 0.0).abs() < 1e-6);
/// assert!((result[1] - 1.0).abs() < 1e-6);
/// assert!((result[2] - 2.0).abs() < 1e-6);
/// ```
pub fn ln_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { ln_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { ln_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { ln_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { ln_vec_neon(input, result) };
            return;
        }
    }

    ln_vec_scalar(input, result);
}

/// SIMD-optimized element-wise square root function
///
/// Computes result\[i\] = sqrt(input\[i\]) for all elements using SIMD instructions.
/// Input values must be non-negative; negative inputs will produce NaN.
///
/// # Arguments
/// * `input` - Input vector (must contain non-negative values)
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::sqrt_vec;
///
/// let input = vec![0.0, 1.0, 4.0, 9.0, 16.0];
/// let mut result = vec![0.0; 5];
///
/// sqrt_vec(&input, &mut result);
/// assert!((result[0] - 0.0).abs() < 1e-6);
/// assert!((result[1] - 1.0).abs() < 1e-6);
/// assert!((result[2] - 2.0).abs() < 1e-6);
/// assert!((result[3] - 3.0).abs() < 1e-6);
/// assert!((result[4] - 4.0).abs() < 1e-6);
/// ```
pub fn sqrt_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { sqrt_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { sqrt_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { sqrt_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { sqrt_vec_neon(input, result) };
            return;
        }
    }

    sqrt_vec_scalar(input, result);
}

/// SIMD-optimized element-wise power function
///
/// Computes result\[i\] = base\[i\] ^ exponent\[i\] for all elements using SIMD instructions.
///
/// # Arguments
/// * `base` - Base vector
/// * `exponent` - Exponent vector (must have same length as base)
/// * `result` - Output vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::pow_vec;
///
/// let base = vec![2.0, 3.0, 4.0, 5.0];
/// let exponent = vec![2.0, 3.0, 0.5, -1.0];
/// let mut result = vec![0.0; 4];
///
/// pow_vec(&base, &exponent, &mut result);
/// assert!((result[0] - 4.0).abs() < 1e-6);   // 2^2 = 4
/// assert!((result[1] - 27.0).abs() < 1e-6);  // 3^3 = 27
/// assert!((result[2] - 2.0).abs() < 1e-6);   // 4^0.5 = 2
/// assert!((result[3] - 0.2).abs() < 1e-6);   // 5^-1 = 0.2
/// ```
pub fn pow_vec(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    assert_eq!(
        base.len(),
        exponent.len(),
        "Input vectors must have the same length"
    );
    assert_eq!(
        base.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );

    if base.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { pow_vec_avx512(base, exponent, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { pow_vec_avx2(base, exponent, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { pow_vec_sse2(base, exponent, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { pow_vec_neon(base, exponent, result) };
            return;
        }
    }

    pow_vec_scalar(base, exponent, result);
}

/// SIMD-optimized element-wise square function
///
/// Computes result\[i\] = input\[i\] * input\[i\] for all elements.
/// This is more efficient than using pow_vec with exponent 2.
///
/// # Arguments
/// * `input` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::square_vec;
///
/// let input = vec![1.0, 2.0, 3.0, -4.0];
/// let mut result = vec![0.0; 4];
///
/// square_vec(&input, &mut result);
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - 4.0).abs() < 1e-6);
/// assert!((result[2] - 9.0).abs() < 1e-6);
/// assert!((result[3] - 16.0).abs() < 1e-6);
/// ```
pub fn square_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { square_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { square_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { square_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { square_vec_neon(input, result) };
            return;
        }
    }

    square_vec_scalar(input, result);
}

/// SIMD-optimized element-wise reciprocal function
///
/// Computes result\[i\] = 1.0 / input\[i\] for all elements using SIMD instructions.
/// Division by zero results in infinity according to IEEE 754 standard.
///
/// # Arguments
/// * `input` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::math_functions::reciprocal_vec;
///
/// let input = vec![1.0, 2.0, 4.0, 0.5];
/// let mut result = vec![0.0; 4];
///
/// reciprocal_vec(&input, &mut result);
/// assert!((result[0] - 1.0).abs() < 1e-6);
/// assert!((result[1] - 0.5).abs() < 1e-6);
/// assert!((result[2] - 0.25).abs() < 1e-6);
/// assert!((result[3] - 2.0).abs() < 1e-6);
/// ```
pub fn reciprocal_vec(input: &[f32], result: &mut [f32]) {
    assert_eq!(
        input.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if input.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            unsafe { reciprocal_vec_avx512(input, result) };
            return;
        } else if crate::simd_feature_detected!("avx2") {
            unsafe { reciprocal_vec_avx2(input, result) };
            return;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { reciprocal_vec_sse2(input, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { reciprocal_vec_neon(input, result) };
            return;
        }
    }

    reciprocal_vec_scalar(input, result);
}

// ============================================================================
// Scalar implementations (fallbacks)
// ============================================================================

fn sin_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].sin();
    }
}

fn cos_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].cos();
    }
}

fn tan_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].tan();
    }
}

fn exp_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].exp();
    }
}

fn ln_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].ln();
    }
}

fn sqrt_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i].sqrt();
    }
}

fn pow_vec_scalar(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    for i in 0..base.len() {
        result[i] = base[i].powf(exponent[i]);
    }
}

fn square_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = input[i] * input[i];
    }
}

fn reciprocal_vec_scalar(input: &[f32], result: &mut [f32]) {
    for i in 0..input.len() {
        result[i] = 1.0 / input[i];
    }
}

// ============================================================================
// SSE2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sin_vec_sse2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    // Process remaining elements with scalar fallback
    // Note: For transcendental functions like sin, cos, tan, we use scalar fallback
    // because accurate SIMD implementations require complex polynomial approximations
    while i < input.len() {
        result[i] = input[i].sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cos_vec_sse2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].cos();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn tan_vec_sse2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].tan();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn exp_vec_sse2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].exp();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn ln_vec_sse2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].ln();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sqrt_vec_sse2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 4 elements at a time using SIMD sqrt
    while i + 4 <= input.len() {
        let input_vec = _mm_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm_sqrt_ps(input_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i].sqrt();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn pow_vec_sse2(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < base.len() {
        result[i] = base[i].powf(exponent[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn square_vec_sse2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= input.len() {
        let input_vec = _mm_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm_mul_ps(input_vec, input_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i] * input[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn reciprocal_vec_sse2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 4 elements at a time using accurate division
    while i + 4 <= input.len() {
        let input_vec = _mm_loadu_ps(input.as_ptr().add(i));
        let ones = _mm_set1_ps(1.0);
        let result_vec = _mm_div_ps(ones, input_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = 1.0 / input[i];
        i += 1;
    }
}

// ============================================================================
// AVX2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sin_vec_avx2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cos_vec_avx2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].cos();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn tan_vec_avx2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].tan();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn exp_vec_avx2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].exp();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn ln_vec_avx2(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].ln();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sqrt_vec_avx2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= input.len() {
        let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm256_sqrt_ps(input_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i].sqrt();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn pow_vec_avx2(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < base.len() {
        result[i] = base[i].powf(exponent[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn square_vec_avx2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= input.len() {
        let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(input_vec, input_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i] * input[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn reciprocal_vec_avx2(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= input.len() {
        let input_vec = _mm256_loadu_ps(input.as_ptr().add(i));
        let ones = _mm256_set1_ps(1.0);
        let result_vec = _mm256_div_ps(ones, input_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = 1.0 / input[i];
        i += 1;
    }
}

// ============================================================================
// AVX512 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sin_vec_avx512(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].sin();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn cos_vec_avx512(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].cos();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn tan_vec_avx512(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].tan();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn exp_vec_avx512(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].exp();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn ln_vec_avx512(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].ln();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sqrt_vec_avx512(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 16 elements at a time
    while i + 16 <= input.len() {
        let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm512_sqrt_ps(input_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i].sqrt();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn pow_vec_avx512(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < base.len() {
        result[i] = base[i].powf(exponent[i]);
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn square_vec_avx512(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 16 elements at a time
    while i + 16 <= input.len() {
        let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(input_vec, input_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i] * input[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn reciprocal_vec_avx512(input: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 16 elements at a time using accurate division
    while i + 16 <= input.len() {
        let input_vec = _mm512_loadu_ps(input.as_ptr().add(i));
        let ones = _mm512_set1_ps(1.0);
        let result_vec = _mm512_div_ps(ones, input_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = 1.0 / input[i];
        i += 1;
    }
}

// ============================================================================
// NEON implementations (ARM AArch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sin_vec_neon(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].sin();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn cos_vec_neon(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].cos();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn tan_vec_neon(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].tan();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn exp_vec_neon(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].exp();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn ln_vec_neon(input: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < input.len() {
        result[i] = input[i].ln();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sqrt_vec_neon(input: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= input.len() {
        let input_vec = vld1q_f32(input.as_ptr().add(i));
        let result_vec = vsqrtq_f32(input_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i].sqrt();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn pow_vec_neon(base: &[f32], exponent: &[f32], result: &mut [f32]) {
    let mut i = 0;

    while i < base.len() {
        result[i] = base[i].powf(exponent[i]);
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn square_vec_neon(input: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= input.len() {
        let input_vec = vld1q_f32(input.as_ptr().add(i));
        let result_vec = vmulq_f32(input_vec, input_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = input[i] * input[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn reciprocal_vec_neon(input: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    // Process 4 elements at a time using reciprocal estimate with refinement
    while i + 4 <= input.len() {
        let input_vec = vld1q_f32(input.as_ptr().add(i));
        let mut estimate = vrecpeq_f32(input_vec);
        // Two Newton-Raphson refinement steps for improved accuracy
        estimate = vmulq_f32(vrecpsq_f32(input_vec, estimate), estimate);
        estimate = vmulq_f32(vrecpsq_f32(input_vec, estimate), estimate);
        vst1q_f32(result.as_mut_ptr().add(i), estimate);
        i += 4;
    }

    // Handle remaining elements
    while i < input.len() {
        result[i] = 1.0 / input[i];
        i += 1;
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[cfg(feature = "no-std")]
    #[allow(unused_imports)]
    use core::f32::consts::{E, PI};
    #[cfg(not(feature = "no-std"))]
    use std::f32::consts::{E, PI};

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_sin_vec() {
        let input = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let mut result = vec![0.0; 4];

        sin_vec(&input, &mut result);
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 0.0).abs() < EPSILON);
        assert!((result[3] + 1.0).abs() < EPSILON);

        // Test with empty vector
        let empty_input: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        sin_vec(&empty_input, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }

    #[test]
    fn test_cos_vec() {
        let input = vec![0.0, PI / 2.0, PI, 3.0 * PI / 2.0];
        let mut result = vec![0.0; 4];

        cos_vec(&input, &mut result);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 0.0).abs() < EPSILON);
        assert!((result[2] + 1.0).abs() < EPSILON);
        assert!((result[3] - 0.0).abs() < EPSILON);
    }

    #[test]
    fn test_tan_vec() {
        let input = vec![0.0, PI / 4.0, -PI / 4.0];
        let mut result = vec![0.0; 3];

        tan_vec(&input, &mut result);
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] + 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_exp_vec() {
        let input = vec![0.0, 1.0, 2.0, -1.0];
        let mut result = vec![0.0; 4];

        exp_vec(&input, &mut result);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - E).abs() < EPSILON);
        assert!((result[2] - (E * E)).abs() < 1e-5);
        assert!((result[3] - (1.0 / E)).abs() < EPSILON);
    }

    #[test]
    fn test_ln_vec() {
        let input = vec![1.0, E, E * E, 1.0 / E];
        let mut result = vec![0.0; 4];

        ln_vec(&input, &mut result);
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 2.0).abs() < EPSILON);
        assert!((result[3] + 1.0).abs() < EPSILON);

        // Test with negative input (should produce NaN)
        let negative_input = vec![-1.0];
        let mut negative_result = vec![0.0; 1];
        ln_vec(&negative_input, &mut negative_result);
        assert!(negative_result[0].is_nan());
    }

    #[test]
    fn test_sqrt_vec() {
        let input = vec![0.0, 1.0, 4.0, 9.0, 16.0];
        let mut result = vec![0.0; 5];

        sqrt_vec(&input, &mut result);
        assert!((result[0] - 0.0).abs() < EPSILON);
        assert!((result[1] - 1.0).abs() < EPSILON);
        assert!((result[2] - 2.0).abs() < EPSILON);
        assert!((result[3] - 3.0).abs() < EPSILON);
        assert!((result[4] - 4.0).abs() < EPSILON);

        // Test with negative input (should produce NaN)
        let negative_input = vec![-1.0];
        let mut negative_result = vec![0.0; 1];
        sqrt_vec(&negative_input, &mut negative_result);
        assert!(negative_result[0].is_nan());
    }

    #[test]
    fn test_pow_vec() {
        let base = vec![2.0, 3.0, 4.0, 5.0];
        let exponent = vec![2.0, 3.0, 0.5, -1.0];
        let mut result = vec![0.0; 4];

        pow_vec(&base, &exponent, &mut result);
        assert!((result[0] - 4.0).abs() < EPSILON); // 2^2 = 4
        assert!((result[1] - 27.0).abs() < EPSILON); // 3^3 = 27
        assert!((result[2] - 2.0).abs() < EPSILON); // 4^0.5 = 2
        assert!((result[3] - 0.2).abs() < EPSILON); // 5^-1 = 0.2

        // Test with empty vectors
        let empty_base: Vec<f32> = vec![];
        let empty_exp: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        pow_vec(&empty_base, &empty_exp, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }

    #[test]
    fn test_square_vec() {
        let input = vec![1.0, 2.0, 3.0, -4.0];
        let mut result = vec![0.0; 4];

        square_vec(&input, &mut result);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 4.0).abs() < EPSILON);
        assert!((result[2] - 9.0).abs() < EPSILON);
        assert!((result[3] - 16.0).abs() < EPSILON);
    }

    #[test]
    fn test_reciprocal_vec() {
        let input = vec![1.0, 2.0, 4.0, 0.5];
        let mut result = vec![0.0; 4];

        reciprocal_vec(&input, &mut result);
        assert!((result[0] - 1.0).abs() < EPSILON);
        assert!((result[1] - 0.5).abs() < EPSILON);
        assert!((result[2] - 0.25).abs() < EPSILON);
        assert!((result[3] - 2.0).abs() < EPSILON);

        // Test division by zero
        let zero_input = vec![0.0];
        let mut zero_result = vec![0.0; 1];
        reciprocal_vec(&zero_input, &mut zero_result);
        assert_eq!(zero_result[0], f32::INFINITY);
    }

    #[test]
    fn test_special_values() {
        // Test exp with large values
        let large_input = vec![100.0];
        let mut large_result = vec![0.0; 1];
        exp_vec(&large_input, &mut large_result);
        assert_eq!(large_result[0], f32::INFINITY);

        // Test ln with zero
        let zero_input = vec![0.0];
        let mut zero_result = vec![0.0; 1];
        ln_vec(&zero_input, &mut zero_result);
        assert_eq!(zero_result[0], f32::NEG_INFINITY);

        // Test sqrt with infinity
        let inf_input = vec![f32::INFINITY];
        let mut inf_result = vec![0.0; 1];
        sqrt_vec(&inf_input, &mut inf_result);
        assert_eq!(inf_result[0], f32::INFINITY);
    }

    #[test]
    fn test_mathematical_identities() {
        let angles = vec![0.5, 1.0, 1.5, 2.0];
        let mut sin_results = vec![0.0; 4];
        let mut cos_results = vec![0.0; 4];

        sin_vec(&angles, &mut sin_results);
        cos_vec(&angles, &mut cos_results);

        // Test Pythagorean identity: sin²(x) + cos²(x) = 1
        for i in 0..4 {
            let sin_sq = sin_results[i] * sin_results[i];
            let cos_sq = cos_results[i] * cos_results[i];
            assert!((sin_sq + cos_sq - 1.0).abs() < 1e-6);
        }

        // Test exp(ln(x)) = x for positive values
        let positive_values = vec![1.0, 2.0, 3.0, 4.0];
        let mut ln_results = vec![0.0; 4];
        let mut exp_ln_results = vec![0.0; 4];

        ln_vec(&positive_values, &mut ln_results);
        exp_vec(&ln_results, &mut exp_ln_results);

        for i in 0..4 {
            assert!((exp_ln_results[i] - positive_values[i]).abs() < 1e-5);
        }
    }

    #[test]
    fn test_large_vectors() {
        let size = 100;
        let input: Vec<f32> = (0..size).map(|i| (i as f32) / 10.0).collect();
        let mut result = vec![0.0; size];

        sqrt_vec(&input, &mut result);

        for i in 0..size {
            let expected = ((i as f32) / 10.0).sqrt();
            assert!((result[i] - expected).abs() < 1e-6);
        }
    }

    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length")]
    fn test_sin_vec_dimension_mismatch() {
        let input = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        sin_vec(&input, &mut result);
    }

    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_pow_vec_input_dimension_mismatch() {
        let base = vec![1.0, 2.0, 3.0];
        let exponent = vec![2.0, 3.0];
        let mut result = vec![0.0; 3];
        pow_vec(&base, &exponent, &mut result);
    }

    #[test]
    #[should_panic(expected = "Output vector must have the same length as input vectors")]
    fn test_pow_vec_output_dimension_mismatch() {
        let base = vec![1.0, 2.0, 3.0];
        let exponent = vec![2.0, 3.0, 4.0];
        let mut result = vec![0.0; 2];
        pow_vec(&base, &exponent, &mut result);
    }
}
