//! # Arithmetic SIMD Vector Operations
//!
//! SIMD-optimized arithmetic operations for element-wise computations on vectors.
//! Provides high-performance implementations for basic arithmetic operations with
//! automatic platform detection and fallback support.
//!
//! ## Features
//!
//! - **Element-wise Operations**: Add, subtract, multiply, divide vectors
//! - **Fused Operations**: Fused multiply-add (FMA) for maximum performance
//! - **Scaling Operations**: Scalar-vector multiplication and division
//! - **In-place Operations**: Memory-efficient in-place modifications
//! - **Multi-Platform SIMD**: SSE2, AVX2, AVX512, NEON optimizations
//! - **Automatic Fallback**: Graceful fallback to scalar implementations
//!
//! ## Performance Notes
//!
//! All operations are optimized for modern CPU architectures and automatically
//! select the best available instruction set. FMA operations provide the highest
//! performance when supported by the hardware.

#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

/// SIMD-optimized element-wise vector addition
///
/// Computes c[i] = a[i] + b[i] for all elements using SIMD instructions
/// when available. The operation is performed in-place on the output vector.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output vector (must have same length as `a` and `b`)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::add_vec;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![5.0, 6.0, 7.0, 8.0];
/// let mut result = vec![0.0; 4];
///
/// add_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);
/// ```
pub fn add_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );

    if a.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { add_vec_avx512(a, b, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { add_vec_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { add_vec_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { add_vec_neon(a, b, result) };
            return;
        }
    }

    add_vec_scalar(a, b, result);
}

/// SIMD-optimized element-wise vector subtraction
///
/// Computes c[i] = a[i] - b[i] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector (minuend)
/// * `b` - Second input vector (subtrahend, must have same length as `a`)
/// * `result` - Output vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::subtract_vec;
///
/// let a = vec![10.0, 8.0, 6.0, 4.0];
/// let b = vec![3.0, 2.0, 1.0, 1.0];
/// let mut result = vec![0.0; 4];
///
/// subtract_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![7.0, 6.0, 5.0, 3.0]);
/// ```
pub fn subtract_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );

    if a.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { subtract_vec_avx512(a, b, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { subtract_vec_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { subtract_vec_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { subtract_vec_neon(a, b, result) };
            return;
        }
    }

    subtract_vec_scalar(a, b, result);
}

/// SIMD-optimized element-wise vector multiplication
///
/// Computes c[i] = a[i] * b[i] for all elements using SIMD instructions.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
/// * `result` - Output vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::multiply_vec;
///
/// let a = vec![2.0, 3.0, 4.0, 5.0];
/// let b = vec![3.0, 4.0, 5.0, 6.0];
/// let mut result = vec![0.0; 4];
///
/// multiply_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0]);
/// ```
pub fn multiply_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );

    if a.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { multiply_vec_avx512(a, b, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { multiply_vec_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { multiply_vec_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { multiply_vec_neon(a, b, result) };
            return;
        }
    }

    multiply_vec_scalar(a, b, result);
}

/// SIMD-optimized element-wise vector division
///
/// Computes c[i] = a[i] / b[i] for all elements using SIMD instructions.
/// Division by zero results in infinity or NaN according to IEEE 754 standard.
///
/// # Arguments
/// * `a` - First input vector (dividend)
/// * `b` - Second input vector (divisor, must have same length as `a`)
/// * `result` - Output vector (must have same length as inputs)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::divide_vec;
///
/// let a = vec![12.0, 15.0, 20.0, 25.0];
/// let b = vec![3.0, 3.0, 4.0, 5.0];
/// let mut result = vec![0.0; 4];
///
/// divide_vec(&a, &b, &mut result);
/// assert_eq!(result, vec![4.0, 5.0, 5.0, 5.0]);
/// ```
pub fn divide_vec(a: &[f32], b: &[f32], result: &mut [f32]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(
        a.len(),
        result.len(),
        "Output vector must have the same length as input vectors"
    );

    if a.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { divide_vec_avx512(a, b, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { divide_vec_avx2(a, b, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { divide_vec_sse2(a, b, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { divide_vec_neon(a, b, result) };
            return;
        }
    }

    divide_vec_scalar(a, b, result);
}

/// SIMD-optimized fused multiply-add operation
///
/// Computes a[i] = a[i] * b[i] + c[i] for all elements in-place on vector `a`.
/// This operation provides maximum performance and precision when FMA instructions
/// are available, as it performs multiplication and addition in a single step.
///
/// # Arguments
/// * `a` - Input/output vector (will be modified in-place)
/// * `b` - Multiplier vector (must have same length as `a`)
/// * `c` - Addend vector (must have same length as `a`)
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::fma;
///
/// let mut a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![2.0, 3.0, 4.0, 5.0];
/// let c = vec![1.0, 1.0, 1.0, 1.0];
///
/// fma(&mut a, &b, &c);
/// // a = a * b + c = [1*2+1, 2*3+1, 3*4+1, 4*5+1] = [3, 7, 13, 21]
/// assert_eq!(a, vec![3.0, 7.0, 13.0, 21.0]);
/// ```
pub fn fma(a: &mut [f32], b: &[f32], c: &[f32]) {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
    assert_eq!(a.len(), c.len(), "Input vectors must have the same length");

    if a.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("fma") {
            unsafe { fma_fma_intrinsic(a, b, c) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { fma_avx2(a, b, c) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { fma_sse2(a, b, c) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { fma_neon(a, b, c) };
            return;
        }
    }

    fma_scalar(a, b, c);
}

/// SIMD-optimized vector scaling (scalar multiplication)
///
/// Computes result[i] = vector[i] * scalar for all elements.
/// This is optimized for the common case of multiplying a vector by a scalar value.
///
/// # Arguments
/// * `vector` - Input vector
/// * `scalar` - Scalar value to multiply each element by
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::scale_vec;
///
/// let vector = vec![1.0, 2.0, 3.0, 4.0];
/// let mut result = vec![0.0; 4];
///
/// scale_vec(&vector, 2.5, &mut result);
/// assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);
/// ```
pub fn scale_vec(vector: &[f32], scalar: f32, result: &mut [f32]) {
    assert_eq!(
        vector.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if vector.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { scale_vec_avx512(vector, scalar, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { scale_vec_avx2(vector, scalar, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { scale_vec_sse2(vector, scalar, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { scale_vec_neon(vector, scalar, result) };
            return;
        }
    }

    scale_vec_scalar(vector, scalar, result);
}

/// SIMD-optimized in-place vector scaling
///
/// Computes vector[i] = vector[i] * scalar for all elements in-place.
/// This is more memory-efficient than the out-of-place version.
///
/// # Arguments
/// * `vector` - Input/output vector (will be modified in-place)
/// * `scalar` - Scalar value to multiply each element by
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::scale_vec_inplace;
///
/// let mut vector = vec![1.0, 2.0, 3.0, 4.0];
/// scale_vec_inplace(&mut vector, 3.0);
/// assert_eq!(vector, vec![3.0, 6.0, 9.0, 12.0]);
/// ```
pub fn scale_vec_inplace(vector: &mut [f32], scalar: f32) {
    let len = vector.len();
    if len == 0 {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe {
                let mut i = 0;
                while i + 16 <= len {
                    let vec_slice = std::slice::from_raw_parts(vector.as_ptr().add(i), 16);
                    let mut result_slice =
                        std::slice::from_raw_parts_mut(vector.as_mut_ptr().add(i), 16);
                    scale_vec_avx512(vec_slice, scalar, &mut result_slice);
                    i += 16;
                }
                while i < len {
                    vector[i] *= scalar;
                    i += 1;
                }
            }
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe {
                let mut i = 0;
                while i + 8 <= len {
                    let vec_slice = std::slice::from_raw_parts(vector.as_ptr().add(i), 8);
                    let mut result_slice =
                        std::slice::from_raw_parts_mut(vector.as_mut_ptr().add(i), 8);
                    scale_vec_avx2(vec_slice, scalar, &mut result_slice);
                    i += 8;
                }
                while i < len {
                    vector[i] *= scalar;
                    i += 1;
                }
            }
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe {
                let mut i = 0;
                while i + 4 <= len {
                    let vec_slice = std::slice::from_raw_parts(vector.as_ptr().add(i), 4);
                    let mut result_slice =
                        std::slice::from_raw_parts_mut(vector.as_mut_ptr().add(i), 4);
                    scale_vec_sse2(vec_slice, scalar, &mut result_slice);
                    i += 4;
                }
                while i < len {
                    vector[i] *= scalar;
                    i += 1;
                }
            }
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe {
                let mut i = 0;
                while i + 4 <= len {
                    let vec_slice = std::slice::from_raw_parts(vector.as_ptr().add(i), 4);
                    let result_slice =
                        std::slice::from_raw_parts_mut(vector.as_mut_ptr().add(i), 4);
                    scale_vec_neon(vec_slice, scalar, result_slice);
                    i += 4;
                }
                while i < len {
                    vector[i] *= scalar;
                    i += 1;
                }
            }
            return;
        }
    }

    for i in 0..len {
        vector[i] *= scalar;
    }
}

/// SIMD-optimized vector absolute value
///
/// Computes result[i] = |vector[i]| for all elements.
///
/// # Arguments
/// * `vector` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::abs_vec;
///
/// let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let mut result = vec![0.0; 5];
///
/// abs_vec(&vector, &mut result);
/// assert_eq!(result, vec![2.0, 1.0, 0.0, 1.0, 2.0]);
/// ```
pub fn abs_vec(vector: &[f32], result: &mut [f32]) {
    assert_eq!(
        vector.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if vector.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { abs_vec_avx512(vector, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { abs_vec_avx2(vector, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { abs_vec_sse2(vector, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { abs_vec_neon(vector, result) };
            return;
        }
    }

    abs_vec_scalar(vector, result);
}

/// SIMD-optimized vector negation
///
/// Computes result[i] = -vector[i] for all elements.
///
/// # Arguments
/// * `vector` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::neg_vec;
///
/// let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
/// let mut result = vec![0.0; 5];
///
/// neg_vec(&vector, &mut result);
/// assert_eq!(result, vec![2.0, 1.0, 0.0, -1.0, -2.0]);
/// ```
pub fn neg_vec(vector: &[f32], result: &mut [f32]) {
    assert_eq!(
        vector.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if vector.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { neg_vec_avx512(vector, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { neg_vec_avx2(vector, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { neg_vec_sse2(vector, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { neg_vec_neon(vector, result) };
            return;
        }
    }

    neg_vec_scalar(vector, result);
}

/// SIMD-optimized vector reciprocal
///
/// Computes result[i] = 1.0 / vector[i] for all elements.
/// Division by zero results in infinity according to IEEE 754 standard.
///
/// # Arguments
/// * `vector` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::reciprocal_vec;
///
/// let vector = vec![1.0, 2.0, 4.0, 0.5];
/// let mut result = vec![0.0; 4];
///
/// reciprocal_vec(&vector, &mut result);
/// assert_eq!(result, vec![1.0, 0.5, 0.25, 2.0]);
/// ```
pub fn reciprocal_vec(vector: &[f32], result: &mut [f32]) {
    assert_eq!(
        vector.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if vector.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { reciprocal_vec_avx512(vector, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { reciprocal_vec_avx2(vector, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { reciprocal_vec_sse2(vector, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { reciprocal_vec_neon(vector, result) };
            return;
        }
    }

    reciprocal_vec_scalar(vector, result);
}

/// SIMD-optimized vector squaring
///
/// Computes result[i] = vector[i] * vector[i] for all elements.
///
/// # Arguments
/// * `vector` - Input vector
/// * `result` - Output vector (must have same length as input)
///
/// # Panics
/// Panics if input and output vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::arithmetic_ops::square_vec;
///
/// let vector = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
/// let mut result = vec![0.0; 7];
///
/// square_vec(&vector, &mut result);
/// assert_eq!(result, vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);
/// ```
pub fn square_vec(vector: &[f32], result: &mut [f32]) {
    assert_eq!(
        vector.len(),
        result.len(),
        "Input and output vectors must have the same length"
    );

    if vector.is_empty() {
        return;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx512f") {
            unsafe { square_vec_avx512(vector, result) };
            return;
        } else if is_x86_feature_detected!("avx2") {
            unsafe { square_vec_avx2(vector, result) };
            return;
        } else if is_x86_feature_detected!("sse2") {
            unsafe { square_vec_sse2(vector, result) };
            return;
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            unsafe { square_vec_neon(vector, result) };
            return;
        }
    }

    square_vec_scalar(vector, result);
}

// ============================================================================
// Scalar implementations (fallbacks)
// ============================================================================

fn add_vec_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
}

fn subtract_vec_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] - b[i];
    }
}

fn multiply_vec_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }
}

fn divide_vec_scalar(a: &[f32], b: &[f32], result: &mut [f32]) {
    for i in 0..a.len() {
        result[i] = a[i] / b[i];
    }
}

fn fma_scalar(a: &mut [f32], b: &[f32], c: &[f32]) {
    for i in 0..a.len() {
        a[i] = a[i] * b[i] + c[i];
    }
}

fn scale_vec_scalar(vector: &[f32], scalar: f32, result: &mut [f32]) {
    for i in 0..vector.len() {
        result[i] = vector[i] * scalar;
    }
}

fn abs_vec_scalar(vector: &[f32], result: &mut [f32]) {
    for i in 0..vector.len() {
        result[i] = vector[i].abs();
    }
}

fn neg_vec_scalar(vector: &[f32], result: &mut [f32]) {
    for i in 0..vector.len() {
        result[i] = -vector[i];
    }
}

fn reciprocal_vec_scalar(vector: &[f32], result: &mut [f32]) {
    for i in 0..vector.len() {
        result[i] = 1.0 / vector[i];
    }
}

fn square_vec_scalar(vector: &[f32], result: &mut [f32]) {
    for i in 0..vector.len() {
        result[i] = vector[i] * vector[i];
    }
}

// ============================================================================
// SSE2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn add_vec_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm_add_ps(a_vec, b_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn subtract_vec_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm_sub_ps(a_vec, b_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn multiply_vec_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm_mul_ps(a_vec, b_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn divide_vec_sse2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm_div_ps(a_vec, b_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn fma_sse2(a: &mut [f32], b: &[f32], c: &[f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let c_vec = _mm_loadu_ps(c.as_ptr().add(i));

        // Simulate FMA with separate multiply and add
        let result_vec = _mm_add_ps(_mm_mul_ps(a_vec, b_vec), c_vec);
        _mm_storeu_ps(a.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn scale_vec_sse2(vector: &[f32], scalar: f32, result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let scalar_vec = _mm_set1_ps(scalar);
    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = _mm_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm_mul_ps(vector_vec, scalar_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn abs_vec_sse2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Create mask to clear sign bit (absolute value)
    let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = _mm_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm_and_ps(vector_vec, abs_mask);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn neg_vec_sse2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Create mask to flip sign bit
    let sign_mask = _mm_set1_ps(f32::from_bits(0x80000000));
    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = _mm_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm_xor_ps(vector_vec, sign_mask);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn reciprocal_vec_sse2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = _mm_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm_div_ps(_mm_set1_ps(1.0), vector_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn square_vec_sse2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = _mm_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm_mul_ps(vector_vec, vector_vec);
        _mm_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}

// ============================================================================
// AVX2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn add_vec_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm256_add_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn subtract_vec_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm256_sub_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn multiply_vec_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn divide_vec_avx2(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm256_div_ps(a_vec, b_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn fma_avx2(a: &mut [f32], b: &[f32], c: &[f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let c_vec = _mm256_loadu_ps(c.as_ptr().add(i));

        // Simulate FMA with separate multiply and add
        let result_vec = _mm256_add_ps(_mm256_mul_ps(a_vec, b_vec), c_vec);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scale_vec_avx2(vector: &[f32], scalar: f32, result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let scalar_vec = _mm256_set1_ps(scalar);
    let mut i = 0;

    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(vector_vec, scalar_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn abs_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut i = 0;

    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_and_ps(vector_vec, abs_mask);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn neg_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let sign_mask = _mm256_set1_ps(f32::from_bits(0x80000000));
    let mut i = 0;

    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_xor_ps(vector_vec, sign_mask);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn reciprocal_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let one_vec = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_div_ps(one_vec, vector_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn square_vec_avx2(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 8 <= vector.len() {
        let vector_vec = _mm256_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm256_mul_ps(vector_vec, vector_vec);
        _mm256_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}

// ============================================================================
// AVX512 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn add_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Process 16 elements at a time
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_add_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn subtract_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_sub_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn multiply_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn divide_vec_avx512(a: &[f32], b: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        let result_vec = _mm512_div_ps(a_vec, b_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn scale_vec_avx512(vector: &[f32], scalar: f32, result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let scalar_vec = _mm512_set1_ps(scalar);
    let mut i = 0;

    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(vector_vec, scalar_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn abs_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_abs_ps(vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn neg_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let zero = _mm512_setzero_ps();
    let mut i = 0;

    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_sub_ps(zero, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn reciprocal_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let one_vec = _mm512_set1_ps(1.0);
    let mut i = 0;

    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_div_ps(one_vec, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn square_vec_avx512(vector: &[f32], result: &mut [f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    while i + 16 <= vector.len() {
        let vector_vec = _mm512_loadu_ps(vector.as_ptr().add(i));
        let result_vec = _mm512_mul_ps(vector_vec, vector_vec);
        _mm512_storeu_ps(result.as_mut_ptr().add(i), result_vec);
        i += 16;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}

// ============================================================================
// FMA intrinsic implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "fma")]
unsafe fn fma_fma_intrinsic(a: &mut [f32], b: &[f32], c: &[f32]) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut i = 0;

    // Use AVX2 with FMA instructions
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let c_vec = _mm256_loadu_ps(c.as_ptr().add(i));

        // True FMA: a * b + c
        let result_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
        _mm256_storeu_ps(a.as_mut_ptr().add(i), result_vec);
        i += 8;
    }

    // Handle remaining elements
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

// ============================================================================
// NEON implementations (ARM AArch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn add_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vaddq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] + b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn subtract_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vsubq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] - b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn multiply_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vmulq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] * b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn divide_vec_neon(a: &[f32], b: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let result_vec = vdivq_f32(a_vec, b_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        result[i] = a[i] / b[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn fma_neon(a: &mut [f32], b: &[f32], c: &[f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        let c_vec = vld1q_f32(c.as_ptr().add(i));

        // True FMA: a * b + c
        let result_vec = vfmaq_f32(c_vec, a_vec, b_vec);
        vst1q_f32(a.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < a.len() {
        a[i] = a[i] * b[i] + c[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn scale_vec_neon(vector: &[f32], scalar: f32, result: &mut [f32]) {
    use core::arch::aarch64::*;

    let scalar_vec = vdupq_n_f32(scalar);
    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vmulq_f32(vector_vec, scalar_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * scalar;
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn abs_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vabsq_f32(vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i].abs();
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn neg_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vnegq_f32(vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = -vector[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn reciprocal_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let one_vec = vdupq_n_f32(1.0);
        let result_vec = vdivq_f32(one_vec, vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = 1.0 / vector[i];
        i += 1;
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn square_vec_neon(vector: &[f32], result: &mut [f32]) {
    use core::arch::aarch64::*;

    let mut i = 0;

    while i + 4 <= vector.len() {
        let vector_vec = vld1q_f32(vector.as_ptr().add(i));
        let result_vec = vmulq_f32(vector_vec, vector_vec);
        vst1q_f32(result.as_mut_ptr().add(i), result_vec);
        i += 4;
    }

    // Handle remaining elements
    while i < vector.len() {
        result[i] = vector[i] * vector[i];
        i += 1;
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_vec() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result = vec![0.0; 4];

        add_vec(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 8.0, 10.0, 12.0]);

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        add_vec(&empty_a, &empty_b, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }

    #[test]
    fn test_subtract_vec() {
        let a = vec![10.0, 8.0, 6.0, 4.0];
        let b = vec![3.0, 2.0, 1.0, 1.0];
        let mut result = vec![0.0; 4];

        subtract_vec(&a, &b, &mut result);
        assert_eq!(result, vec![7.0, 6.0, 5.0, 3.0]);
    }

    #[test]
    fn test_multiply_vec() {
        let a = vec![2.0, 3.0, 4.0, 5.0];
        let b = vec![3.0, 4.0, 5.0, 6.0];
        let mut result = vec![0.0; 4];

        multiply_vec(&a, &b, &mut result);
        assert_eq!(result, vec![6.0, 12.0, 20.0, 30.0]);
    }

    #[test]
    fn test_divide_vec() {
        let a = vec![12.0, 15.0, 20.0, 25.0];
        let b = vec![3.0, 3.0, 4.0, 5.0];
        let mut result = vec![0.0; 4];

        divide_vec(&a, &b, &mut result);
        assert_eq!(result, vec![4.0, 5.0, 5.0, 5.0]);

        // Test division by zero
        let div_zero_a = vec![1.0, 2.0];
        let div_zero_b = vec![0.0, 0.0];
        let mut div_zero_result = vec![0.0; 2];

        divide_vec(&div_zero_a, &div_zero_b, &mut div_zero_result);
        assert_eq!(div_zero_result[0], f32::INFINITY);
        assert_eq!(div_zero_result[1], f32::INFINITY);
    }

    #[test]
    fn test_fma() {
        let mut a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![2.0, 3.0, 4.0, 5.0];
        let c = vec![1.0, 1.0, 1.0, 1.0];

        fma(&mut a, &b, &c);
        // a = a * b + c = [1*2+1, 2*3+1, 3*4+1, 4*5+1] = [3, 7, 13, 21]
        assert_eq!(a, vec![3.0, 7.0, 13.0, 21.0]);

        // Test with empty vectors
        let mut empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        let empty_c: Vec<f32> = vec![];
        fma(&mut empty_a, &empty_b, &empty_c);
        assert_eq!(empty_a, Vec::<f32>::new());
    }

    #[test]
    fn test_scale_vec() {
        let vector = vec![1.0, 2.0, 3.0, 4.0];
        let mut result = vec![0.0; 4];

        scale_vec(&vector, 2.5, &mut result);
        assert_eq!(result, vec![2.5, 5.0, 7.5, 10.0]);

        // Test scaling by zero
        scale_vec(&vector, 0.0, &mut result);
        assert_eq!(result, vec![0.0, 0.0, 0.0, 0.0]);

        // Test scaling by negative
        scale_vec(&vector, -2.0, &mut result);
        assert_eq!(result, vec![-2.0, -4.0, -6.0, -8.0]);
    }

    #[test]
    fn test_scale_vec_inplace() {
        let mut vector = vec![1.0, 2.0, 3.0, 4.0];
        scale_vec_inplace(&mut vector, 3.0);
        assert_eq!(vector, vec![3.0, 6.0, 9.0, 12.0]);
    }

    #[test]
    fn test_abs_vec() {
        let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 5];

        abs_vec(&vector, &mut result);
        assert_eq!(result, vec![2.0, 1.0, 0.0, 1.0, 2.0]);

        // Test with all positive
        let positive = vec![1.0, 2.0, 3.0];
        let mut positive_result = vec![0.0; 3];
        abs_vec(&positive, &mut positive_result);
        assert_eq!(positive_result, positive);
    }

    #[test]
    fn test_neg_vec() {
        let vector = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        let mut result = vec![0.0; 5];

        neg_vec(&vector, &mut result);
        assert_eq!(result, vec![2.0, 1.0, 0.0, -1.0, -2.0]);

        // Test double negation
        let temp_result = result.clone();
        neg_vec(&temp_result, &mut result);
        assert_eq!(result, vector);
    }

    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_add_vec_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let mut result = vec![0.0; 2];
        add_vec(&a, &b, &mut result);
    }

    #[test]
    #[should_panic(expected = "Output vector must have the same length as input vectors")]
    fn test_add_vec_output_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let mut result = vec![0.0; 2];
        add_vec(&a, &b, &mut result);
    }

    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_fma_dimension_mismatch() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        let c = vec![7.0, 8.0, 9.0];
        fma(&mut a, &b, &c);
    }

    #[test]
    fn test_large_vectors() {
        // Test with vectors larger than SIMD register sizes to ensure
        // remainder handling works correctly
        let size = 100;
        let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let b: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();
        let mut result = vec![0.0; size];

        add_vec(&a, &b, &mut result);

        for i in 0..size {
            assert_eq!(result[i], (2 * i + 1) as f32);
        }
    }

    #[test]
    fn test_arithmetic_properties() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let mut result1 = vec![0.0; 4];
        let mut result2 = vec![0.0; 4];

        // Test commutativity of addition: a + b = b + a
        add_vec(&a, &b, &mut result1);
        add_vec(&b, &a, &mut result2);
        assert_eq!(result1, result2);

        // Test commutativity of multiplication: a * b = b * a
        multiply_vec(&a, &b, &mut result1);
        multiply_vec(&b, &a, &mut result2);
        assert_eq!(result1, result2);

        // Test subtraction is not commutative: a - b ≠ b - a (in general)
        subtract_vec(&a, &b, &mut result1);
        subtract_vec(&b, &a, &mut result2);
        assert_ne!(result1, result2);
    }

    #[test]
    fn test_reciprocal_vec() {
        let vector = vec![1.0, 2.0, 4.0, 0.5];
        let mut result = vec![0.0; 4];

        reciprocal_vec(&vector, &mut result);
        assert_eq!(result, vec![1.0, 0.5, 0.25, 2.0]);

        // Test with empty vectors
        let empty_vector: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        reciprocal_vec(&empty_vector, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }

    #[test]
    fn test_square_vec() {
        let vector = vec![-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
        let mut result = vec![0.0; 7];

        square_vec(&vector, &mut result);
        assert_eq!(result, vec![9.0, 4.0, 1.0, 0.0, 1.0, 4.0, 9.0]);

        // Test with empty vectors
        let empty_vector: Vec<f32> = vec![];
        let mut empty_result: Vec<f32> = vec![];
        square_vec(&empty_vector, &mut empty_result);
        assert_eq!(empty_result, Vec::<f32>::new());
    }

    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length")]
    fn test_reciprocal_vec_dimension_mismatch() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        reciprocal_vec(&vector, &mut result);
    }

    #[test]
    #[should_panic(expected = "Input and output vectors must have the same length")]
    fn test_square_vec_dimension_mismatch() {
        let vector = vec![1.0, 2.0, 3.0];
        let mut result = vec![0.0; 2];
        square_vec(&vector, &mut result);
    }
}
