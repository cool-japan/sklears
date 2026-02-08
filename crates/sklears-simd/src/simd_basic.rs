//! Basic SIMD Vector Operations
//!
//! This module provides fundamental SIMD-accelerated vector operations including
//! dot products, norms, scaling, and basic arithmetic operations optimized for
//! different SIMD instruction sets.

#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

use crate::simd_types::*;
use crate::simd_utils::*;

/// SIMD-optimized dot product computation
///
/// Computes the dot product of two vectors using the best available SIMD
/// instruction set. Falls back to optimized scalar implementation if SIMD
/// is not available.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// The dot product of the two vectors
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust,ignore
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![5.0, 6.0, 7.0, 8.0];
/// let result = dot_product(&a, &b);
/// assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { dot_product_avx512(a, b) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { dot_product_avx2(a, b) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { dot_product_sse2(a, b) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { dot_product_neon(a, b) };
    }

    #[cfg(not(any(target_arch = "aarch64", target_arch = "x86", target_arch = "x86_64")))]
    {
        dot_product_scalar(a, b)
    }
}

/// SIMD-optimized L2 norm computation
///
/// Computes the L2 (Euclidean) norm of a vector: √(Σᵢ xᵢ²)
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// The L2 norm of the vector
///
/// # Examples
/// ```rust,ignore
/// let x = vec![3.0, 4.0];
/// let result = norm_l2(&x);
/// assert_eq!(result, 5.0); // √(3² + 4²) = √25 = 5
/// ```
pub fn norm_l2(x: &[f32]) -> f32 {
    dot_product(x, x).sqrt()
}

/// Alternative norm function for compatibility
pub fn norm(vector: &[f32]) -> f32 {
    norm_l2(vector)
}

/// SIMD-optimized vector scaling
///
/// Scales all elements of a vector by a scalar value in-place.
///
/// # Arguments
/// * `vector` - Vector to scale (modified in-place)
/// * `scalar` - Scaling factor
///
/// # Examples
/// ```rust,ignore
/// let mut x = vec![1.0, 2.0, 3.0, 4.0];
/// scale(&mut x, 2.0);
/// assert_eq!(x, vec![2.0, 4.0, 6.0, 8.0]);
/// ```
pub fn scale(vector: &mut [f32], scalar: f32) {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { scale_avx512(vector, scalar) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { scale_avx2(vector, scalar) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { scale_sse2(vector, scalar) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { scale_neon(vector, scalar) };
    }

    // Scalar fallback
    for element in vector.iter_mut() {
        *element *= scalar;
    }
}

/// SIMD-optimized fused multiply-add operation
///
/// Computes a\[i\] = a\[i\] * b\[i\] + c\[i\] for all elements.
/// This is equivalent to a\[i\] * b\[i\] + c\[i\] but may be more efficient
/// on platforms with dedicated FMA instructions.
///
/// # Arguments
/// * `a` - First input vector (modified in-place)
/// * `b` - Second input vector
/// * `c` - Third input vector
///
/// # Panics
/// Panics if vectors have different lengths
///
/// # Examples
/// ```rust,ignore
/// let mut a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let c = vec![7.0, 8.0, 9.0];
/// fma(&mut a, &b, &c);
/// // a[0] = 1.0 * 4.0 + 7.0 = 11.0
/// // a[1] = 2.0 * 5.0 + 8.0 = 18.0
/// // a[2] = 3.0 * 6.0 + 9.0 = 27.0
/// assert_eq!(a, vec![11.0, 18.0, 27.0]);
/// ```
pub fn fma(a: &mut [f32], b: &[f32], c: &[f32]) {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    assert_eq!(a.len(), c.len(), "Vectors must have the same length");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("fma") {
            return unsafe { fma_avx2_fma(a, b, c) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { fma_avx2(a, b, c) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { fma_sse2(a, b, c) };
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        return unsafe { fma_neon(a, b, c) };
    }

    // Scalar fallback
    for i in 0..a.len() {
        a[i] = a[i] * b[i] + c[i];
    }
}

/// Vector addition returning a new vector
///
/// Computes element-wise addition of two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// New vector containing the element-wise sum
///
/// # Panics
/// Panics if vectors have different lengths
pub fn add_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let mut result = vec![0.0; a.len()];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            unsafe { add_avx2(a, b, &mut result) };
            return result;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { add_sse2(a, b, &mut result) };
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { add_neon(a, b, &mut result) };
        return result;
    }

    // Scalar fallback
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }

    result
}

/// Fused multiply-add returning a new vector
///
/// Computes a\[i\] * b\[i\] + c\[i\] for all elements, returning a new vector.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
/// * `c` - Third input vector
///
/// # Returns
/// New vector containing the FMA results
pub fn fused_multiply_add(a: &[f32], b: &[f32], c: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");
    assert_eq!(a.len(), c.len(), "Vectors must have the same length");

    let mut result = a.to_vec();
    fma(&mut result, b, c);
    result
}

/// Vector subtraction returning a new vector
///
/// Computes element-wise subtraction of two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// New vector containing a\[i\] - b\[i\]
pub fn subtract_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let mut result = vec![0.0; a.len()];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            unsafe { subtract_avx2(a, b, &mut result) };
            return result;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { subtract_sse2(a, b, &mut result) };
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { subtract_neon(a, b, &mut result) };
        return result;
    }

    // Scalar fallback
    for i in 0..a.len() {
        result[i] = a[i] - b[i];
    }

    result
}

/// Element-wise vector multiplication
///
/// Computes element-wise multiplication of two vectors.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector
///
/// # Returns
/// New vector containing a\[i\] * b\[i\]
pub fn multiply_vectors(a: &[f32], b: &[f32]) -> Vec<f32> {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    let mut result = vec![0.0; a.len()];

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            unsafe { multiply_avx2(a, b, &mut result) };
            return result;
        } else if crate::simd_feature_detected!("sse2") {
            unsafe { multiply_sse2(a, b, &mut result) };
            return result;
        }
    }

    #[cfg(target_arch = "aarch64")]
    {
        unsafe { multiply_neon(a, b, &mut result) };
        return result;
    }

    // Scalar fallback
    for i in 0..a.len() {
        result[i] = a[i] * b[i];
    }

    result
}

/// Vector scaling returning a new vector
///
/// Scales all elements by a scalar value, returning a new vector.
///
/// # Arguments
/// * `vector` - Input vector
/// * `scalar` - Scaling factor
///
/// # Returns
/// New vector with scaled values
pub fn scale_vector(vector: &[f32], scalar: f32) -> Vec<f32> {
    let mut result = vector.to_vec();
    scale(&mut result, scalar);
    result
}

// Scalar implementations
fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];
        let result = dot_product(&a, &b);
        assert!((result - 70.0).abs() < 1e-6);
    }

    #[test]
    fn test_norm_l2() {
        let x = vec![3.0, 4.0];
        let result = norm_l2(&x);
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        scale(&mut x, 2.0);
        assert_eq!(x, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn test_fma() {
        let mut a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0, 9.0];
        fma(&mut a, &b, &c);
        assert_eq!(a, vec![11.0, 18.0, 27.0]);
    }

    #[test]
    fn test_add_vectors() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = add_vectors(&a, &b);
        assert_eq!(result, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_subtract_vectors() {
        let a = vec![5.0, 7.0, 9.0];
        let b = vec![1.0, 2.0, 3.0];
        let result = subtract_vectors(&a, &b);
        assert_eq!(result, vec![4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_multiply_vectors() {
        let a = vec![2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0];
        let result = multiply_vectors(&a, &b);
        assert_eq!(result, vec![10.0, 18.0, 28.0]);
    }

    #[test]
    fn test_fused_multiply_add() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let c = vec![7.0, 8.0, 9.0];
        let result = fused_multiply_add(&a, &b, &c);
        assert_eq!(result, vec![11.0, 18.0, 27.0]);
    }

    #[test]
    fn test_scale_vector() {
        let x = vec![1.0, 2.0, 3.0];
        let result = scale_vector(&x, 3.0);
        assert_eq!(result, vec![3.0, 6.0, 9.0]);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_dot_product_mismatched_lengths() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0, 2.0, 3.0];
        dot_product(&a, &b);
    }
}