//! SIMD optimizations for ensemble operations
//!
//! This module provides SIMD-accelerated implementations of common ensemble operations
//! such as array addition, scalar multiplication, and weighted averaging.

use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

#[cfg(target_arch = "aarch64")]
use std::arch::aarch64::*;

/// SIMD-optimized array operations for ensemble methods
pub struct SimdOps;

impl SimdOps {
    /// Add two arrays with SIMD acceleration when available
    pub fn add_arrays(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Self::add_arrays_avx2(a, b)
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            not(target_feature = "avx2")
        ))]
        {
            Self::add_arrays_avx(a, b)
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            not(target_feature = "avx")
        ))]
        {
            Self::add_arrays_sse2(a, b)
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::add_arrays_neon(a, b)
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            target_arch = "aarch64"
        )))]
        {
            Self::add_arrays_scalar(a, b)
        }
    }

    /// Multiply array by scalar with SIMD acceleration when available
    pub fn scalar_multiply(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
        {
            Self::scalar_multiply_avx2(array, scalar)
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "avx",
            not(target_feature = "avx2")
        ))]
        {
            Self::scalar_multiply_avx(array, scalar)
        }

        #[cfg(all(
            target_arch = "x86_64",
            target_feature = "sse2",
            not(target_feature = "avx")
        ))]
        {
            Self::scalar_multiply_sse2(array, scalar)
        }

        #[cfg(target_arch = "aarch64")]
        {
            Self::scalar_multiply_neon(array, scalar)
        }

        #[cfg(not(any(
            all(target_arch = "x86_64", target_feature = "sse2"),
            target_arch = "aarch64"
        )))]
        {
            Self::scalar_multiply_scalar(array, scalar)
        }
    }

    /// Weighted sum of multiple arrays with SIMD acceleration
    pub fn weighted_sum(arrays: &[&Array1<Float>], weights: &[Float]) -> Array1<Float> {
        debug_assert_eq!(
            arrays.len(),
            weights.len(),
            "Arrays and weights must have same length"
        );
        debug_assert!(!arrays.is_empty(), "Must have at least one array");

        let len = arrays[0].len();
        debug_assert!(
            arrays.iter().all(|a| a.len() == len),
            "All arrays must have same length"
        );

        let mut result = Array1::zeros(len);

        for (array, &weight) in arrays.iter().zip(weights.iter()) {
            let weighted_array = Self::scalar_multiply(array, weight);
            result = Self::add_arrays(&result, &weighted_array);
        }

        result
    }

    /// Scalar implementation for fallback
    fn add_arrays_scalar(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        a + b
    }

    /// Scalar implementation for fallback
    fn scalar_multiply_scalar(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        array * scalar
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn add_arrays_avx2(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        unsafe {
            let len = a.len();
            let mut result = Array1::zeros(len);
            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let simd_len = len & !7; // Process 8 elements at a time

            for i in (0..simd_len).step_by(8) {
                let a_vec = _mm256_loadu_pd(&a_slice[i] as *const f64);
                let b_vec = _mm256_loadu_pd(&b_slice[i] as *const f64);
                let sum = _mm256_add_pd(a_vec, b_vec);
                _mm256_storeu_pd(&mut result_slice[i] as *mut f64, sum);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = a_slice[i] + b_slice[i];
            }

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    fn scalar_multiply_avx2(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        unsafe {
            let len = array.len();
            let mut result = Array1::zeros(len);
            let array_slice = array.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let scalar_vec = _mm256_set1_pd(scalar);
            let simd_len = len & !7; // Process 8 elements at a time

            for i in (0..simd_len).step_by(8) {
                let array_vec = _mm256_loadu_pd(&array_slice[i] as *const f64);
                let product = _mm256_mul_pd(array_vec, scalar_vec);
                _mm256_storeu_pd(&mut result_slice[i] as *mut f64, product);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = array_slice[i] * scalar;
            }

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn add_arrays_avx(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        unsafe {
            let len = a.len();
            let mut result = Array1::zeros(len);
            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let simd_len = len & !3; // Process 4 elements at a time

            for i in (0..simd_len).step_by(4) {
                let a_vec = _mm256_loadu_pd(&a_slice[i] as *const f64);
                let b_vec = _mm256_loadu_pd(&b_slice[i] as *const f64);
                let sum = _mm256_add_pd(a_vec, b_vec);
                _mm256_storeu_pd(&mut result_slice[i] as *mut f64, sum);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = a_slice[i] + b_slice[i];
            }

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "avx"))]
    fn scalar_multiply_avx(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        unsafe {
            let len = array.len();
            let mut result = Array1::zeros(len);
            let array_slice = array.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let scalar_vec = _mm256_set1_pd(scalar);
            let simd_len = len & !3; // Process 4 elements at a time

            for i in (0..simd_len).step_by(4) {
                let array_vec = _mm256_loadu_pd(&array_slice[i] as *const f64);
                let product = _mm256_mul_pd(array_vec, scalar_vec);
                _mm256_storeu_pd(&mut result_slice[i] as *mut f64, product);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = array_slice[i] * scalar;
            }

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn add_arrays_sse2(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        unsafe {
            let len = a.len();
            let mut result = Array1::zeros(len);
            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let simd_len = len & !1; // Process 2 elements at a time

            for i in (0..simd_len).step_by(2) {
                let a_vec = _mm_loadu_pd(&a_slice[i] as *const f64);
                let b_vec = _mm_loadu_pd(&b_slice[i] as *const f64);
                let sum = _mm_add_pd(a_vec, b_vec);
                _mm_storeu_pd(&mut result_slice[i] as *mut f64, sum);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = a_slice[i] + b_slice[i];
            }

            result
        }
    }

    #[cfg(all(target_arch = "x86_64", target_feature = "sse2"))]
    fn scalar_multiply_sse2(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        unsafe {
            let len = array.len();
            let mut result = Array1::zeros(len);
            let array_slice = array.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let scalar_vec = _mm_set1_pd(scalar);
            let simd_len = len & !1; // Process 2 elements at a time

            for i in (0..simd_len).step_by(2) {
                let array_vec = _mm_loadu_pd(&array_slice[i] as *const f64);
                let product = _mm_mul_pd(array_vec, scalar_vec);
                _mm_storeu_pd(&mut result_slice[i] as *mut f64, product);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = array_slice[i] * scalar;
            }

            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn add_arrays_neon(a: &Array1<Float>, b: &Array1<Float>) -> Array1<Float> {
        unsafe {
            let len = a.len();
            let mut result = Array1::zeros(len);
            let a_slice = a.as_slice().unwrap();
            let b_slice = b.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let simd_len = len & !1; // Process 2 elements at a time

            for i in (0..simd_len).step_by(2) {
                let a_vec = vld1q_f64(&a_slice[i] as *const f64);
                let b_vec = vld1q_f64(&b_slice[i] as *const f64);
                let sum = vaddq_f64(a_vec, b_vec);
                vst1q_f64(&mut result_slice[i] as *mut f64, sum);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = a_slice[i] + b_slice[i];
            }

            result
        }
    }

    #[cfg(target_arch = "aarch64")]
    fn scalar_multiply_neon(array: &Array1<Float>, scalar: Float) -> Array1<Float> {
        unsafe {
            let len = array.len();
            let mut result = Array1::zeros(len);
            let array_slice = array.as_slice().unwrap();
            let result_slice = result.as_slice_mut().unwrap();

            let scalar_vec = vdupq_n_f64(scalar);
            let simd_len = len & !1; // Process 2 elements at a time

            for i in (0..simd_len).step_by(2) {
                let array_vec = vld1q_f64(&array_slice[i] as *const f64);
                let product = vmulq_f64(array_vec, scalar_vec);
                vst1q_f64(&mut result_slice[i] as *mut f64, product);
            }

            // Handle remaining elements
            for i in simd_len..len {
                result_slice[i] = array_slice[i] * scalar;
            }

            result
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_add_arrays() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];
        let result = SimdOps::add_arrays(&a, &b);
        let expected = array![3.0, 5.0, 7.0, 9.0, 11.0];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_scalar_multiply() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = SimdOps::scalar_multiply(&a, 2.0);
        let expected = array![2.0, 4.0, 6.0, 8.0, 10.0];

        assert_eq!(result, expected);
    }

    #[test]
    fn test_weighted_sum() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];
        let arrays = vec![&a, &b];
        let weights = vec![0.5, 0.5];

        let result = SimdOps::weighted_sum(&arrays, &weights);
        let expected = array![2.5, 3.5, 4.5];

        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert!((actual - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_large_array_operations() {
        let size = 1000;
        let a = Array1::from_elem(size, 1.0);
        let b = Array1::from_elem(size, 2.0);

        let result = SimdOps::add_arrays(&a, &b);
        assert_eq!(result.len(), size);
        assert!(result.iter().all(|&x| (x - 3.0).abs() < 1e-10));

        let scaled = SimdOps::scalar_multiply(&a, 5.0);
        assert!(scaled.iter().all(|&x| (x - 5.0).abs() < 1e-10));
    }
}
