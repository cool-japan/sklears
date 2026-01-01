/// SIMD-optimized operations for core machine learning computations
///
/// This module provides SIMD-accelerated implementations of fundamental operations
/// to achieve maximum performance on modern hardware.
use crate::error::Result;
use crate::types::{Array1, Array2, FloatBounds};
// SciRS2 Policy: Using scirs2_core::ndarray (COMPLIANT)
use rayon::prelude::*;
use scirs2_core::ndarray::Axis;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized operations for machine learning
pub struct SimdOps;

impl SimdOps {
    /// Vectorized dot product with SIMD optimization
    #[cfg(feature = "simd")]
    pub fn dot_product_simd_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::dot_product_avx2_f32(a, b) };
            } else if is_x86_feature_detected!("sse4.1") {
                return unsafe { Self::dot_product_sse_f32(a, b) };
            }
        }

        // Fallback to regular implementation
        Self::dot_product_fallback(a, b)
    }

    /// Vectorized dot product with SIMD optimization for f64
    #[cfg(feature = "simd")]
    pub fn dot_product_simd_f64(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::dot_product_avx2_f64(a, b) };
            } else if is_x86_feature_detected!("sse4.1") {
                return unsafe { Self::dot_product_sse_f64(a, b) };
            }
        }

        // Fallback to regular implementation
        Self::dot_product_fallback(a, b)
    }

    /// AVX2-optimized dot product for f32
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let len = a.len();
        let simd_len = len & !7; // Round down to nearest multiple of 8

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let vmul = _mm256_mul_ps(va, vb);
            sum = _mm256_add_ps(sum, vmul);
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0f32; 8];
        _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f32>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// AVX2-optimized dot product for f64
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2_f64(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = _mm256_setzero_pd();
        let len = a.len();
        let simd_len = len & !3; // Round down to nearest multiple of 4

        // Process 4 elements at a time
        for i in (0..simd_len).step_by(4) {
            let va = _mm256_loadu_pd(a.as_ptr().add(i));
            let vb = _mm256_loadu_pd(b.as_ptr().add(i));
            let vmul = _mm256_mul_pd(va, vb);
            sum = _mm256_add_pd(sum, vmul);
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0f64; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// SSE-optimized dot product for f32
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn dot_product_sse_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm_setzero_ps();
        let len = a.len();
        let simd_len = len & !3; // Round down to nearest multiple of 4

        // Process 4 elements at a time
        for i in (0..simd_len).step_by(4) {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let vmul = _mm_mul_ps(va, vb);
            sum = _mm_add_ps(sum, vmul);
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0f32; 4];
        _mm_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f32>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// SSE-optimized dot product for f64
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn dot_product_sse_f64(a: &[f64], b: &[f64]) -> f64 {
        let mut sum = _mm_setzero_pd();
        let len = a.len();
        let simd_len = len & !1; // Round down to nearest multiple of 2

        // Process 2 elements at a time
        for i in (0..simd_len).step_by(2) {
            let va = _mm_loadu_pd(a.as_ptr().add(i));
            let vb = _mm_loadu_pd(b.as_ptr().add(i));
            let vmul = _mm_mul_pd(va, vb);
            sum = _mm_add_pd(sum, vmul);
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0f64; 2];
        _mm_storeu_pd(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f64>();

        // Handle remaining elements
        for i in simd_len..len {
            total += a[i] * b[i];
        }

        total
    }

    /// Fallback dot product implementation
    fn dot_product_fallback<T: FloatBounds + std::iter::Sum>(a: &[T], b: &[T]) -> T {
        a.iter().zip(b.iter()).map(|(&ai, &bi)| ai * bi).sum()
    }

    /// Vectorized addition with SIMD
    #[cfg(feature = "simd")]
    pub fn add_arrays_simd_f32(a: &mut [f32], b: &[f32]) {
        assert_eq!(a.len(), b.len());

        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                unsafe { Self::add_arrays_avx2_f32(a, b) };
                return;
            } else if is_x86_feature_detected!("sse4.1") {
                unsafe { Self::add_arrays_sse_f32(a, b) };
                return;
            }
        }

        // Fallback
        for (ai, &bi) in a.iter_mut().zip(b.iter()) {
            *ai += bi;
        }
    }

    /// AVX2-optimized array addition for f32
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn add_arrays_avx2_f32(a: &mut [f32], b: &[f32]) {
        let len = a.len();
        let simd_len = len & !7; // Round down to nearest multiple of 8

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let result = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(a.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            a[i] += b[i];
        }
    }

    /// SSE-optimized array addition for f32
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn add_arrays_sse_f32(a: &mut [f32], b: &[f32]) {
        let len = a.len();
        let simd_len = len & !3; // Round down to nearest multiple of 4

        // Process 4 elements at a time
        for i in (0..simd_len).step_by(4) {
            let va = _mm_loadu_ps(a.as_ptr().add(i));
            let vb = _mm_loadu_ps(b.as_ptr().add(i));
            let result = _mm_add_ps(va, vb);
            _mm_storeu_ps(a.as_mut_ptr().add(i), result);
        }

        // Handle remaining elements
        for i in simd_len..len {
            a[i] += b[i];
        }
    }

    /// SIMD-optimized element-wise operations on matrices
    #[cfg(feature = "simd")]
    pub fn elementwise_op_simd<F>(a: &Array2<f32>, b: &Array2<f32>, result: &mut Array2<f32>, op: F)
    where
        F: Fn(f32, f32) -> f32 + Send + Sync,
    {
        assert_eq!(a.shape(), b.shape());
        assert_eq!(a.shape(), result.shape());

        // Try to get contiguous slices for SIMD operations
        if let (Some(a_slice), Some(b_slice), Some(result_slice)) =
            (a.as_slice(), b.as_slice(), result.as_slice_mut())
        {
            // Use parallel SIMD processing for large arrays
            if a_slice.len() > 1000 {
                result_slice
                    .par_chunks_mut(8)
                    .zip(a_slice.par_chunks(8))
                    .zip(b_slice.par_chunks(8))
                    .for_each(|((result_chunk, a_chunk), b_chunk)| {
                        for ((r, &ai), &bi) in result_chunk
                            .iter_mut()
                            .zip(a_chunk.iter())
                            .zip(b_chunk.iter())
                        {
                            *r = op(ai, bi);
                        }
                    });
            } else {
                // Sequential processing for smaller arrays
                for ((r, &ai), &bi) in result_slice
                    .iter_mut()
                    .zip(a_slice.iter())
                    .zip(b_slice.iter())
                {
                    *r = op(ai, bi);
                }
            }
        } else {
            // Fallback to ndarray iteration if not contiguous
            result
                .iter_mut()
                .zip(a.iter())
                .zip(b.iter())
                .for_each(|((r, &ai), &bi)| *r = op(ai, bi));
        }
    }

    /// SIMD-optimized matrix multiplication using cache-friendly blocking
    #[cfg(feature = "simd")]
    pub fn matrix_multiply_simd(a: &Array2<f32>, b: &Array2<f32>) -> Result<Array2<f32>> {
        let (m, k) = a.dim();
        let (k2, n) = b.dim();

        if k != k2 {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: format!("({m}, {k}) × ({k}, {n})"),
                actual: format!("({m}, {k}) × ({k2}, {n})"),
            });
        }

        let mut result = Array2::<f32>::zeros((m, n));

        const BLOCK_SIZE: usize = 64; // Cache-friendly block size

        // Blocked matrix multiplication with SIMD
        for i_block in (0..m).step_by(BLOCK_SIZE) {
            for j_block in (0..n).step_by(BLOCK_SIZE) {
                for k_block in (0..k).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(m);
                    let j_end = (j_block + BLOCK_SIZE).min(n);
                    let k_end = (k_block + BLOCK_SIZE).min(k);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0f32;

                            // Get row and column slices
                            let a_row = a.row(i);
                            let b_col = b.column(j);

                            // Use SIMD dot product if slices are contiguous
                            if let (Some(a_slice), Some(b_slice)) =
                                (a_row.as_slice(), b_col.as_slice())
                            {
                                let k_slice = &a_slice[k_block..k_end];
                                let b_k_slice = &b_slice[k_block..k_end];
                                sum += Self::dot_product_simd_f32(k_slice, b_k_slice);
                            } else {
                                // Fallback to manual computation
                                for ki in k_block..k_end {
                                    sum += a[[i, ki]] * b[[ki, j]];
                                }
                            }

                            result[[i, j]] += sum;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    /// SIMD-optimized distance computations
    #[cfg(feature = "simd")]
    pub fn euclidean_distances_simd(x: &Array2<f32>, y: &Array2<f32>) -> Result<Array2<f32>> {
        let (n_x, d_x) = x.dim();
        let (n_y, d_y) = y.dim();

        if d_x != d_y {
            return Err(crate::error::SklearsError::ShapeMismatch {
                expected: "same number of features".to_string(),
                actual: format!("{d_x} vs {d_y}"),
            });
        }

        let mut distances = Array2::<f32>::zeros((n_x, n_y));

        // Parallel computation of pairwise distances
        distances
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .enumerate()
            .for_each(|(i, mut row)| {
                let x_i = x.row(i);

                for (j, dist) in row.iter_mut().enumerate() {
                    let y_j = y.row(j);

                    // Compute squared euclidean distance using SIMD
                    if let (Some(x_slice), Some(y_slice)) = (x_i.as_slice(), y_j.as_slice()) {
                        let sum_sq;

                        #[cfg(target_arch = "x86_64")]
                        {
                            if is_x86_feature_detected!("avx2") {
                                sum_sq =
                                    unsafe { Self::squared_distance_avx2_f32(x_slice, y_slice) };
                            } else {
                                sum_sq = Self::squared_distance_fallback(x_slice, y_slice);
                            }
                        }

                        #[cfg(not(target_arch = "x86_64"))]
                        {
                            sum_sq = Self::squared_distance_fallback(x_slice, y_slice);
                        }

                        *dist = sum_sq.sqrt();
                    } else {
                        // Fallback for non-contiguous arrays
                        let sum_sq: f32 = x_i
                            .iter()
                            .zip(y_j.iter())
                            .map(|(&xi, &yi)| (xi - yi).powi(2))
                            .sum();
                        *dist = sum_sq.sqrt();
                    }
                }
            });

        Ok(distances)
    }

    /// AVX2-optimized squared distance computation
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn squared_distance_avx2_f32(a: &[f32], b: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let len = a.len();
        let simd_len = len & !7; // Round down to nearest multiple of 8

        // Process 8 elements at a time
        for i in (0..simd_len).step_by(8) {
            let va = _mm256_loadu_ps(a.as_ptr().add(i));
            let vb = _mm256_loadu_ps(b.as_ptr().add(i));
            let diff = _mm256_sub_ps(va, vb);
            let sq_diff = _mm256_mul_ps(diff, diff);
            sum = _mm256_add_ps(sum, sq_diff);
        }

        // Extract sum from SIMD register
        let mut result_array = [0.0f32; 8];
        _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f32>();

        // Handle remaining elements
        for i in simd_len..len {
            let diff = a[i] - b[i];
            total += diff * diff;
        }

        total
    }

    /// Fallback squared distance computation
    fn squared_distance_fallback(a: &[f32], b: &[f32]) -> f32 {
        a.iter()
            .zip(b.iter())
            .map(|(&ai, &bi)| (ai - bi).powi(2))
            .sum()
    }

    /// SIMD-optimized reduction operations
    #[cfg(feature = "simd")]
    pub fn sum_simd_f32(array: &[f32]) -> f32 {
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { Self::sum_avx2_f32(array) };
            } else if is_x86_feature_detected!("sse4.1") {
                return unsafe { Self::sum_sse_f32(array) };
            }
        }

        array.iter().sum()
    }

    /// AVX2-optimized sum
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "avx2")]
    unsafe fn sum_avx2_f32(array: &[f32]) -> f32 {
        let mut sum = _mm256_setzero_ps();
        let len = array.len();
        let simd_len = len & !7;

        for i in (0..simd_len).step_by(8) {
            let v = _mm256_loadu_ps(array.as_ptr().add(i));
            sum = _mm256_add_ps(sum, v);
        }

        let mut result_array = [0.0f32; 8];
        _mm256_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f32>();

        total += array
            .iter()
            .skip(simd_len)
            .take(len - simd_len)
            .sum::<f32>();

        total
    }

    /// SSE-optimized sum
    #[cfg(all(target_arch = "x86_64", feature = "simd"))]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sum_sse_f32(array: &[f32]) -> f32 {
        let mut sum = _mm_setzero_ps();
        let len = array.len();
        let simd_len = len & !3;

        for i in (0..simd_len).step_by(4) {
            let v = _mm_loadu_ps(array.as_ptr().add(i));
            sum = _mm_add_ps(sum, v);
        }

        let mut result_array = [0.0f32; 4];
        _mm_storeu_ps(result_array.as_mut_ptr(), sum);
        let mut total = result_array.iter().sum::<f32>();

        total += array
            .iter()
            .skip(simd_len)
            .take(len - simd_len)
            .sum::<f32>();

        total
    }
}

/// High-level SIMD-accelerated array operations
pub trait SimdArrayOps<T> {
    /// Compute dot product using SIMD if available
    fn dot_simd(&self, other: &Self) -> T;

    /// Add arrays in-place using SIMD if available
    fn add_assign_simd(&mut self, other: &Self);

    /// Sum all elements using SIMD if available
    fn sum_simd(&self) -> T;
}

impl SimdArrayOps<f32> for Array1<f32> {
    fn dot_simd(&self, other: &Self) -> f32 {
        #[cfg(feature = "simd")]
        {
            if let (Some(self_slice), Some(other_slice)) = (self.as_slice(), other.as_slice()) {
                return SimdOps::dot_product_simd_f32(self_slice, other_slice);
            }
        }

        // Fallback
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn add_assign_simd(&mut self, other: &Self) {
        #[cfg(feature = "simd")]
        {
            if let (Some(self_slice), Some(other_slice)) = (self.as_slice_mut(), other.as_slice()) {
                SimdOps::add_arrays_simd_f32(self_slice, other_slice);
                return;
            }
        }

        // Fallback
        *self += other;
    }

    fn sum_simd(&self) -> f32 {
        #[cfg(feature = "simd")]
        {
            if let Some(slice) = self.as_slice() {
                return SimdOps::sum_simd_f32(slice);
            }
        }

        // Fallback
        self.sum()
    }
}

impl SimdArrayOps<f64> for Array1<f64> {
    fn dot_simd(&self, other: &Self) -> f64 {
        #[cfg(feature = "simd")]
        {
            if let (Some(self_slice), Some(other_slice)) = (self.as_slice(), other.as_slice()) {
                return SimdOps::dot_product_simd_f64(self_slice, other_slice);
            }
        }

        // Fallback
        self.iter().zip(other.iter()).map(|(&a, &b)| a * b).sum()
    }

    fn add_assign_simd(&mut self, other: &Self) {
        // f64 SIMD operations would be implemented similarly
        *self += other;
    }

    fn sum_simd(&self) -> f64 {
        // f64 SIMD sum would be implemented similarly
        self.sum()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_dot_product_f32() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let b = vec![8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let expected = SimdOps::dot_product_fallback(&a, &b);

        #[cfg(feature = "simd")]
        {
            let simd_result = SimdOps::dot_product_simd_f32(&a, &b);
            assert_relative_eq!(simd_result, expected, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_array_operations() {
        let mut a = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let b = Array1::from_vec(vec![4.0, 3.0, 2.0, 1.0]);

        let dot_result = a.dot_simd(&b);
        assert_relative_eq!(dot_result, 20.0, epsilon = 1e-6);

        let sum_result = a.sum_simd();
        assert_relative_eq!(sum_result, 10.0, epsilon = 1e-6);

        let original_a = a.clone();
        a.add_assign_simd(&b);

        for (i, &val) in a.iter().enumerate() {
            assert_relative_eq!(val, original_a[i] + b[i], epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_matrix_multiply() {
        let a = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let b = Array2::from_shape_vec((3, 2), vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]).unwrap();

        let result = SimdOps::matrix_multiply_simd(&a, &b).unwrap();
        let expected = a.dot(&b);

        assert_eq!(result.shape(), expected.shape());
        for (actual, expected) in result.iter().zip(expected.iter()) {
            assert_relative_eq!(*actual, *expected, epsilon = 1e-6);
        }
    }

    #[test]
    #[cfg(feature = "simd")]
    fn test_simd_euclidean_distances() {
        let x =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])
                .unwrap();
        let y =
            Array2::from_shape_vec((2, 3), vec![1.0f32, 2.0f32, 3.0f32, 4.0f32, 5.0f32, 6.0f32])
                .unwrap();

        let distances = SimdOps::euclidean_distances_simd(&x, &y).unwrap();

        assert_eq!(distances.shape(), &[2, 2]);

        // Distance from point to itself should be 0
        assert_relative_eq!(distances[[0, 0]], 0.0f32, epsilon = 1e-6);
        assert_relative_eq!(distances[[1, 1]], 0.0f32, epsilon = 1e-6);

        // Distance should be symmetric
        assert_relative_eq!(distances[[0, 1]], distances[[1, 0]], epsilon = 1e-6);

        // Cross distances should be non-zero
        assert!(distances[[0, 1]] > 0.0);
    }
}
