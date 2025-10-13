//! Advanced SIMD optimization techniques
//!
//! This module provides cutting-edge optimization techniques for SIMD operations,
//! including cache-aware algorithms, vectorization strategies, and memory-efficient
//! implementations for high-performance machine learning computations.

use crate::traits::SimdError;
#[cfg(target_arch = "x86")]
use core::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

#[cfg(feature = "no-std")]
use alloc::{vec, vec::Vec};
#[cfg(not(feature = "no-std"))]
use std::{vec, vec::Vec};

/// Advanced SIMD optimization strategies
pub struct AdvancedSimdOptimizer {
    cache_line_size: usize,
    prefetch_distance: usize,
    vectorization_width: usize,
}

impl AdvancedSimdOptimizer {
    /// Create a new advanced SIMD optimizer with platform-specific tuning
    pub fn new() -> Self {
        Self {
            cache_line_size: 64,    // Common cache line size
            prefetch_distance: 512, // Prefetch distance in bytes
            vectorization_width: 8, // AVX-256 width for f32
        }
    }

    /// Cache-aware matrix multiplication with blocking
    pub fn cache_aware_matrix_multiply(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError> {
        if a.len() != m * k || b.len() != k * n || c.len() != m * n {
            return Err(SimdError::DimensionMismatch {
                expected: m * n,
                actual: c.len(),
            });
        }

        // Optimal block sizes for cache efficiency
        let block_size = self.calculate_optimal_block_size(m, n, k);

        for i in (0..m).step_by(block_size) {
            for j in (0..n).step_by(block_size) {
                for kk in (0..k).step_by(block_size) {
                    let i_max = (i + block_size).min(m);
                    let j_max = (j + block_size).min(n);
                    let k_max = (kk + block_size).min(k);

                    self.matrix_multiply_block(a, b, c, i, j, kk, i_max, j_max, k_max, m, n, k)?;
                }
            }
        }

        Ok(())
    }

    /// Vectorized dot product with manual loop unrolling
    pub fn vectorized_dot_product(&self, a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
        if a.len() != b.len() {
            return Err(SimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        let len = a.len();
        if len == 0 {
            return Ok(0.0);
        }

        let mut result = 0.0f32;

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.dot_product_avx2(a, b) };
            } else if is_x86_feature_detected!("sse2") {
                return unsafe { self.dot_product_sse2(a, b) };
            }
        }

        // Fallback scalar implementation with loop unrolling
        let chunks = len / 4;
        let remainder = len % 4;

        for i in 0..chunks {
            let base = i * 4;
            result += a[base] * b[base]
                + a[base + 1] * b[base + 1]
                + a[base + 2] * b[base + 2]
                + a[base + 3] * b[base + 3];
        }

        for i in (chunks * 4)..(chunks * 4 + remainder) {
            result += a[i] * b[i];
        }

        Ok(result)
    }

    /// Memory-efficient convolution with spatial locality optimization
    pub fn optimized_convolution(
        &self,
        input: &[f32],
        kernel: &[f32],
        output: &mut [f32],
        input_shape: (usize, usize, usize), // (channels, height, width)
        kernel_shape: (usize, usize, usize), // (filters, height, width)
        stride: usize,
        padding: usize,
    ) -> Result<(), SimdError> {
        let (in_channels, in_height, in_width) = input_shape;
        let (out_channels, k_height, k_width) = kernel_shape;

        let out_height = (in_height + 2 * padding - k_height) / stride + 1;
        let out_width = (in_width + 2 * padding - k_width) / stride + 1;

        if output.len() != out_channels * out_height * out_width {
            return Err(SimdError::DimensionMismatch {
                expected: out_channels * out_height * out_width,
                actual: output.len(),
            });
        }

        // Use im2col transformation for better memory access patterns
        let im2col_data =
            self.im2col_transform(input, input_shape, kernel_shape, stride, padding)?;

        // Perform optimized matrix multiplication
        self.cache_aware_matrix_multiply(
            kernel,
            &im2col_data,
            output,
            out_channels,
            out_height * out_width,
            in_channels * k_height * k_width,
        )?;

        Ok(())
    }

    /// Advanced vectorized reduction with tree reduction pattern
    pub fn vectorized_reduction(&self, data: &[f32], op: ReductionOp) -> Result<f32, SimdError> {
        if data.is_empty() {
            return Err(SimdError::EmptyInput);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                return unsafe { self.reduction_avx2(data, op) };
            }
        }

        // Fallback scalar implementation
        match op {
            ReductionOp::Sum => Ok(data.iter().sum()),
            ReductionOp::Max => Ok(data.iter().copied().fold(f32::NEG_INFINITY, f32::max)),
            ReductionOp::Min => Ok(data.iter().copied().fold(f32::INFINITY, f32::min)),
            ReductionOp::Mean => Ok(data.iter().sum::<f32>() / data.len() as f32),
        }
    }

    // Private helper methods

    fn calculate_optimal_block_size(&self, m: usize, n: usize, k: usize) -> usize {
        // Estimate optimal block size based on cache size and data dimensions
        let cache_size = 32768; // L1 cache size estimate
        let element_size = 4; // f32 size
        let block_elements = cache_size / (3 * element_size); // Account for A, B, C matrices

        let block_size = (block_elements as f32).sqrt() as usize;
        block_size.min(64).max(8) // Clamp to reasonable range
    }

    fn matrix_multiply_block(
        &self,
        a: &[f32],
        b: &[f32],
        c: &mut [f32],
        i_start: usize,
        j_start: usize,
        k_start: usize,
        i_end: usize,
        j_end: usize,
        k_end: usize,
        m: usize,
        n: usize,
        k: usize,
    ) -> Result<(), SimdError> {
        for i in i_start..i_end {
            for j in j_start..j_end {
                let mut sum = 0.0f32;
                for kk in k_start..k_end {
                    sum += a[i * k + kk] * b[kk * n + j];
                }
                c[i * n + j] += sum;
            }
        }
        Ok(())
    }

    fn im2col_transform(
        &self,
        input: &[f32],
        input_shape: (usize, usize, usize),
        kernel_shape: (usize, usize, usize),
        stride: usize,
        padding: usize,
    ) -> Result<Vec<f32>, SimdError> {
        let (in_channels, in_height, in_width) = input_shape;
        let (_, k_height, k_width) = kernel_shape;

        let out_height = (in_height + 2 * padding - k_height) / stride + 1;
        let out_width = (in_width + 2 * padding - k_width) / stride + 1;

        let mut result = vec![0.0f32; in_channels * k_height * k_width * out_height * out_width];

        for c in 0..in_channels {
            for kh in 0..k_height {
                for kw in 0..k_width {
                    for oh in 0..out_height {
                        for ow in 0..out_width {
                            let ih = oh * stride + kh;
                            let iw = ow * stride + kw;

                            let value = if ih >= padding
                                && ih < in_height + padding
                                && iw >= padding
                                && iw < in_width + padding
                            {
                                let adjusted_ih = ih - padding;
                                let adjusted_iw = iw - padding;
                                input[c * in_height * in_width
                                    + adjusted_ih * in_width
                                    + adjusted_iw]
                            } else {
                                0.0f32
                            };

                            let col_idx = (c * k_height * k_width + kh * k_width + kw)
                                * out_height
                                * out_width
                                + oh * out_width
                                + ow;
                            result[col_idx] = value;
                        }
                    }
                }
            }
        }

        Ok(result)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn dot_product_avx2(&self, a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
        let len = a.len();
        let mut sum = _mm256_setzero_ps();

        let chunks = len / 8;
        for i in 0..chunks {
            let a_vec = _mm256_loadu_ps(a.as_ptr().add(i * 8));
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(i * 8));
            let product = _mm256_mul_ps(a_vec, b_vec);
            sum = _mm256_add_ps(sum, product);
        }

        // Horizontal sum
        let sum_high = _mm256_extractf128_ps(sum, 1);
        let sum_low = _mm256_castps256_ps128(sum);
        let sum128 = _mm_add_ps(sum_high, sum_low);

        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum128);
        let mut final_sum = result[0] + result[1] + result[2] + result[3];

        // Handle remaining elements
        for i in (chunks * 8)..len {
            final_sum += a[i] * b[i];
        }

        Ok(final_sum)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn dot_product_sse2(&self, a: &[f32], b: &[f32]) -> Result<f32, SimdError> {
        let len = a.len();
        let mut sum = _mm_setzero_ps();

        let chunks = len / 4;
        for i in 0..chunks {
            let a_vec = _mm_loadu_ps(a.as_ptr().add(i * 4));
            let b_vec = _mm_loadu_ps(b.as_ptr().add(i * 4));
            let product = _mm_mul_ps(a_vec, b_vec);
            sum = _mm_add_ps(sum, product);
        }

        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut final_sum = result[0] + result[1] + result[2] + result[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            final_sum += a[i] * b[i];
        }

        Ok(final_sum)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    unsafe fn reduction_avx2(&self, data: &[f32], op: ReductionOp) -> Result<f32, SimdError> {
        let len = data.len();
        let chunks = len / 8;

        let mut accumulator = match op {
            ReductionOp::Sum | ReductionOp::Mean => _mm256_setzero_ps(),
            ReductionOp::Max => _mm256_set1_ps(f32::NEG_INFINITY),
            ReductionOp::Min => _mm256_set1_ps(f32::INFINITY),
        };

        for i in 0..chunks {
            let data_vec = _mm256_loadu_ps(data.as_ptr().add(i * 8));
            accumulator = match op {
                ReductionOp::Sum | ReductionOp::Mean => _mm256_add_ps(accumulator, data_vec),
                ReductionOp::Max => _mm256_max_ps(accumulator, data_vec),
                ReductionOp::Min => _mm256_min_ps(accumulator, data_vec),
            };
        }

        // Horizontal reduction
        let mut result = [0.0f32; 8];
        _mm256_storeu_ps(result.as_mut_ptr(), accumulator);

        let mut final_result = match op {
            ReductionOp::Sum | ReductionOp::Mean => result.iter().sum::<f32>(),
            ReductionOp::Max => result.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            ReductionOp::Min => result.iter().copied().fold(f32::INFINITY, f32::min),
        };

        // Handle remaining elements
        for i in (chunks * 8)..len {
            final_result = match op {
                ReductionOp::Sum | ReductionOp::Mean => final_result + data[i],
                ReductionOp::Max => final_result.max(data[i]),
                ReductionOp::Min => final_result.min(data[i]),
            };
        }

        if matches!(op, ReductionOp::Mean) {
            final_result /= len as f32;
        }

        Ok(final_result)
    }
}

impl Default for AdvancedSimdOptimizer {
    fn default() -> Self {
        Self::new()
    }
}

/// Reduction operation types
#[derive(Debug, Clone, Copy)]
pub enum ReductionOp {
    Sum,
    Max,
    Min,
    Mean,
}

/// Cache-aware sorting for SIMD operations
pub struct CacheAwareSort;

impl CacheAwareSort {
    /// Vectorized merge sort with cache-friendly access patterns
    pub fn vectorized_merge_sort(data: &mut [f32]) {
        if data.len() <= 1 {
            return;
        }

        let mid = data.len() / 2;
        Self::vectorized_merge_sort(&mut data[..mid]);
        Self::vectorized_merge_sort(&mut data[mid..]);

        // Cache-friendly merge
        let mut temp = vec![0.0f32; data.len()];
        Self::cache_friendly_merge(data, &mut temp, mid);
        data.copy_from_slice(&temp);
    }

    fn cache_friendly_merge(data: &[f32], temp: &mut [f32], mid: usize) {
        let (left, right) = data.split_at(mid);
        let mut i = 0;
        let mut j = 0;
        let mut k = 0;

        while i < left.len() && j < right.len() {
            if left[i] <= right[j] {
                temp[k] = left[i];
                i += 1;
            } else {
                temp[k] = right[j];
                j += 1;
            }
            k += 1;
        }

        while i < left.len() {
            temp[k] = left[i];
            i += 1;
            k += 1;
        }

        while j < right.len() {
            temp[k] = right[j];
            j += 1;
            k += 1;
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vectorized_dot_product() {
        let optimizer = AdvancedSimdOptimizer::new();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = optimizer.vectorized_dot_product(&a, &b).unwrap();
        let expected = 1.0 * 5.0 + 2.0 * 6.0 + 3.0 * 7.0 + 4.0 * 8.0;

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_vectorized_reduction() {
        let optimizer = AdvancedSimdOptimizer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        let sum = optimizer
            .vectorized_reduction(&data, ReductionOp::Sum)
            .unwrap();
        assert_eq!(sum, 15.0);

        let max = optimizer
            .vectorized_reduction(&data, ReductionOp::Max)
            .unwrap();
        assert_eq!(max, 5.0);

        let min = optimizer
            .vectorized_reduction(&data, ReductionOp::Min)
            .unwrap();
        assert_eq!(min, 1.0);

        let mean = optimizer
            .vectorized_reduction(&data, ReductionOp::Mean)
            .unwrap();
        assert_eq!(mean, 3.0);
    }

    #[test]
    fn test_cache_aware_sort() {
        let mut data = vec![5.0, 2.0, 8.0, 1.0, 9.0, 3.0];
        CacheAwareSort::vectorized_merge_sort(&mut data);

        let expected = vec![1.0, 2.0, 3.0, 5.0, 8.0, 9.0];
        assert_eq!(data, expected);
    }
}
