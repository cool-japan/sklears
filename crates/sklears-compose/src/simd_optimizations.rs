//! SIMD optimizations for parallel pipeline execution
//!
//! This module provides SIMD-optimized operations for data processing
//! in machine learning pipelines, focusing on vectorized operations
//! that can significantly improve performance.

use num_cpus;
use scirs2_core::ndarray::{
    s, Array1, Array2, ArrayView1, ArrayView2, ArrayViewMut1, ArrayViewMut2, Zip,
};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    traits::Estimator,
    types::{Float, FloatBounds},
};
#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
use std::arch::x86_64::*;

/// SIMD configuration for optimized operations
#[derive(Debug, Clone)]
pub struct SimdConfig {
    /// Enable AVX2 instructions
    pub use_avx2: bool,
    /// Enable AVX512 instructions
    pub use_avx512: bool,
    /// Enable FMA instructions
    pub use_fma: bool,
    /// Vectorization width (elements per vector)
    pub vector_width: usize,
    /// Memory alignment requirements
    pub alignment: usize,
    /// Minimum array size for SIMD operations
    pub simd_threshold: usize,
}

impl Default for SimdConfig {
    fn default() -> Self {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            Self {
                use_avx2: is_x86_feature_detected!("avx2"),
                use_avx512: is_x86_feature_detected!("avx512f"),
                use_fma: is_x86_feature_detected!("fma"),
                vector_width: if is_x86_feature_detected!("avx512f") {
                    16
                } else if is_x86_feature_detected!("avx2") {
                    8
                } else {
                    4
                },
                alignment: 64,
                simd_threshold: 64,
            }
        }
        #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
        {
            Self {
                use_avx2: false,
                use_avx512: false,
                use_fma: false,
                vector_width: 4,
                alignment: 64,
                simd_threshold: 64,
            }
        }
    }
}

/// SIMD-optimized operations for pipeline processing
pub struct SimdOps {
    config: SimdConfig,
}

impl SimdOps {
    /// Create new SIMD operations instance
    #[must_use]
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Create with default configuration
    #[must_use]
    pub fn default() -> Self {
        Self::new(SimdConfig::default())
    }

    /// Vectorized addition of two arrays
    pub fn add_arrays(
        &self,
        a: &ArrayView1<Float>,
        b: &ArrayView1<Float>,
    ) -> SklResult<Array1<Float>> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let len = a.len();
        let mut result = Array1::zeros(len);

        // Use standard addition (SIMD optimizations would be implemented here)
        Zip::from(&mut result)
            .and(a)
            .and(b)
            .for_each(|r, &a_val, &b_val| *r = a_val + b_val);

        Ok(result)
    }

    /// AVX2-optimized array addition
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn add_arrays_avx2(
        &self,
        a: &ArrayView1<Float>,
        b: &ArrayView1<Float>,
        result: &mut ArrayViewMut1<Float>,
    ) -> SklResult<()> {
        let len = a.len();
        let vector_len = 4; // AVX2 processes 4 f64 values at once
        let chunks = len / vector_len;
        let remainder = len % vector_len;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();
        let result_ptr = result.as_mut_ptr();

        // Process chunks of 4 elements
        for i in 0..chunks {
            let offset = i * vector_len;

            let a_vec = _mm256_loadu_pd(a_ptr.add(offset));
            let b_vec = _mm256_loadu_pd(b_ptr.add(offset));
            let sum_vec = _mm256_add_pd(a_vec, b_vec);

            _mm256_storeu_pd(result_ptr.add(offset), sum_vec);
        }

        // Handle remaining elements
        for i in (chunks * vector_len)..len {
            *result_ptr.add(i) = *a_ptr.add(i) + *b_ptr.add(i);
        }

        Ok(())
    }

    /// Vectorized matrix multiplication
    pub fn matrix_multiply(
        &self,
        a: &ArrayView2<Float>,
        b: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        if a.ncols() != b.nrows() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions incompatible for multiplication".to_string(),
            ));
        }

        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        let mut result = Array2::zeros((m, n));

        // Use standard matrix multiplication (SIMD optimizations would be implemented here)
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0;
                for idx in 0..k {
                    sum += a[[i, idx]] * b[[idx, j]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// AVX2-optimized matrix multiplication
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn matrix_multiply_avx2(
        &self,
        a: &ArrayView2<Float>,
        b: &ArrayView2<Float>,
        result: &mut ArrayViewMut2<Float>,
    ) -> SklResult<()> {
        let (m, k) = (a.nrows(), a.ncols());
        let n = b.ncols();
        let vector_len = 4; // AVX2 processes 4 f64 values at once

        for i in 0..m {
            for j in (0..n).step_by(vector_len) {
                let end_j = std::cmp::min(j + vector_len, n);
                let mut sum_vec = _mm256_setzero_pd();

                for idx in 0..k {
                    let a_val = _mm256_broadcast_sd(&a[[i, idx]]);

                    if end_j - j == vector_len {
                        let b_vec = _mm256_loadu_pd(b.as_ptr().add(idx * n + j));
                        sum_vec = _mm256_fmadd_pd(a_val, b_vec, sum_vec);
                    } else {
                        // Handle remainder
                        for col in j..end_j {
                            let prev_sum = result[[i, col]];
                            result[[i, col]] = prev_sum + a[[i, idx]] * b[[idx, col]];
                        }
                        continue;
                    }
                }

                if end_j - j == vector_len {
                    _mm256_storeu_pd(result.as_mut_ptr().add(i * n + j), sum_vec);
                }
            }
        }

        Ok(())
    }

    /// Vectorized dot product
    pub fn dot_product(&self, a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> SklResult<Float> {
        if a.len() != b.len() {
            return Err(SklearsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        let len = a.len();

        // Use standard dot product (SIMD optimizations would be implemented here)
        Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).sum())
    }

    /// AVX2-optimized dot product
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    unsafe fn dot_product_avx2(
        &self,
        a: &ArrayView1<Float>,
        b: &ArrayView1<Float>,
    ) -> SklResult<Float> {
        let len = a.len();
        let vector_len = 4; // AVX2 processes 4 f64 values at once
        let chunks = len / vector_len;
        let remainder = len % vector_len;

        let a_ptr = a.as_ptr();
        let b_ptr = b.as_ptr();

        let mut sum_vec = _mm256_setzero_pd();

        // Process chunks
        for i in 0..chunks {
            let offset = i * vector_len;
            let a_vec = _mm256_loadu_pd(a_ptr.add(offset));
            let b_vec = _mm256_loadu_pd(b_ptr.add(offset));
            sum_vec = _mm256_fmadd_pd(a_vec, b_vec, sum_vec);
        }

        // Horizontal sum of vector elements
        let mut result: [f64; 4] = [0.0; 4];
        _mm256_storeu_pd(result.as_mut_ptr(), sum_vec);
        let mut total_sum = result.iter().sum::<f64>();

        // Handle remainder
        for i in (chunks * vector_len)..len {
            total_sum += *a_ptr.add(i) * *b_ptr.add(i);
        }

        Ok(total_sum)
    }

    /// Vectorized element-wise operations
    pub fn elementwise_op<F>(&self, a: &ArrayView1<Float>, op: F) -> SklResult<Array1<Float>>
    where
        F: Fn(Float) -> Float + Copy,
    {
        let len = a.len();
        let mut result = Array1::zeros(len);

        if len >= self.config.simd_threshold && self.can_vectorize_op() {
            // For operations that can be vectorized, use SIMD
            self.elementwise_op_simd(a, op, &mut result.view_mut())?;
        } else {
            // Fallback to scalar operations
            Zip::from(&mut result)
                .and(a)
                .for_each(|r, &val| *r = op(val));
        }

        Ok(result)
    }

    /// Check if operation can be vectorized efficiently
    fn can_vectorize_op(&self) -> bool {
        // This is a simplified check - in practice, you'd analyze the operation
        self.config.use_avx2
    }

    /// SIMD element-wise operations (simplified implementation)
    fn elementwise_op_simd<F>(
        &self,
        a: &ArrayView1<Float>,
        op: F,
        result: &mut ArrayViewMut1<Float>,
    ) -> SklResult<()>
    where
        F: Fn(Float) -> Float + Copy,
    {
        // For complex operations, fallback to scalar for now
        // Real implementations would use lookup tables or polynomial approximations
        Zip::from(result).and(a).for_each(|r, &val| *r = op(val));
        Ok(())
    }

    /// Vectorized normalization (L2 norm)
    pub fn normalize_l2(&self, a: &ArrayView1<Float>) -> SklResult<Array1<Float>> {
        let norm = self.dot_product(a, a)?.sqrt();
        if norm == 0.0 {
            return Ok(a.to_owned());
        }

        let inv_norm = 1.0 / norm;
        self.elementwise_op(a, |x| x * inv_norm)
    }

    /// Vectorized scaling
    pub fn scale(&self, a: &ArrayView1<Float>, scale_factor: Float) -> SklResult<Array1<Float>> {
        // Use standard scaling (SIMD optimizations would be implemented here)
        Ok(a.mapv(|x| x * scale_factor))
    }

    /// AVX2-optimized scaling
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn scale_avx2(
        &self,
        a: &ArrayView1<Float>,
        scale_factor: Float,
    ) -> SklResult<Array1<Float>> {
        let len = a.len();
        let vector_len = 4; // AVX2 processes 4 f64 values at once
        let chunks = len / vector_len;
        let remainder = len % vector_len;

        let mut result = Array1::zeros(len);
        let a_ptr = a.as_ptr();
        let result_ptr: *mut Float = result.as_mut_ptr();
        let scale_vec = _mm256_broadcast_sd(&scale_factor);

        // Process chunks
        for i in 0..chunks {
            let offset = i * vector_len;
            let a_vec = _mm256_loadu_pd(a_ptr.add(offset));
            let scaled_vec = _mm256_mul_pd(a_vec, scale_vec);
            _mm256_storeu_pd(result_ptr.add(offset), scaled_vec);
        }

        // Handle remainder
        for i in (chunks * vector_len)..len {
            *result_ptr.add(i) = *a_ptr.add(i) * scale_factor;
        }

        Ok(result)
    }

    /// Memory-aligned array allocation for optimal SIMD performance
    #[must_use]
    pub fn aligned_array_1d(&self, size: usize) -> Array1<Float> {
        // Use ndarray's default allocation which should be reasonably aligned
        // In production, you'd want to use aligned allocation
        Array1::zeros(size)
    }

    /// Memory-aligned matrix allocation
    #[must_use]
    pub fn aligned_array_2d(&self, rows: usize, cols: usize) -> Array2<Float> {
        Array2::zeros((rows, cols))
    }

    /// Check if arrays are properly aligned for SIMD operations
    #[must_use]
    pub fn is_aligned(&self, ptr: *const Float) -> bool {
        (ptr as usize) % self.config.alignment == 0
    }

    /// Get optimal chunk size for parallel SIMD operations
    #[must_use]
    pub fn optimal_chunk_size(&self, total_size: usize) -> usize {
        let base_chunk = std::cmp::max(self.config.simd_threshold, total_size / num_cpus::get());

        // Round to vector boundary
        (base_chunk / self.config.vector_width) * self.config.vector_width
    }

    /// Vectorized feature standardization (z-score normalization)
    pub fn standardize_features(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut standardized = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let column = data.column(j);
            let mean = column.sum() / (n_samples as Float);
            let variance = column.iter().map(|&x| (x - mean).powi(2)).sum::<Float>()
                / ((n_samples - 1) as Float);
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                for i in 0..n_samples {
                    standardized[[i, j]] = (data[[i, j]] - mean) / std_dev;
                }
            } else {
                for i in 0..n_samples {
                    standardized[[i, j]] = 0.0;
                }
            }
        }

        Ok(standardized)
    }

    /// Vectorized min-max scaling
    pub fn min_max_scale(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        let mut scaled = Array2::zeros((n_samples, n_features));

        for j in 0..n_features {
            let column = data.column(j);
            let min_val = column.fold(Float::INFINITY, |acc, &x| acc.min(x));
            let max_val = column.fold(Float::NEG_INFINITY, |acc, &x| acc.max(x));
            let range = max_val - min_val;

            if range > 0.0 {
                for i in 0..n_samples {
                    scaled[[i, j]] = (data[[i, j]] - min_val) / range;
                }
            } else {
                for i in 0..n_samples {
                    scaled[[i, j]] = 0.0;
                }
            }
        }

        Ok(scaled)
    }

    /// Generate polynomial features up to specified degree
    pub fn polynomial_features(
        &self,
        data: &ArrayView2<Float>,
        degree: usize,
    ) -> SklResult<Array2<Float>> {
        if degree == 0 {
            return Err(SklearsError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        let (n_samples, n_features) = data.dim();

        // Calculate number of output features for polynomial expansion
        let mut n_output_features = 0;
        for d in 1..=degree {
            n_output_features += Self::binomial_coefficient(n_features + d - 1, d);
        }

        let mut result = Array2::zeros((n_samples, n_output_features));

        for i in 0..n_samples {
            let mut col_idx = 0;

            // Generate polynomial terms
            for d in 1..=degree {
                Self::generate_polynomial_terms_recursive(
                    data.row(i).as_slice().unwrap(),
                    d,
                    0,
                    1.0,
                    result.row_mut(i).as_slice_mut().unwrap(),
                    &mut col_idx,
                );
            }
        }

        Ok(result)
    }

    /// Generate polynomial terms recursively
    fn generate_polynomial_terms_recursive(
        features: &[Float],
        remaining_degree: usize,
        start_idx: usize,
        current_term: Float,
        output: &mut [Float],
        col_idx: &mut usize,
    ) {
        if remaining_degree == 0 {
            output[*col_idx] = current_term;
            *col_idx += 1;
            return;
        }

        for i in start_idx..features.len() {
            Self::generate_polynomial_terms_recursive(
                features,
                remaining_degree - 1,
                i,
                current_term * features[i],
                output,
                col_idx,
            );
        }
    }

    /// Calculate binomial coefficient C(n, k)
    fn binomial_coefficient(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let k = k.min(n - k); // Take advantage of symmetry
        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }

    /// Vectorized sum with SIMD optimization
    pub fn vectorized_sum(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        if data.len() >= self.config.simd_threshold && self.config.use_avx2 {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                return self.vectorized_sum_avx2(data);
            }
        }

        // Fallback to standard sum
        Ok(data.sum())
    }

    /// AVX2-optimized vectorized sum
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn vectorized_sum_avx2(&self, data: &ArrayView1<Float>) -> SklResult<Float> {
        let len = data.len();
        let vector_len = 4; // AVX2 processes 4 f64 values at once
        let chunks = len / vector_len;
        let remainder = len % vector_len;

        let data_ptr = data.as_ptr();
        let mut sum_vec = _mm256_setzero_pd();

        // Process chunks of 4 elements
        for i in 0..chunks {
            let offset = i * vector_len;
            let data_vec = _mm256_loadu_pd(data_ptr.add(offset));
            sum_vec = _mm256_add_pd(sum_vec, data_vec);
        }

        // Horizontal sum of vector elements
        let mut result_array: [f64; 4] = [0.0; 4];
        _mm256_storeu_pd(result_array.as_mut_ptr(), sum_vec);
        let mut total_sum = result_array.iter().sum::<f64>();

        // Handle remaining elements
        for i in (chunks * vector_len)..len {
            total_sum += *data_ptr.add(i);
        }

        Ok(total_sum)
    }

    /// Ultra-fast unsafe memory copy optimized for aligned data
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It assumes src and dst pointers are valid for `count` elements
    /// - It assumes proper memory alignment for SIMD operations
    /// - It performs unchecked pointer arithmetic
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    pub unsafe fn fast_aligned_copy(
        &self,
        src: *const Float,
        dst: *mut Float,
        count: usize,
    ) -> SklResult<()> {
        // Validate alignment assumptions
        debug_assert!(src as usize % self.config.alignment == 0);
        debug_assert!(dst as usize % self.config.alignment == 0);
        debug_assert!(count > 0);

        let vector_len = 4; // AVX2 handles 4 f64 at once
        let simd_count = count / vector_len;
        let remainder = count % vector_len;

        // SIMD copy for bulk data
        for i in 0..simd_count {
            let offset = i * vector_len;
            let data_vec = _mm256_load_pd(src.add(offset));
            _mm256_store_pd(dst.add(offset), data_vec);
        }

        // Handle remainder with scalar copy
        let remainder_start = simd_count * vector_len;
        for i in 0..remainder {
            *dst.add(remainder_start + i) = *src.add(remainder_start + i);
        }

        Ok(())
    }

    /// Unsafe unrolled matrix multiplication for small fixed-size matrices
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It assumes matrix pointers are valid and properly aligned
    /// - It performs manual loop unrolling with unchecked indexing
    /// - It assumes matrices are in row-major order
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn fast_small_matrix_mul(
        &self,
        a: *const Float,
        b: *const Float,
        c: *mut Float,
        m: usize,
        n: usize,
        k: usize,
    ) -> SklResult<()> {
        debug_assert!(m > 0 && n > 0 && k > 0);
        debug_assert!(m <= 16 && n <= 16 && k <= 16); // Optimized for small matrices

        // Manual loop unrolling for small matrices (4x4 case optimized)
        if m == 4 && n == 4 && k == 4 {
            // Highly optimized 4x4 x 4x4 multiplication
            let a00 = *a.add(0);
            let a01 = *a.add(1);
            let a02 = *a.add(2);
            let a03 = *a.add(3);
            let a10 = *a.add(4);
            let a11 = *a.add(5);
            let a12 = *a.add(6);
            let a13 = *a.add(7);
            let a20 = *a.add(8);
            let a21 = *a.add(9);
            let a22 = *a.add(10);
            let a23 = *a.add(11);
            let a30 = *a.add(12);
            let a31 = *a.add(13);
            let a32 = *a.add(14);
            let a33 = *a.add(15);

            let b00 = *b.add(0);
            let b01 = *b.add(1);
            let b02 = *b.add(2);
            let b03 = *b.add(3);
            let b10 = *b.add(4);
            let b11 = *b.add(5);
            let b12 = *b.add(6);
            let b13 = *b.add(7);
            let b20 = *b.add(8);
            let b21 = *b.add(9);
            let b22 = *b.add(10);
            let b23 = *b.add(11);
            let b30 = *b.add(12);
            let b31 = *b.add(13);
            let b32 = *b.add(14);
            let b33 = *b.add(15);

            // Unrolled computation
            *c.add(0) = a00 * b00 + a01 * b10 + a02 * b20 + a03 * b30;
            *c.add(1) = a00 * b01 + a01 * b11 + a02 * b21 + a03 * b31;
            *c.add(2) = a00 * b02 + a01 * b12 + a02 * b22 + a03 * b32;
            *c.add(3) = a00 * b03 + a01 * b13 + a02 * b23 + a03 * b33;

            *c.add(4) = a10 * b00 + a11 * b10 + a12 * b20 + a13 * b30;
            *c.add(5) = a10 * b01 + a11 * b11 + a12 * b21 + a13 * b31;
            *c.add(6) = a10 * b02 + a11 * b12 + a12 * b22 + a13 * b32;
            *c.add(7) = a10 * b03 + a11 * b13 + a12 * b23 + a13 * b33;

            *c.add(8) = a20 * b00 + a21 * b10 + a22 * b20 + a23 * b30;
            *c.add(9) = a20 * b01 + a21 * b11 + a22 * b21 + a23 * b31;
            *c.add(10) = a20 * b02 + a21 * b12 + a22 * b22 + a23 * b32;
            *c.add(11) = a20 * b03 + a21 * b13 + a22 * b23 + a23 * b33;

            *c.add(12) = a30 * b00 + a31 * b10 + a32 * b20 + a33 * b30;
            *c.add(13) = a30 * b01 + a31 * b11 + a32 * b21 + a33 * b31;
            *c.add(14) = a30 * b02 + a31 * b12 + a32 * b22 + a33 * b32;
            *c.add(15) = a30 * b03 + a31 * b13 + a32 * b23 + a33 * b33;
        } else {
            // Fallback for other small sizes with manual unrolling
            for i in 0..m {
                for j in 0..n {
                    let mut sum = 0.0;
                    for idx in 0..k {
                        sum += *a.add(i * k + idx) * *b.add(idx * n + j);
                    }
                    *c.add(i * n + j) = sum;
                }
            }
        }

        Ok(())
    }

    /// Unsafe cache-oblivious matrix transpose for better cache performance
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It performs unchecked pointer arithmetic
    /// - It assumes valid memory layouts for source and destination
    /// - It uses recursive divide-and-conquer with raw pointers
    pub unsafe fn cache_oblivious_transpose(
        &self,
        src: *const Float,
        dst: *mut Float,
        src_rows: usize,
        src_cols: usize,
        dst_cols: usize,
        row_start: usize,
        col_start: usize,
        block_rows: usize,
        block_cols: usize,
    ) -> SklResult<()> {
        const CACHE_BLOCK_SIZE: usize = 64; // Cache line optimized

        if block_rows <= CACHE_BLOCK_SIZE && block_cols <= CACHE_BLOCK_SIZE {
            // Base case: small block, do simple transpose
            for i in 0..block_rows {
                for j in 0..block_cols {
                    let src_idx = (row_start + i) * src_cols + (col_start + j);
                    let dst_idx = (col_start + j) * dst_cols + (row_start + i);
                    *dst.add(dst_idx) = *src.add(src_idx);
                }
            }
        } else if block_rows >= block_cols {
            // Split rows
            let mid_rows = block_rows / 2;
            self.cache_oblivious_transpose(
                src, dst, src_rows, src_cols, dst_cols, row_start, col_start, mid_rows, block_cols,
            )?;
            self.cache_oblivious_transpose(
                src,
                dst,
                src_rows,
                src_cols,
                dst_cols,
                row_start + mid_rows,
                col_start,
                block_rows - mid_rows,
                block_cols,
            )?;
        } else {
            // Split columns
            let mid_cols = block_cols / 2;
            self.cache_oblivious_transpose(
                src, dst, src_rows, src_cols, dst_cols, row_start, col_start, block_rows, mid_cols,
            )?;
            self.cache_oblivious_transpose(
                src,
                dst,
                src_rows,
                src_cols,
                dst_cols,
                row_start,
                col_start + mid_cols,
                block_rows,
                block_cols - mid_cols,
            )?;
        }

        Ok(())
    }

    /// Unsafe vectorized sum with manual unrolling and FMA
    ///
    /// # Safety
    ///
    /// This function is unsafe because:
    /// - It assumes proper memory alignment and validity
    /// - It performs unchecked SIMD operations
    /// - It uses manual loop unrolling for maximum performance
    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2", enable = "fma")]
    pub unsafe fn fast_vectorized_sum(&self, data: *const Float, len: usize) -> SklResult<Float> {
        debug_assert!(len > 0);
        debug_assert!(data as usize % self.config.alignment == 0);

        let vector_len = 4; // AVX2 processes 4 f64 values at once
        let unroll_factor = 4; // Process 4 vectors per iteration
        let unrolled_len = vector_len * unroll_factor;
        let unrolled_count = len / unrolled_len;
        let remainder = len % unrolled_len;

        // Initialize accumulators
        let mut sum1 = _mm256_setzero_pd();
        let mut sum2 = _mm256_setzero_pd();
        let mut sum3 = _mm256_setzero_pd();
        let mut sum4 = _mm256_setzero_pd();

        // Unrolled vectorized loop
        for i in 0..unrolled_count {
            let offset = i * unrolled_len;

            let vec1 = _mm256_load_pd(data.add(offset));
            let vec2 = _mm256_load_pd(data.add(offset + vector_len));
            let vec3 = _mm256_load_pd(data.add(offset + vector_len * 2));
            let vec4 = _mm256_load_pd(data.add(offset + vector_len * 3));

            sum1 = _mm256_add_pd(sum1, vec1);
            sum2 = _mm256_add_pd(sum2, vec2);
            sum3 = _mm256_add_pd(sum3, vec3);
            sum4 = _mm256_add_pd(sum4, vec4);
        }

        // Combine partial sums
        let combined = _mm256_add_pd(_mm256_add_pd(sum1, sum2), _mm256_add_pd(sum3, sum4));

        // Horizontal sum
        let mut result: [f64; 4] = [0.0; 4];
        _mm256_store_pd(result.as_mut_ptr(), combined);
        let mut total: Float = result.iter().sum();

        // Handle remainder
        let remainder_start = unrolled_count * unrolled_len;
        for i in 0..remainder {
            total += *data.add(remainder_start + i);
        }

        Ok(total)
    }
}

/// SIMD-optimized feature transformations
pub struct SimdFeatureOps {
    simd_ops: SimdOps,
}

impl SimdFeatureOps {
    /// Create new SIMD feature operations
    #[must_use]
    pub fn new(config: SimdConfig) -> Self {
        Self {
            simd_ops: SimdOps::new(config),
        }
    }

    /// Vectorized standardization (z-score normalization)
    pub fn standardize(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_rows, n_cols) = data.dim();
        let mut result = Array2::zeros((n_rows, n_cols));

        // Compute mean and std for each feature
        for col in 0..n_cols {
            let column = data.column(col);
            let mean = column.sum() / n_rows as Float;

            // Compute variance
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / n_rows as Float;
            let std_dev = variance.sqrt();

            if std_dev > 0.0 {
                let inv_std = 1.0 / std_dev;
                let standardized = self
                    .simd_ops
                    .elementwise_op(&column, |x| (x - mean) * inv_std)?;
                result.column_mut(col).assign(&standardized);
            } else {
                result.column_mut(col).fill(0.0);
            }
        }

        Ok(result)
    }

    /// Vectorized min-max scaling
    pub fn min_max_scale(
        &self,
        data: &ArrayView2<Float>,
        feature_range: (Float, Float),
    ) -> SklResult<Array2<Float>> {
        let (min_val, max_val) = feature_range;
        let scale = max_val - min_val;

        let (n_rows, n_cols) = data.dim();
        let mut result = Array2::zeros((n_rows, n_cols));

        for col in 0..n_cols {
            let column = data.column(col);
            let col_min = column.iter().copied().fold(Float::INFINITY, Float::min);
            let col_max = column.iter().copied().fold(Float::NEG_INFINITY, Float::max);

            if col_max > col_min {
                let col_range = col_max - col_min;
                let scaled = self
                    .simd_ops
                    .elementwise_op(&column, |x| min_val + scale * (x - col_min) / col_range)?;
                result.column_mut(col).assign(&scaled);
            } else {
                result.column_mut(col).fill(min_val);
            }
        }

        Ok(result)
    }

    /// Vectorized polynomial features
    pub fn polynomial_features(
        &self,
        data: &ArrayView2<Float>,
        degree: usize,
    ) -> SklResult<Array2<Float>> {
        if degree < 1 {
            return Err(SklearsError::InvalidInput(
                "Degree must be at least 1".to_string(),
            ));
        }

        let (n_rows, n_cols) = data.dim();

        // Calculate number of polynomial features
        let n_output_features = self.calculate_poly_features(n_cols, degree);
        let mut result = Array2::zeros((n_rows, n_output_features));

        // Copy original features
        result.slice_mut(s![.., ..n_cols]).assign(data);
        let mut feature_idx = n_cols;

        // Generate polynomial combinations
        for deg in 2..=degree {
            feature_idx +=
                self.generate_combinations(data, &mut result.view_mut(), deg, feature_idx)?;
        }

        Ok(result)
    }

    /// Calculate number of polynomial features
    fn calculate_poly_features(&self, n_features: usize, degree: usize) -> usize {
        // This is a simplified calculation - real implementation would use combinatorics
        let mut total = n_features;
        for d in 2..=degree {
            total += (n_features as f64).powi(d as i32) as usize / d; // Approximation
        }
        total
    }

    /// Generate polynomial combinations
    fn generate_combinations(
        &self,
        data: &ArrayView2<Float>,
        result: &mut ArrayViewMut2<Float>,
        degree: usize,
        start_idx: usize,
    ) -> SklResult<usize> {
        let (n_rows, n_cols) = data.dim();
        let mut feature_idx = start_idx;

        // Simplified: just add squared features for degree 2
        if degree == 2 {
            for col in 0..n_cols {
                let column = data.column(col);
                let squared = self.simd_ops.elementwise_op(&column, |x| x * x)?;
                result.column_mut(feature_idx).assign(&squared);
                feature_idx += 1;
            }
        }

        Ok(feature_idx - start_idx)
    }
}

/// Cache-friendly data layouts for SIMD operations
pub struct SimdDataLayout {
    config: SimdConfig,
}

impl SimdDataLayout {
    /// Create new SIMD data layout optimizer
    #[must_use]
    pub fn new(config: SimdConfig) -> Self {
        Self { config }
    }

    /// Transpose matrix for better cache locality
    #[must_use]
    pub fn transpose_for_simd(&self, data: &ArrayView2<Float>) -> Array2<Float> {
        data.t().to_owned()
    }

    /// Reshape data for optimal SIMD processing
    #[must_use]
    pub fn optimize_layout(&self, data: &ArrayView2<Float>) -> Array2<Float> {
        let (rows, cols) = data.dim();

        // Pad columns to vector boundary if beneficial
        let padded_cols = if cols % self.config.vector_width != 0 {
            ((cols / self.config.vector_width) + 1) * self.config.vector_width
        } else {
            cols
        };

        if padded_cols > cols {
            let mut padded = Array2::zeros((rows, padded_cols));
            padded.slice_mut(s![.., ..cols]).assign(data);
            padded
        } else {
            data.to_owned()
        }
    }

    /// Create memory-efficient chunks for parallel processing
    #[must_use]
    pub fn create_chunks(
        &self,
        data: &ArrayView2<Float>,
        chunk_size: Option<usize>,
    ) -> Vec<Array2<Float>> {
        let (rows, _cols) = data.dim();
        let chunk_sz = chunk_size.unwrap_or(self.config.simd_threshold);

        let mut chunks = Vec::new();
        for start in (0..rows).step_by(chunk_sz) {
            let end = std::cmp::min(start + chunk_sz, rows);
            let chunk = data.slice(s![start..end, ..]).to_owned();
            chunks.push(chunk);
        }

        chunks
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_simd_config() {
        let config = SimdConfig::default();
        assert!(config.vector_width >= 4);
        assert!(config.alignment >= 16);
        assert!(config.simd_threshold > 0);
    }

    #[test]
    fn test_simd_add_arrays() {
        let simd_ops = SimdOps::default();
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![5.0, 6.0, 7.0, 8.0];

        let result = simd_ops.add_arrays(&a.view(), &b.view()).unwrap();
        let expected = array![6.0, 8.0, 10.0, 12.0];

        assert!((result - expected).mapv(|x| x.abs()).sum() < 1e-6);
    }

    #[test]
    fn test_dot_product() {
        let simd_ops = SimdOps::default();
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 3.0, 4.0, 5.0];

        let result = simd_ops.dot_product(&a.view(), &b.view()).unwrap();
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0; // = 40.0

        assert!((result - expected).abs() < 1e-6);
    }

    #[test]
    fn test_matrix_multiply() {
        let simd_ops = SimdOps::default();
        let a = array![[1.0, 2.0], [3.0, 4.0]];
        let b = array![[5.0, 6.0], [7.0, 8.0]];

        let result = simd_ops.matrix_multiply(&a.view(), &b.view()).unwrap();
        let expected = array![[19.0, 22.0], [43.0, 50.0]];

        assert!((result - expected).mapv(|x| x.abs()).sum() < 1e-6);
    }

    #[test]
    fn test_scale() {
        let simd_ops = SimdOps::default();
        let a = array![1.0, 2.0, 3.0, 4.0];
        let scale_factor = 2.0;

        let result = simd_ops.scale(&a.view(), scale_factor).unwrap();
        let expected = array![2.0, 4.0, 6.0, 8.0];

        assert!((result - expected).mapv(|x| x.abs()).sum() < 1e-6);
    }

    #[test]
    fn test_normalize_l2() {
        let simd_ops = SimdOps::default();
        let a = array![3.0, 4.0, 0.0];

        let result = simd_ops.normalize_l2(&a.view()).unwrap();
        let norm = (3.0f32 * 3.0 + 4.0 * 4.0 + 0.0 * 0.0).sqrt(); // = 5.0
        let expected = array![3.0 / 5.0, 4.0 / 5.0, 0.0 / 5.0];

        assert!((result - expected).mapv(|x| x.abs()).sum() < 1e-6);
    }

    #[test]
    fn test_feature_standardize() {
        let config = SimdConfig::default();
        let feature_ops = SimdFeatureOps::new(config);

        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let result = feature_ops.standardize(&data.view()).unwrap();

        // Check that each column has approximately zero mean and unit variance
        for col in 0..result.ncols() {
            let column = result.column(col);
            let mean = column.sum() / column.len() as Float;
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / column.len() as Float;

            assert!(mean.abs() < 1e-10, "Mean should be approximately zero");
            assert!(
                (variance - 1.0).abs() < 1e-6,
                "Variance should be approximately one"
            );
        }
    }

    #[test]
    fn test_min_max_scale() {
        let config = SimdConfig::default();
        let feature_ops = SimdFeatureOps::new(config);

        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let result = feature_ops.min_max_scale(&data.view(), (0.0, 1.0)).unwrap();

        // Check that values are in [0, 1] range
        for &val in result.iter() {
            assert!(val >= 0.0 && val <= 1.0, "Values should be in [0, 1] range");
        }

        // Check that min/max values are mapped correctly
        assert!((result[[0, 0]] - 0.0).abs() < 1e-6); // min of first column
        assert!((result[[2, 0]] - 1.0).abs() < 1e-6); // max of first column
    }

    #[test]
    fn test_optimal_chunk_size() {
        let simd_ops = SimdOps::default();
        let chunk_size = simd_ops.optimal_chunk_size(1000);

        assert!(chunk_size > 0);
        assert!(chunk_size % simd_ops.config.vector_width == 0);
    }

    #[test]
    fn test_data_layout_optimization() {
        let config = SimdConfig::default();
        let layout = SimdDataLayout::new(config);

        let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let optimized = layout.optimize_layout(&data.view());

        // Should maintain data integrity
        assert_eq!(optimized.slice(s![.., ..3]), data);
    }
}
