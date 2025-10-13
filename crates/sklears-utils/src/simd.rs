//! SIMD (Single Instruction, Multiple Data) optimizations for high-performance computing
//!
//! This module provides SIMD-optimized utility functions for common operations
//! in machine learning workloads, offering significant performance improvements
//! for vectorized operations.

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// SIMD-optimized vector operations for f32 slices
pub struct SimdF32Ops;

impl SimdF32Ops {
    /// Compute dot product of two f32 slices using optimized SIMD processing
    pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if SimdCapabilities::has_avx2() {
                return unsafe { Self::avx2_dot_product(a, b) };
            } else if SimdCapabilities::has_sse41() {
                return unsafe { Self::sse_dot_product(a, b) };
            }
        }

        // Fallback to optimized scalar implementation
        Self::scalar_dot_product(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_dot_product(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 8; // AVX2 processes 8 f32s at once
        let mut result = _mm256_setzero_ps();

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 8 elements at a time with AVX2
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm256_loadu_ps(chunk_a.as_ptr());
            let vb = _mm256_loadu_ps(chunk_b.as_ptr());
            let prod = _mm256_mul_ps(va, vb);
            result = _mm256_add_ps(result, prod);
        }

        // Horizontal sum of AVX2 register
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            final_result += a_val * b_val;
        }

        final_result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_dot_product(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 4; // SSE processes 4 f32s at once
        let mut result = _mm_setzero_ps();

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 4 elements at a time with SSE
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm_loadu_ps(chunk_a.as_ptr());
            let vb = _mm_loadu_ps(chunk_b.as_ptr());
            let prod = _mm_mul_ps(va, vb);
            result = _mm_add_ps(result, prod);
        }

        // Horizontal sum of SSE register
        let mut sum_array = [0.0f32; 4];
        _mm_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            final_result += a_val * b_val;
        }

        final_result
    }

    fn scalar_dot_product(a: &[f32], b: &[f32]) -> f32 {
        // Use chunked processing for better performance
        const CHUNK_SIZE: usize = 8;
        let mut result = 0.0f32;

        // Process chunks
        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            for i in 0..CHUNK_SIZE {
                result += chunk_a[i] * chunk_b[i];
            }
        }

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            result += a_val * b_val;
        }

        result
    }

    /// Add two f32 vectors using optimized SIMD processing
    pub fn add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
        assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
        assert_eq!(
            a.len(),
            result.len(),
            "Result vector must have the same length as inputs"
        );

        #[cfg(target_arch = "x86_64")]
        {
            if SimdCapabilities::has_avx2() {
                unsafe { Self::avx2_add_vectors(a, b, result) };
                return;
            } else if SimdCapabilities::has_sse41() {
                unsafe { Self::sse_add_vectors(a, b, result) };
                return;
            }
        }

        // Fallback to scalar implementation
        Self::scalar_add_vectors(a, b, result);
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
        const CHUNK_SIZE: usize = 8; // AVX2 processes 8 f32s at once
        let len = a.len();
        let chunk_count = len / CHUNK_SIZE;
        let remainder_start = chunk_count * CHUNK_SIZE;

        // Process 8 elements at a time with AVX2
        for i in 0..chunk_count {
            let start = i * CHUNK_SIZE;
            let va = _mm256_loadu_ps(a.as_ptr().add(start));
            let vb = _mm256_loadu_ps(b.as_ptr().add(start));
            let vresult = _mm256_add_ps(va, vb);
            _mm256_storeu_ps(result.as_mut_ptr().add(start), vresult);
        }

        // Handle remaining elements
        for i in remainder_start..len {
            result[i] = a[i] + b[i];
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
        const CHUNK_SIZE: usize = 4; // SSE processes 4 f32s at once
        let len = a.len();
        let chunk_count = len / CHUNK_SIZE;
        let remainder_start = chunk_count * CHUNK_SIZE;

        // Process 4 elements at a time with SSE
        for i in 0..chunk_count {
            let start = i * CHUNK_SIZE;
            let va = _mm_loadu_ps(a.as_ptr().add(start));
            let vb = _mm_loadu_ps(b.as_ptr().add(start));
            let vresult = _mm_add_ps(va, vb);
            _mm_storeu_ps(result.as_mut_ptr().add(start), vresult);
        }

        // Handle remaining elements
        for i in remainder_start..len {
            result[i] = a[i] + b[i];
        }
    }

    fn scalar_add_vectors(a: &[f32], b: &[f32], result: &mut [f32]) {
        const CHUNK_SIZE: usize = 8;
        let len = a.len();
        let chunk_count = len / CHUNK_SIZE;
        let remainder_start = chunk_count * CHUNK_SIZE;

        // Process chunks
        for i in 0..chunk_count {
            let start = i * CHUNK_SIZE;
            for j in 0..CHUNK_SIZE {
                result[start + j] = a[start + j] + b[start + j];
            }
        }

        // Handle remaining elements
        for i in remainder_start..len {
            result[i] = a[i] + b[i];
        }
    }

    /// Multiply f32 vector by scalar using optimized processing
    pub fn scalar_multiply(vector: &[f32], scalar: f32, result: &mut [f32]) {
        assert_eq!(
            vector.len(),
            result.len(),
            "Vector and result must have the same length"
        );

        const CHUNK_SIZE: usize = 8;
        let len = vector.len();
        let chunk_len = len - (len % CHUNK_SIZE);

        // Process chunks
        for i in (0..chunk_len).step_by(CHUNK_SIZE) {
            for j in 0..CHUNK_SIZE {
                result[i + j] = vector[i + j] * scalar;
            }
        }

        // Handle remaining elements
        for i in chunk_len..len {
            result[i] = vector[i] * scalar;
        }
    }

    /// Compute L2 norm squared using optimized processing
    pub fn norm_squared(vector: &[f32]) -> f32 {
        if vector.is_empty() {
            return 0.0;
        }

        const CHUNK_SIZE: usize = 8;
        let mut result = 0.0f32;

        // Process chunks
        let chunks = vector.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for &val in chunk {
                result += val * val;
            }
        }

        // Handle remaining elements
        for &val in remainder {
            result += val * val;
        }

        result
    }

    /// Apply exponential function element-wise using fast approximation
    pub fn exp_approx(input: &[f32], result: &mut [f32]) {
        assert_eq!(
            input.len(),
            result.len(),
            "Input and result must have the same length"
        );

        for (input_val, result_val) in input.iter().zip(result.iter_mut()) {
            *result_val = Self::fast_exp(*input_val);
        }
    }

    /// Fast exponential approximation
    fn fast_exp(x: f32) -> f32 {
        // Fast exp approximation using polynomial approximation
        // e^x ≈ 1 + x + x²/2 + x³/6 + x⁴/24 for small x
        if x.abs() < 1.0 {
            let x2 = x * x;
            let x3 = x2 * x;
            let x4 = x3 * x;
            1.0 + x + x2 * 0.5 + x3 / 6.0 + x4 / 24.0
        } else {
            x.exp() // Fall back to standard library for larger values
        }
    }

    /// Compute softmax using SIMD
    pub fn softmax(input: &[f32], result: &mut [f32]) {
        assert_eq!(
            input.len(),
            result.len(),
            "Input and result must have the same length"
        );

        if input.is_empty() {
            return;
        }

        // Find maximum for numerical stability
        let max_val = input.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

        // Compute exp(x - max) and sum
        let mut temp = vec![0.0f32; input.len()];
        for (i, &val) in input.iter().enumerate() {
            temp[i] = (val - max_val).exp();
        }

        let sum: f32 = temp.iter().sum();
        let inv_sum = 1.0 / sum;

        // Normalize
        Self::scalar_multiply(&temp, inv_sum, result);
    }
}

/// SIMD-optimized vector operations for f64 slices
pub struct SimdF64Ops;

impl SimdF64Ops {
    /// Compute dot product of two f64 slices using optimized chunked processing
    pub fn dot_product(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        // Use chunked processing for better performance
        const CHUNK_SIZE: usize = 4;
        let mut result = 0.0f64;

        // Process chunks
        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            for i in 0..CHUNK_SIZE {
                result += chunk_a[i] * chunk_b[i];
            }
        }

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            result += a_val * b_val;
        }

        result
    }

    /// Matrix-vector multiplication using SIMD
    pub fn matrix_vector_multiply(
        matrix: &[f64],
        vector: &[f64],
        result: &mut [f64],
        rows: usize,
        cols: usize,
    ) {
        assert_eq!(
            matrix.len(),
            rows * cols,
            "Matrix size must match dimensions"
        );
        assert_eq!(
            vector.len(),
            cols,
            "Vector length must match matrix columns"
        );
        assert_eq!(result.len(), rows, "Result length must match matrix rows");

        for (i, result_item) in result.iter_mut().enumerate().take(rows) {
            let row_start = i * cols;
            let row = &matrix[row_start..row_start + cols];
            *result_item = Self::dot_product(row, vector);
        }
    }

    /// Add two f64 vectors using optimized chunked processing
    pub fn add_vectors(a: &[f64], b: &[f64], result: &mut [f64]) {
        assert_eq!(a.len(), b.len(), "Input vectors must have the same length");
        assert_eq!(
            a.len(),
            result.len(),
            "Result vector must have the same length as inputs"
        );

        const CHUNK_SIZE: usize = 4;
        let len = a.len();
        let chunk_len = len - (len % CHUNK_SIZE);

        // Process chunks
        for i in (0..chunk_len).step_by(CHUNK_SIZE) {
            for j in 0..CHUNK_SIZE {
                result[i + j] = a[i + j] + b[i + j];
            }
        }

        // Handle remaining elements
        for i in chunk_len..len {
            result[i] = a[i] + b[i];
        }
    }
}

/// SIMD-optimized matrix operations
pub struct SimdMatrixOps;

impl SimdMatrixOps {
    /// Transpose a matrix using SIMD optimizations
    pub fn transpose_f32(input: &[f32], output: &mut [f32], rows: usize, cols: usize) {
        assert_eq!(input.len(), rows * cols);
        assert_eq!(output.len(), rows * cols);

        // For small matrices, use simple transpose
        if rows <= 4 || cols <= 4 {
            for i in 0..rows {
                for j in 0..cols {
                    output[j * rows + i] = input[i * cols + j];
                }
            }
            return;
        }

        // Block transpose for better cache performance
        const BLOCK_SIZE: usize = 8;

        for i in (0..rows).step_by(BLOCK_SIZE) {
            for j in (0..cols).step_by(BLOCK_SIZE) {
                let max_i = (i + BLOCK_SIZE).min(rows);
                let max_j = (j + BLOCK_SIZE).min(cols);

                for ii in i..max_i {
                    for jj in j..max_j {
                        output[jj * rows + ii] = input[ii * cols + jj];
                    }
                }
            }
        }
    }

    /// Matrix multiplication using SIMD
    pub fn matrix_multiply_f32(a: &[f32], b: &[f32], c: &mut [f32], m: usize, n: usize, k: usize) {
        assert_eq!(a.len(), m * k);
        assert_eq!(b.len(), k * n);
        assert_eq!(c.len(), m * n);

        // Initialize result matrix
        c.fill(0.0);

        // Block matrix multiplication for better cache performance
        const BLOCK_SIZE: usize = 64;

        for i in (0..m).step_by(BLOCK_SIZE) {
            for j in (0..n).step_by(BLOCK_SIZE) {
                for l in (0..k).step_by(BLOCK_SIZE) {
                    let max_i = (i + BLOCK_SIZE).min(m);
                    let max_j = (j + BLOCK_SIZE).min(n);
                    let max_l = (l + BLOCK_SIZE).min(k);

                    for ii in i..max_i {
                        for jj in j..max_j {
                            let mut sum = 0.0f32;
                            let a_row = &a[ii * k + l..ii * k + max_l];
                            let b_col: Vec<f32> = (l..max_l).map(|ll| b[ll * n + jj]).collect();

                            sum += SimdF32Ops::dot_product(a_row, &b_col);
                            c[ii * n + jj] += sum;
                        }
                    }
                }
            }
        }
    }
}

/// SIMD-optimized statistical operations
pub struct SimdStatsOps;

impl SimdStatsOps {
    /// Compute mean using optimized chunked processing
    pub fn mean_f32(data: &[f32]) -> f32 {
        if data.is_empty() {
            return 0.0;
        }

        const CHUNK_SIZE: usize = 8;
        let mut result = 0.0f32;

        // Process chunks
        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for &val in chunk {
                result += val;
            }
        }

        // Handle remaining elements
        for &val in remainder {
            result += val;
        }

        result / data.len() as f32
    }

    /// Compute variance using optimized chunked processing
    pub fn variance_f32(data: &[f32]) -> f32 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mean = Self::mean_f32(data);
        const CHUNK_SIZE: usize = 8;
        let mut result = 0.0f32;

        // Process chunks
        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for &val in chunk {
                let diff = val - mean;
                result += diff * diff;
            }
        }

        // Handle remaining elements
        for &val in remainder {
            let diff = val - mean;
            result += diff * diff;
        }

        result / (data.len() - 1) as f32
    }

    /// Find minimum and maximum values using optimized chunked processing
    pub fn min_max_f32(data: &[f32]) -> Option<(f32, f32)> {
        if data.is_empty() {
            return None;
        }

        const CHUNK_SIZE: usize = 8;
        let mut min_val = f32::INFINITY;
        let mut max_val = f32::NEG_INFINITY;

        // Process chunks
        let chunks = data.chunks_exact(CHUNK_SIZE);
        let remainder = chunks.remainder();

        for chunk in chunks {
            for &val in chunk {
                min_val = min_val.min(val);
                max_val = max_val.max(val);
            }
        }

        // Handle remaining elements
        for &val in remainder {
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        Some((min_val, max_val))
    }
}

/// SIMD-optimized distance calculations
pub struct SimdDistanceOps;

impl SimdDistanceOps {
    /// Compute Euclidean distance between two vectors using SIMD optimization
    pub fn euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if SimdCapabilities::has_avx2() {
                return unsafe { Self::avx2_euclidean_distance(a, b) };
            } else if SimdCapabilities::has_sse41() {
                return unsafe { Self::sse_euclidean_distance(a, b) };
            }
        }

        // Fallback to scalar implementation
        Self::scalar_euclidean_distance(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 8; // AVX2 processes 8 f32s at once
        let mut result = _mm256_setzero_ps();

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 8 elements at a time with AVX2
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm256_loadu_ps(chunk_a.as_ptr());
            let vb = _mm256_loadu_ps(chunk_b.as_ptr());
            let diff = _mm256_sub_ps(va, vb);
            let squared = _mm256_mul_ps(diff, diff);
            result = _mm256_add_ps(result, squared);
        }

        // Horizontal sum of AVX2 register
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            let diff = a_val - b_val;
            final_result += diff * diff;
        }

        final_result.sqrt()
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 4; // SSE processes 4 f32s at once
        let mut result = _mm_setzero_ps();

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 4 elements at a time with SSE
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm_loadu_ps(chunk_a.as_ptr());
            let vb = _mm_loadu_ps(chunk_b.as_ptr());
            let diff = _mm_sub_ps(va, vb);
            let squared = _mm_mul_ps(diff, diff);
            result = _mm_add_ps(result, squared);
        }

        // Horizontal sum of SSE register
        let mut sum_array = [0.0f32; 4];
        _mm_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            let diff = a_val - b_val;
            final_result += diff * diff;
        }

        final_result.sqrt()
    }

    fn scalar_euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut result = 0.0f32;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            let diff = a_val - b_val;
            result += diff * diff;
        }
        result.sqrt()
    }

    /// Compute Manhattan distance using SIMD optimization
    pub fn manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        #[cfg(target_arch = "x86_64")]
        {
            if SimdCapabilities::has_avx2() {
                return unsafe { Self::avx2_manhattan_distance(a, b) };
            } else if SimdCapabilities::has_sse41() {
                return unsafe { Self::sse_manhattan_distance(a, b) };
            }
        }

        // Fallback to scalar implementation
        Self::scalar_manhattan_distance(a, b)
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2")]
    unsafe fn avx2_manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 8; // AVX2 processes 8 f32s at once
        let mut result = _mm256_setzero_ps();
        let sign_mask = _mm256_set1_ps(-0.0f32); // Sign bit mask for abs

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 8 elements at a time with AVX2
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm256_loadu_ps(chunk_a.as_ptr());
            let vb = _mm256_loadu_ps(chunk_b.as_ptr());
            let diff = _mm256_sub_ps(va, vb);
            let abs_diff = _mm256_andnot_ps(sign_mask, diff); // Absolute value using bitwise AND
            result = _mm256_add_ps(result, abs_diff);
        }

        // Horizontal sum of AVX2 register
        let mut sum_array = [0.0f32; 8];
        _mm256_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            final_result += (a_val - b_val).abs();
        }

        final_result
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "sse4.1")]
    unsafe fn sse_manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
        const CHUNK_SIZE: usize = 4; // SSE processes 4 f32s at once
        let mut result = _mm_setzero_ps();
        let sign_mask = _mm_set1_ps(-0.0f32); // Sign bit mask for abs

        let chunks_a = a.chunks_exact(CHUNK_SIZE);
        let chunks_b = b.chunks_exact(CHUNK_SIZE);
        let remainder_a = chunks_a.remainder();
        let remainder_b = chunks_b.remainder();

        // Process 4 elements at a time with SSE
        for (chunk_a, chunk_b) in chunks_a.zip(chunks_b) {
            let va = _mm_loadu_ps(chunk_a.as_ptr());
            let vb = _mm_loadu_ps(chunk_b.as_ptr());
            let diff = _mm_sub_ps(va, vb);
            let abs_diff = _mm_andnot_ps(sign_mask, diff); // Absolute value using bitwise AND
            result = _mm_add_ps(result, abs_diff);
        }

        // Horizontal sum of SSE register
        let mut sum_array = [0.0f32; 4];
        _mm_storeu_ps(sum_array.as_mut_ptr(), result);
        let mut final_result = sum_array.iter().sum::<f32>();

        // Handle remaining elements
        for (a_val, b_val) in remainder_a.iter().zip(remainder_b.iter()) {
            final_result += (a_val - b_val).abs();
        }

        final_result
    }

    fn scalar_manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
        let mut result = 0.0f32;
        for (a_val, b_val) in a.iter().zip(b.iter()) {
            result += (a_val - b_val).abs();
        }
        result
    }

    /// Compute cosine similarity using SIMD
    pub fn cosine_similarity_f32(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "Vectors must have the same length");

        if a.is_empty() {
            return 0.0;
        }

        let dot_product = SimdF32Ops::dot_product(a, b);
        let norm_a = SimdF32Ops::norm_squared(a).sqrt();
        let norm_b = SimdF32Ops::norm_squared(b).sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot_product / (norm_a * norm_b)
        }
    }
}

/// Utility functions for SIMD capability detection
pub struct SimdCapabilities;

impl SimdCapabilities {
    /// Check if AVX is available
    #[cfg(target_arch = "x86_64")]
    pub fn has_avx() -> bool {
        std::arch::is_x86_feature_detected!("avx")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_avx() -> bool {
        false
    }

    /// Check if AVX2 is available
    #[cfg(target_arch = "x86_64")]
    pub fn has_avx2() -> bool {
        std::arch::is_x86_feature_detected!("avx2")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_avx2() -> bool {
        false
    }

    /// Check if SSE4.1 is available
    #[cfg(target_arch = "x86_64")]
    pub fn has_sse41() -> bool {
        std::arch::is_x86_feature_detected!("sse4.1")
    }

    #[cfg(not(target_arch = "x86_64"))]
    pub fn has_sse41() -> bool {
        false
    }

    /// Get a summary of available SIMD capabilities
    pub fn capabilities_summary() -> String {
        format!(
            "SIMD Capabilities: AVX={}, AVX2={}, SSE4.1={}",
            Self::has_avx(),
            Self::has_avx2(),
            Self::has_sse41()
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_simd_dot_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];

        let result = SimdF32Ops::dot_product(&a, &b);
        let expected: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_vector_addition() {
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b = vec![9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let mut result = vec![0.0; a.len()];

        SimdF32Ops::add_vectors(&a, &b, &mut result);

        for i in 0..a.len() {
            assert_relative_eq!(result[i], a[i] + b[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_scalar_multiply() {
        let vector = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let scalar = 2.5;
        let mut result = vec![0.0; vector.len()];

        SimdF32Ops::scalar_multiply(&vector, scalar, &mut result);

        for i in 0..vector.len() {
            assert_relative_eq!(result[i], vector[i] * scalar, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_norm_squared() {
        let vector = vec![3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let result = SimdF32Ops::norm_squared(&vector);
        let expected: f32 = vector.iter().map(|x| x * x).sum();

        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_simd_stats() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let mean = SimdStatsOps::mean_f32(&data);
        assert_relative_eq!(mean, 5.5, epsilon = 1e-6);

        let variance = SimdStatsOps::variance_f32(&data);
        assert_relative_eq!(variance, 9.166667, epsilon = 1e-5);

        let (min_val, max_val) = SimdStatsOps::min_max_f32(&data).unwrap();
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 10.0);
    }

    #[test]
    fn test_simd_distances() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let euclidean = SimdDistanceOps::euclidean_distance_f32(&a, &b);
        assert_relative_eq!(euclidean, (27.0_f32).sqrt(), epsilon = 1e-6);

        let manhattan = SimdDistanceOps::manhattan_distance_f32(&a, &b);
        assert_relative_eq!(manhattan, 9.0, epsilon = 1e-6);

        let cosine = SimdDistanceOps::cosine_similarity_f32(&a, &b);
        let expected = 32.0 / ((14.0_f32).sqrt() * (77.0_f32).sqrt());
        assert_relative_eq!(cosine, expected, epsilon = 1e-6);
    }

    #[test]
    fn test_matrix_transpose() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3 matrix
        let mut output = vec![0.0; 6];

        SimdMatrixOps::transpose_f32(&input, &mut output, 2, 3);

        let expected = [1.0, 4.0, 2.0, 5.0, 3.0, 6.0]; // 3x2 matrix
        for i in 0..6 {
            assert_relative_eq!(output[i], expected[i], epsilon = 1e-6);
        }
    }

    #[test]
    fn test_simd_capabilities() {
        let summary = SimdCapabilities::capabilities_summary();
        assert!(summary.contains("SIMD Capabilities"));
    }
}
