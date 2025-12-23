//! No-std support for embedded systems
//!
//! This module provides no-std compatible implementations of SIMD operations
//! for use in embedded systems and other environments without std.

#![cfg(feature = "no-std")]

use core::mem;
use core::ptr;

/// Error type for no-std environments
#[derive(Debug, Clone)]
pub enum NoStdSimdError {
    /// Input vectors have mismatched dimensions
    DimensionMismatch { expected: usize, actual: usize },
    /// Input data is empty
    EmptyInput,
    /// SIMD operation is not supported on this platform
    UnsupportedPlatform,
    /// Operation is not implemented for this type
    UnsupportedOperation,
    /// Numerical error (overflow, underflow, NaN)
    NumericalError,
    /// Invalid parameter value
    InvalidParameter,
    /// Memory allocation error
    AllocationError,
}

impl core::fmt::Display for NoStdSimdError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            NoStdSimdError::DimensionMismatch { expected, actual } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, got {}",
                    expected, actual
                )
            }
            NoStdSimdError::EmptyInput => write!(f, "Input data is empty"),
            NoStdSimdError::UnsupportedPlatform => {
                write!(f, "SIMD operation not supported on this platform")
            }
            NoStdSimdError::UnsupportedOperation => write!(f, "Unsupported operation"),
            NoStdSimdError::NumericalError => write!(f, "Numerical error"),
            NoStdSimdError::InvalidParameter => write!(f, "Invalid parameter"),
            NoStdSimdError::AllocationError => write!(f, "Memory allocation failed"),
        }
    }
}

/// Result type for no-std operations
pub type NoStdResult<T> = Result<T, NoStdSimdError>;

/// Fixed-size vector for no-std environments
#[derive(Debug, Clone, Copy)]
pub struct FixedVec<T, const N: usize> {
    data: [T; N],
    len: usize,
}

impl<T: Copy + Default, const N: usize> FixedVec<T, N> {
    /// Create a new fixed vector
    pub fn new() -> Self {
        Self {
            data: [T::default(); N],
            len: 0,
        }
    }

    /// Create from slice
    pub fn from_slice(slice: &[T]) -> NoStdResult<Self> {
        if slice.len() > N {
            return Err(NoStdSimdError::InvalidParameter);
        }

        let mut vec = Self::new();
        vec.data[..slice.len()].copy_from_slice(slice);
        vec.len = slice.len();
        Ok(vec)
    }

    /// Push an element
    pub fn push(&mut self, value: T) -> NoStdResult<()> {
        if self.len >= N {
            return Err(NoStdSimdError::AllocationError);
        }

        self.data[self.len] = value;
        self.len += 1;
        Ok(())
    }

    /// Get length
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get as slice
    pub fn as_slice(&self) -> &[T] {
        &self.data[..self.len]
    }

    /// Get as mutable slice
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.data[..self.len]
    }

    /// Clear the vector
    pub fn clear(&mut self) {
        self.len = 0;
    }

    /// Get capacity
    pub fn capacity(&self) -> usize {
        N
    }
}

impl<T: Copy + Default, const N: usize> Default for FixedVec<T, N> {
    fn default() -> Self {
        Self::new()
    }
}

/// No-std compatible SIMD vector operations
pub struct NoStdSimdOps;

impl NoStdSimdOps {
    /// Dot product for no-std
    pub fn dot_product(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        if x.len() != y.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let mut result = 0.0f32;

        // Use SIMD if available on the platform
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if x.len() >= 4 && crate::simd_feature_detected!("sse") {
                return unsafe { Self::dot_product_sse(x, y) };
            }
        }

        #[cfg(target_arch = "aarch64")]
        {
            if x.len() >= 4 {
                return unsafe { Self::dot_product_neon(x, y) };
            }
        }

        // Fallback to scalar
        for (&xi, &yi) in x.iter().zip(y.iter()) {
            result += xi * yi;
        }

        Ok(result)
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse")]
    unsafe fn dot_product_sse(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let mut sum = _mm_setzero_ps();
        let chunks = x.len() / 4;

        for i in 0..chunks {
            let x_vec = _mm_loadu_ps(x.as_ptr().add(i * 4));
            let y_vec = _mm_loadu_ps(y.as_ptr().add(i * 4));
            let prod = _mm_mul_ps(x_vec, y_vec);
            sum = _mm_add_ps(sum, prod);
        }

        // Horizontal sum
        let mut result = [0.0f32; 4];
        _mm_storeu_ps(result.as_mut_ptr(), sum);
        let mut total = result[0] + result[1] + result[2] + result[3];

        // Handle remaining elements
        for i in (chunks * 4)..x.len() {
            total += x[i] * y[i];
        }

        Ok(total)
    }

    #[cfg(target_arch = "aarch64")]
    unsafe fn dot_product_neon(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        use core::arch::aarch64::*;

        let mut sum = vdupq_n_f32(0.0);
        let chunks = x.len() / 4;

        for i in 0..chunks {
            let x_vec = vld1q_f32(x.as_ptr().add(i * 4));
            let y_vec = vld1q_f32(y.as_ptr().add(i * 4));
            sum = vfmaq_f32(sum, x_vec, y_vec);
        }

        // Horizontal sum
        let mut total = vaddvq_f32(sum);

        // Handle remaining elements
        for i in (chunks * 4)..x.len() {
            total += x[i] * y[i];
        }

        Ok(total)
    }

    /// Vector addition for no-std
    pub fn add(x: &[f32], y: &[f32], result: &mut [f32]) -> NoStdResult<()> {
        if x.len() != y.len() || x.len() != result.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: x.len(),
                actual: result.len(),
            });
        }

        if x.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        // Scalar fallback
        for ((xi, yi), ri) in x.iter().zip(y.iter()).zip(result.iter_mut()) {
            *ri = xi + yi;
        }

        Ok(())
    }

    /// Vector scaling for no-std
    pub fn scale(vector: &mut [f32], scalar: f32) -> NoStdResult<()> {
        if vector.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        for value in vector.iter_mut() {
            *value *= scalar;
        }

        Ok(())
    }

    /// Sum reduction for no-std
    pub fn sum(vector: &[f32]) -> NoStdResult<f32> {
        if vector.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        Ok(vector.iter().sum())
    }

    /// L2 norm for no-std
    pub fn norm(vector: &[f32]) -> NoStdResult<f32> {
        if vector.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let sum_of_squares: f32 = vector.iter().map(|&x| x * x).sum();

        #[cfg(feature = "libm")]
        {
            Ok(libm::sqrtf(sum_of_squares))
        }

        #[cfg(not(feature = "libm"))]
        {
            // Use a simple Newton-Raphson approximation for sqrt
            if sum_of_squares == 0.0 {
                return Ok(0.0);
            }

            let mut x = sum_of_squares;
            for _ in 0..10 {
                x = 0.5 * (x + sum_of_squares / x);
            }
            Ok(x)
        }
    }
}

/// No-std compatible activation functions
pub struct NoStdActivations;

impl NoStdActivations {
    /// ReLU activation for no-std
    pub fn relu(input: &[f32], output: &mut [f32]) -> NoStdResult<()> {
        if input.len() != output.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }

        for (&x, y) in input.iter().zip(output.iter_mut()) {
            *y = if x > 0.0 { x } else { 0.0 };
        }

        Ok(())
    }

    /// Sigmoid activation for no-std (using Taylor series approximation)
    pub fn sigmoid(input: &[f32], output: &mut [f32]) -> NoStdResult<()> {
        if input.len() != output.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }

        for (&x, y) in input.iter().zip(output.iter_mut()) {
            // Clamp input to avoid overflow
            let clamped = if x > 10.0 {
                10.0
            } else if x < -10.0 {
                -10.0
            } else {
                x
            };

            // Use Taylor series approximation for exp
            #[cfg(feature = "libm")]
            {
                *y = 1.0 / (1.0 + libm::expf(-clamped));
            }

            #[cfg(not(feature = "libm"))]
            {
                // Simple polynomial approximation for exp(-x)
                let neg_x = -clamped;
                let exp_approx = if neg_x.abs() < 1.0 {
                    1.0 + neg_x + neg_x * neg_x / 2.0 + neg_x * neg_x * neg_x / 6.0
                } else {
                    if neg_x > 0.0 {
                        0.0001
                    } else {
                        20.0
                    }
                };
                *y = 1.0 / (1.0 + exp_approx);
            }
        }

        Ok(())
    }

    /// Tanh activation for no-std
    pub fn tanh(input: &[f32], output: &mut [f32]) -> NoStdResult<()> {
        if input.len() != output.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: input.len(),
                actual: output.len(),
            });
        }

        for (&x, y) in input.iter().zip(output.iter_mut()) {
            #[cfg(feature = "libm")]
            {
                *y = libm::tanhf(x);
            }

            #[cfg(not(feature = "libm"))]
            {
                // Rational approximation for tanh
                let x_abs = if x >= 0.0 { x } else { -x };
                let tanh_abs = if x_abs < 2.5 {
                    let x2 = x_abs * x_abs;
                    x_abs * (135135.0 + x2 * (17325.0 + x2 * (378.0 + x2)))
                        / (135135.0 + x2 * (62370.0 + x2 * (3150.0 + x2 * 28.0)))
                } else {
                    1.0
                };
                *y = if x >= 0.0 { tanh_abs } else { -tanh_abs };
            }
        }

        Ok(())
    }
}

/// No-std compatible memory management
pub struct NoStdMemory;

impl NoStdMemory {
    /// Aligned memory copy for no-std
    pub unsafe fn aligned_copy(src: *const f32, dst: *mut f32, len: usize, alignment: usize) {
        if alignment <= mem::align_of::<f32>() {
            ptr::copy_nonoverlapping(src, dst, len);
        } else {
            // Use aligned loads/stores if supported
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                if len >= 4 && alignment >= 16 && crate::simd_feature_detected!("sse2") {
                    Self::aligned_copy_sse2(src, dst, len);
                    return;
                }
            }

            // Fallback to regular copy
            ptr::copy_nonoverlapping(src, dst, len);
        }
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "sse2")]
    unsafe fn aligned_copy_sse2(src: *const f32, dst: *mut f32, len: usize) {
        #[cfg(target_arch = "x86")]
        use core::arch::x86::*;
        #[cfg(target_arch = "x86_64")]
        use core::arch::x86_64::*;

        let chunks = len / 4;

        for i in 0..chunks {
            let data = _mm_load_ps(src.add(i * 4));
            _mm_store_ps(dst.add(i * 4), data);
        }

        // Handle remaining elements
        for i in (chunks * 4)..len {
            *dst.add(i) = *src.add(i);
        }
    }

    /// Prefetch memory for no-std
    pub fn prefetch(ptr: *const u8, _locality: i32) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if crate::simd_feature_detected!("sse") {
                unsafe {
                    #[cfg(target_arch = "x86")]
                    use core::arch::x86::*;
                    #[cfg(target_arch = "x86_64")]
                    use core::arch::x86_64::*;

                    _mm_prefetch(ptr as *const i8, _MM_HINT_T0);
                }
            }
        }

        // For other architectures, this is a no-op
        let _ = ptr;
    }
}

/// No-std compatible matrix operations
pub struct NoStdMatrixOps;

impl NoStdMatrixOps {
    /// Matrix-vector multiplication for no-std
    pub fn matvec_multiply(
        matrix: &[&[f32]],
        vector: &[f32],
        result: &mut [f32],
    ) -> NoStdResult<()> {
        if matrix.is_empty() || vector.is_empty() || result.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let rows = matrix.len();
        let cols = matrix[0].len();

        if vector.len() != cols {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: cols,
                actual: vector.len(),
            });
        }

        if result.len() != rows {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: rows,
                actual: result.len(),
            });
        }

        for (i, row) in matrix.iter().enumerate() {
            let mut sum = 0.0f32;
            for (j, &matrix_val) in row.iter().enumerate() {
                sum += matrix_val * vector[j];
            }
            result[i] = sum;
        }

        Ok(())
    }

    /// Matrix transpose for no-std
    pub fn transpose(input: &[&[f32]], output: &mut [FixedVec<f32, 256>]) -> NoStdResult<()> {
        if input.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let _rows = input.len();
        let cols = input[0].len();

        if output.len() != cols {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: cols,
                actual: output.len(),
            });
        }

        // Clear output vectors
        for col_vec in output.iter_mut() {
            col_vec.clear();
        }

        // Transpose operation
        for row in input.iter() {
            for (col_idx, &value) in row.iter().enumerate() {
                output[col_idx].push(value)?;
            }
        }

        Ok(())
    }

    /// Element-wise matrix addition for no-std
    pub fn add_matrices(
        a: &[&[f32]],
        b: &[&[f32]],
        result: &mut [FixedVec<f32, 256>],
    ) -> NoStdResult<()> {
        if a.is_empty() || b.is_empty() || result.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        if a.len() != b.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: a.len(),
                actual: b.len(),
            });
        }

        if result.len() != a.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: a.len(),
                actual: result.len(),
            });
        }

        for ((a_row, b_row), result_row) in a.iter().zip(b.iter()).zip(result.iter_mut()) {
            if a_row.len() != b_row.len() {
                return Err(NoStdSimdError::DimensionMismatch {
                    expected: a_row.len(),
                    actual: b_row.len(),
                });
            }

            result_row.clear();
            for (&a_val, &b_val) in a_row.iter().zip(b_row.iter()) {
                result_row.push(a_val + b_val)?;
            }
        }

        Ok(())
    }
}

/// No-std compatible distance metrics
pub struct NoStdDistanceOps;

impl NoStdDistanceOps {
    /// Euclidean distance for no-std
    pub fn euclidean_distance(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        if x.len() != y.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let sum_of_squares: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).powi(2)).sum();

        NoStdSimdOps::sqrt_scalar(sum_of_squares)
    }

    /// Manhattan distance for no-std
    pub fn manhattan_distance(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        if x.len() != y.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let sum: f32 = x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();

        Ok(sum)
    }

    /// Cosine distance for no-std
    pub fn cosine_distance(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        if x.len() != y.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: x.len(),
                actual: y.len(),
            });
        }

        if x.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let dot_product = NoStdSimdOps::dot_product(x, y)?;
        let norm_x = NoStdSimdOps::norm(x)?;
        let norm_y = NoStdSimdOps::norm(y)?;

        if norm_x == 0.0 || norm_y == 0.0 {
            return Ok(1.0); // Maximum distance
        }

        let cosine_similarity = dot_product / (norm_x * norm_y);
        Ok(1.0 - cosine_similarity)
    }
}

impl NoStdSimdOps {
    /// Standalone square root implementation for no-std
    pub fn sqrt_scalar(x: f32) -> NoStdResult<f32> {
        if x < 0.0 {
            return Err(NoStdSimdError::NumericalError);
        }

        #[cfg(feature = "libm")]
        {
            Ok(libm::sqrtf(x))
        }

        #[cfg(not(feature = "libm"))]
        {
            if x == 0.0 {
                return Ok(0.0);
            }

            // Newton-Raphson method for square root
            let mut result = x;
            for _ in 0..10 {
                result = 0.5 * (result + x / result);
            }
            Ok(result)
        }
    }

    /// Fused multiply-add for no-std (a * b + c)
    pub fn fma(a: &[f32], b: &[f32], c: &[f32], result: &mut [f32]) -> NoStdResult<()> {
        if a.len() != b.len() || a.len() != c.len() || a.len() != result.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: a.len(),
                actual: result.len(),
            });
        }

        if a.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        for (((a_val, b_val), c_val), res) in
            a.iter().zip(b.iter()).zip(c.iter()).zip(result.iter_mut())
        {
            *res = a_val * b_val + c_val;
        }

        Ok(())
    }

    /// Vector normalization for no-std
    pub fn normalize(vector: &[f32], result: &mut [f32]) -> NoStdResult<()> {
        if vector.len() != result.len() {
            return Err(NoStdSimdError::DimensionMismatch {
                expected: vector.len(),
                actual: result.len(),
            });
        }

        if vector.is_empty() {
            return Err(NoStdSimdError::EmptyInput);
        }

        let norm = Self::norm(vector)?;

        if norm == 0.0 {
            // Zero vector case
            for res in result.iter_mut() {
                *res = 0.0;
            }
        } else {
            let inv_norm = 1.0 / norm;
            for (val, res) in vector.iter().zip(result.iter_mut()) {
                *res = val * inv_norm;
            }
        }

        Ok(())
    }
}

/// No-std compatible kernel functions
pub struct NoStdKernelOps;

impl NoStdKernelOps {
    /// RBF (Gaussian) kernel for no-std
    pub fn rbf_kernel(x: &[f32], y: &[f32], gamma: f32) -> NoStdResult<f32> {
        let distance_squared = x
            .iter()
            .zip(y.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>();

        #[cfg(feature = "libm")]
        {
            Ok(libm::expf(-gamma * distance_squared))
        }

        #[cfg(not(feature = "libm"))]
        {
            // Simple exponential approximation
            let x = -gamma * distance_squared;
            let exp_approx = if x.abs() < 1.0 {
                1.0 + x + x * x / 2.0 + x * x * x / 6.0
            } else {
                if x > 0.0 {
                    20.0
                } else {
                    0.0001
                }
            };
            Ok(exp_approx)
        }
    }

    /// Linear kernel for no-std
    pub fn linear_kernel(x: &[f32], y: &[f32]) -> NoStdResult<f32> {
        NoStdSimdOps::dot_product(x, y)
    }

    /// Polynomial kernel for no-std
    pub fn polynomial_kernel(x: &[f32], y: &[f32], degree: u32, coef0: f32) -> NoStdResult<f32> {
        let dot = NoStdSimdOps::dot_product(x, y)?;
        let base = dot + coef0;

        // Simple power computation for no-std
        let mut result = 1.0;
        for _ in 0..degree {
            result *= base;
        }

        Ok(result)
    }
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    #[test]
    fn test_fixed_vec() {
        let mut vec: FixedVec<f32, 10> = FixedVec::new();
        assert_eq!(vec.len(), 0);
        assert!(vec.is_empty());

        vec.push(1.0).unwrap();
        vec.push(2.0).unwrap();
        assert_eq!(vec.len(), 2);
        assert_eq!(vec.as_slice(), &[1.0, 2.0]);
    }

    #[test]
    fn test_fixed_vec_from_slice() {
        let data = [1.0, 2.0, 3.0, 4.0];
        let vec: FixedVec<f32, 10> = FixedVec::from_slice(&data).unwrap();
        assert_eq!(vec.as_slice(), &data);
    }

    #[test]
    fn test_fixed_vec_overflow() {
        let mut vec: FixedVec<f32, 2> = FixedVec::new();
        vec.push(1.0).unwrap();
        vec.push(2.0).unwrap();
        assert!(vec.push(3.0).is_err());
    }

    #[test]
    fn test_nostd_dot_product() {
        let x = [1.0, 2.0, 3.0, 4.0];
        let y = [2.0, 3.0, 4.0, 5.0];
        let result = NoStdSimdOps::dot_product(&x, &y).unwrap();
        assert_eq!(result, 40.0); // 1*2 + 2*3 + 3*4 + 4*5
    }

    #[test]
    fn test_nostd_vector_add() {
        let x = [1.0, 2.0, 3.0];
        let y = [4.0, 5.0, 6.0];
        let mut result = [0.0; 3];

        NoStdSimdOps::add(&x, &y, &mut result).unwrap();
        assert_eq!(result, [5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_nostd_scale() {
        let mut vec = [1.0, 2.0, 3.0];
        NoStdSimdOps::scale(&mut vec, 2.0).unwrap();
        assert_eq!(vec, [2.0, 4.0, 6.0]);
    }

    #[test]
    fn test_nostd_sum() {
        let vec = [1.0, 2.0, 3.0, 4.0];
        let result = NoStdSimdOps::sum(&vec).unwrap();
        assert_eq!(result, 10.0);
    }

    #[test]
    fn test_nostd_norm() {
        let vec = [3.0, 4.0, 0.0];
        let result = NoStdSimdOps::norm(&vec).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_nostd_relu() {
        let input = [-1.0, 0.0, 1.0, 2.0];
        let mut output = [0.0; 4];

        NoStdActivations::relu(&input, &mut output).unwrap();
        assert_eq!(output, [0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_nostd_sigmoid() {
        let input = [0.0, 1.0, -1.0];
        let mut output = [0.0; 3];

        NoStdActivations::sigmoid(&input, &mut output).unwrap();

        // Check that sigmoid(0) ≈ 0.5
        assert!((output[0] - 0.5).abs() < 0.1);
        // Check that sigmoid(1) > 0.5
        assert!(output[1] > 0.5);
        // Check that sigmoid(-1) < 0.5
        assert!(output[2] < 0.5);
    }

    #[test]
    fn test_dimension_mismatch() {
        let x = [1.0, 2.0];
        let y = [1.0, 2.0, 3.0];

        assert!(NoStdSimdOps::dot_product(&x, &y).is_err());
    }

    #[test]
    fn test_empty_input() {
        let empty: &[f32] = &[];
        assert!(NoStdSimdOps::sum(empty).is_err());
    }

    #[test]
    fn test_nostd_matrix_vector_multiply() {
        let row1 = [1.0, 2.0, 3.0];
        let row2 = [4.0, 5.0, 6.0];
        let matrix = [row1.as_slice(), row2.as_slice()];
        let vector = [2.0, 3.0, 4.0];
        let mut result = [0.0; 2];

        NoStdMatrixOps::matvec_multiply(&matrix, &vector, &mut result).unwrap();
        assert_eq!(result, [20.0, 47.0]); // [1*2+2*3+3*4, 4*2+5*3+6*4]
    }

    #[test]
    fn test_nostd_matrix_transpose() {
        let row1 = [1.0, 2.0, 3.0];
        let row2 = [4.0, 5.0, 6.0];
        let matrix = [row1.as_slice(), row2.as_slice()];
        let mut output = [FixedVec::<f32, 256>::new(); 3];

        NoStdMatrixOps::transpose(&matrix, &mut output).unwrap();
        assert_eq!(output[0].as_slice(), &[1.0, 4.0]);
        assert_eq!(output[1].as_slice(), &[2.0, 5.0]);
        assert_eq!(output[2].as_slice(), &[3.0, 6.0]);
    }

    #[test]
    fn test_nostd_matrix_addition() {
        let a_row1 = [1.0, 2.0];
        let a_row2 = [3.0, 4.0];
        let a = [a_row1.as_slice(), a_row2.as_slice()];

        let b_row1 = [5.0, 6.0];
        let b_row2 = [7.0, 8.0];
        let b = [b_row1.as_slice(), b_row2.as_slice()];

        let mut result = [FixedVec::<f32, 256>::new(); 2];

        NoStdMatrixOps::add_matrices(&a, &b, &mut result).unwrap();
        assert_eq!(result[0].as_slice(), &[6.0, 8.0]);
        assert_eq!(result[1].as_slice(), &[10.0, 12.0]);
    }

    #[test]
    fn test_nostd_euclidean_distance() {
        let x = [0.0, 0.0];
        let y = [3.0, 4.0];

        let result = NoStdDistanceOps::euclidean_distance(&x, &y).unwrap();
        assert!((result - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_nostd_manhattan_distance() {
        let x = [1.0, 2.0];
        let y = [4.0, 6.0];

        let result = NoStdDistanceOps::manhattan_distance(&x, &y).unwrap();
        assert_eq!(result, 7.0); // |1-4| + |2-6| = 3 + 4 = 7
    }

    #[test]
    fn test_nostd_cosine_distance() {
        let x = [1.0, 0.0];
        let y = [0.0, 1.0];

        let result = NoStdDistanceOps::cosine_distance(&x, &y).unwrap();
        assert!((result - 1.0).abs() < 1e-6); // Orthogonal vectors
    }

    #[test]
    fn test_nostd_fma() {
        let a = [1.0, 2.0, 3.0];
        let b = [2.0, 3.0, 4.0];
        let c = [1.0, 1.0, 1.0];
        let mut result = [0.0; 3];

        NoStdSimdOps::fma(&a, &b, &c, &mut result).unwrap();
        assert_eq!(result, [3.0, 7.0, 13.0]); // a[i] * b[i] + c[i]
    }

    #[test]
    fn test_nostd_normalize() {
        let vector = [3.0, 4.0, 0.0];
        let mut result = [0.0; 3];

        NoStdSimdOps::normalize(&vector, &mut result).unwrap();
        let expected_norm = 5.0; // sqrt(3^2 + 4^2)
        let expected = [3.0 / expected_norm, 4.0 / expected_norm, 0.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert!((r - e).abs() < 1e-6);
        }
    }

    #[test]
    fn test_nostd_normalize_zero_vector() {
        let vector = [0.0, 0.0, 0.0];
        let mut result = [0.0; 3];

        NoStdSimdOps::normalize(&vector, &mut result).unwrap();
        assert_eq!(result, [0.0, 0.0, 0.0]);
    }

    #[test]
    fn test_nostd_sqrt_scalar() {
        assert_eq!(NoStdSimdOps::sqrt_scalar(0.0).unwrap(), 0.0);
        assert!((NoStdSimdOps::sqrt_scalar(4.0).unwrap() - 2.0).abs() < 1e-6);
        assert!((NoStdSimdOps::sqrt_scalar(9.0).unwrap() - 3.0).abs() < 1e-6);
        assert!(NoStdSimdOps::sqrt_scalar(-1.0).is_err());
    }

    #[test]
    fn test_nostd_rbf_kernel() {
        let x = [1.0, 2.0];
        let y = [1.0, 2.0];

        let result = NoStdKernelOps::rbf_kernel(&x, &y, 1.0).unwrap();
        assert!((result - 1.0).abs() < 0.1); // Same vectors should give ~1.0
    }

    #[test]
    fn test_nostd_linear_kernel() {
        let x = [1.0, 2.0, 3.0];
        let y = [2.0, 3.0, 4.0];

        let result = NoStdKernelOps::linear_kernel(&x, &y).unwrap();
        assert_eq!(result, 20.0); // 1*2 + 2*3 + 3*4 = 20
    }

    #[test]
    fn test_nostd_polynomial_kernel() {
        let x = [1.0, 2.0];
        let y = [2.0, 3.0];

        let result = NoStdKernelOps::polynomial_kernel(&x, &y, 2, 1.0).unwrap();
        let dot = 8.0f32; // 1*2 + 2*3 = 8
        let expected = (dot + 1.0f32).powi(2); // (8 + 1)^2 = 81
        assert_eq!(result, expected);
    }

    #[test]
    fn test_nostd_dimension_mismatches() {
        let x = [1.0, 2.0];
        let y = [1.0, 2.0, 3.0];

        assert!(NoStdDistanceOps::euclidean_distance(&x, &y).is_err());
        assert!(NoStdDistanceOps::manhattan_distance(&x, &y).is_err());
        assert!(NoStdDistanceOps::cosine_distance(&x, &y).is_err());
    }

    #[test]
    fn test_nostd_empty_inputs() {
        let empty: &[f32] = &[];

        assert!(NoStdDistanceOps::euclidean_distance(empty, empty).is_err());
        assert!(NoStdDistanceOps::manhattan_distance(empty, empty).is_err());
        assert!(NoStdDistanceOps::cosine_distance(empty, empty).is_err());
    }

    #[test]
    fn test_nostd_large_operations() {
        // Test with larger arrays to ensure scalability
        let size = 100;
        let x: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let y: Vec<f32> = (0..size).map(|i| (i + 1) as f32).collect();

        let euclidean = NoStdDistanceOps::euclidean_distance(&x, &y).unwrap();
        let manhattan = NoStdDistanceOps::manhattan_distance(&x, &y).unwrap();
        let cosine = NoStdDistanceOps::cosine_distance(&x, &y).unwrap();

        assert!(euclidean > 0.0);
        assert_eq!(manhattan, size as f32); // Each element differs by 1
        assert!(cosine < 1.0 && cosine >= 0.0);
    }
}
