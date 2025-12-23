//! # SIMD Vector Statistical and Reduction Operations
//!
//! High-performance SIMD-optimized statistical computations and reduction operations.
//! Provides functions for computing statistical measures, norms, and vector reductions
//! with optimal performance on modern CPU architectures.
//!
//! ## Features
//!
//! - **Reduction Operations**: Sum, product, min, max with SIMD horizontal reductions
//! - **Statistical Measures**: Mean, variance, standard deviation computations
//! - **Vector Norms**: L1, L2, and squared L2 norms
//! - **Dot Products**: Optimized vector dot product computation
//! - **Multi-Platform SIMD**: SSE2, AVX2, AVX512, NEON optimizations
//! - **Automatic Fallback**: Graceful fallback to scalar implementations
//!
//! ## Performance Notes
//!
//! Statistical operations benefit greatly from SIMD instructions through parallel
//! computation followed by horizontal reduction. The implementations use the most
//! efficient reduction patterns for each target architecture.

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

/// SIMD-optimized sum of all elements in a vector
///
/// Computes the sum of all elements using SIMD horizontal addition.
///
/// # Arguments
/// * `input` - Input vector
///
/// # Returns
/// The sum of all elements as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::sum_vec;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let result = sum_vec(&input);
/// assert_eq!(result, 10.0);
///
/// // Test with empty vector
/// let empty: Vec<f32> = vec![];
/// assert_eq!(sum_vec(&empty), 0.0);
/// ```
pub fn sum_vec(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { sum_vec_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { sum_vec_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { sum_vec_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { sum_vec_neon(input) };
        }
    }

    sum_vec_scalar(input)
}

/// SIMD-optimized product of all elements in a vector
///
/// Computes the product of all elements using SIMD horizontal multiplication.
///
/// # Arguments
/// * `input` - Input vector
///
/// # Returns
/// The product of all elements as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::product_vec;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let result = product_vec(&input);
/// assert_eq!(result, 24.0);
///
/// // Test with empty vector
/// let empty: Vec<f32> = vec![];
/// assert_eq!(product_vec(&empty), 1.0);
/// ```
pub fn product_vec(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 1.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { product_vec_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { product_vec_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { product_vec_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { product_vec_neon(input) };
        }
    }

    product_vec_scalar(input)
}

/// SIMD-optimized minimum value in a vector
///
/// Finds the minimum value using SIMD horizontal reduction.
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// The minimum value as a single f32 value
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::min_vec;
///
/// let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
/// let result = min_vec(&input);
/// assert_eq!(result, 1.0);
/// ```
pub fn min_vec(input: &[f32]) -> f32 {
    assert!(!input.is_empty(), "Input vector must not be empty");

    if input.iter().any(|x| x.is_nan()) {
        return f32::NAN;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { min_vec_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { min_vec_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { min_vec_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { min_vec_neon(input) };
        }
    }

    min_vec_scalar(input)
}

/// SIMD-optimized maximum value in a vector
///
/// Finds the maximum value using SIMD horizontal reduction.
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// The maximum value as a single f32 value
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::max_vec;
///
/// let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
/// let result = max_vec(&input);
/// assert_eq!(result, 5.0);
/// ```
pub fn max_vec(input: &[f32]) -> f32 {
    assert!(!input.is_empty(), "Input vector must not be empty");

    if input.iter().any(|x| x.is_nan()) {
        return f32::NAN;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { max_vec_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { max_vec_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { max_vec_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { max_vec_neon(input) };
        }
    }

    max_vec_scalar(input)
}

/// SIMD-optimized computation of both minimum and maximum values
///
/// Computes both min and max in a single pass for efficiency.
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// A tuple (min, max) containing the minimum and maximum values
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::min_max_vec;
///
/// let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
/// let (min_val, max_val) = min_max_vec(&input);
/// assert_eq!(min_val, 1.0);
/// assert_eq!(max_val, 5.0);
/// ```
pub fn min_max_vec(input: &[f32]) -> (f32, f32) {
    assert!(!input.is_empty(), "Input vector must not be empty");

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { min_max_vec_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { min_max_vec_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { min_max_vec_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { min_max_vec_neon(input) };
        }
    }

    min_max_vec_scalar(input)
}

/// SIMD-optimized arithmetic mean (average) of a vector
///
/// Computes the arithmetic mean using SIMD sum followed by division.
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// The arithmetic mean as a single f32 value
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::mean_vec;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = mean_vec(&input);
/// assert_eq!(result, 3.0);
/// ```
pub fn mean_vec(input: &[f32]) -> f32 {
    assert!(!input.is_empty(), "Input vector must not be empty");

    let sum = sum_vec(input);
    sum / (input.len() as f32)
}

/// SIMD-optimized dot product of two vectors
///
/// Computes the dot product using SIMD multiply-accumulate operations.
///
/// # Arguments
/// * `a` - First input vector
/// * `b` - Second input vector (must have same length as `a`)
///
/// # Returns
/// The dot product as a single f32 value
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = dot_product(&a, &b);
/// assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
///
/// // Test with empty vectors
/// let empty_a: Vec<f32> = vec![];
/// let empty_b: Vec<f32> = vec![];
/// assert_eq!(dot_product(&empty_a, &empty_b), 0.0);
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Input vectors must have the same length");

    if a.is_empty() {
        return 0.0;
    }

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

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { dot_product_neon(a, b) };
        }
    }

    dot_product_scalar(a, b)
}

/// SIMD-optimized L1 norm (Manhattan distance) of a vector
///
/// Computes the sum of absolute values using SIMD operations.
///
/// # Arguments
/// * `input` - Input vector
///
/// # Returns
/// The L1 norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::norm_l1;
///
/// let input = vec![-1.0, 2.0, -3.0, 4.0];
/// let result = norm_l1(&input);
/// assert_eq!(result, 10.0); // |−1| + |2| + |−3| + |4| = 1 + 2 + 3 + 4 = 10
/// ```
pub fn norm_l1(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { norm_l1_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { norm_l1_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { norm_l1_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { norm_l1_neon(input) };
        }
    }

    norm_l1_scalar(input)
}

/// SIMD-optimized squared L2 norm of a vector
///
/// Computes the sum of squared values, which is the squared L2 norm.
/// This is more efficient than L2 norm when the square root is not needed.
///
/// # Arguments
/// * `input` - Input vector
///
/// # Returns
/// The squared L2 norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::norm_l2_squared;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0];
/// let result = norm_l2_squared(&input);
/// assert_eq!(result, 30.0); // 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30
/// ```
pub fn norm_l2_squared(input: &[f32]) -> f32 {
    if input.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx512f") {
            return unsafe { norm_l2_squared_avx512(input) };
        } else if crate::simd_feature_detected!("avx2") {
            return unsafe { norm_l2_squared_avx2(input) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { norm_l2_squared_sse2(input) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { norm_l2_squared_neon(input) };
        }
    }

    norm_l2_squared_scalar(input)
}

/// SIMD-optimized L2 norm (Euclidean distance) of a vector
///
/// Computes the square root of the sum of squared values.
///
/// # Arguments
/// * `input` - Input vector
///
/// # Returns
/// The L2 norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::norm_l2;
///
/// let input = vec![3.0, 4.0];
/// let result = norm_l2(&input);
/// assert_eq!(result, 5.0); // sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5
/// ```
pub fn norm_l2(input: &[f32]) -> f32 {
    norm_l2_squared(input).sqrt()
}

/// SIMD-optimized population variance of a vector
///
/// Computes the population variance using the formula: Var(X) = E[X²] - E[X]²
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// The population variance as a single f32 value
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::variance_vec;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = variance_vec(&input);
/// assert!((result - 2.0).abs() < 1e-6); // Population variance = 2.0
/// ```
pub fn variance_vec(input: &[f32]) -> f32 {
    assert!(!input.is_empty(), "Input vector must not be empty");

    let mean = mean_vec(input);
    let sum_of_squares = norm_l2_squared(input);
    let n = input.len() as f32;

    sum_of_squares / n - mean * mean
}

/// SIMD-optimized standard deviation of a vector
///
/// Computes the population standard deviation as the square root of the variance.
///
/// # Arguments
/// * `input` - Input vector (must not be empty)
///
/// # Returns
/// The standard deviation as a single f32 value
///
/// # Panics
/// Panics if the input vector is empty
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::statistics_ops::std_dev_vec;
///
/// let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
/// let result = std_dev_vec(&input);
/// assert!((result - 1.41421356).abs() < 1e-6); // sqrt(2) ≈ 1.41421356
/// ```
pub fn std_dev_vec(input: &[f32]) -> f32 {
    variance_vec(input).sqrt()
}

// ============================================================================
// Scalar implementations (fallbacks)
// ============================================================================

fn sum_vec_scalar(input: &[f32]) -> f32 {
    input.iter().sum()
}

fn product_vec_scalar(input: &[f32]) -> f32 {
    input.iter().fold(1.0, |acc, &x| acc * x)
}

fn min_vec_scalar(input: &[f32]) -> f32 {
    input
        .iter()
        .fold(input[0], |min, &x| if x < min { x } else { min })
}

fn max_vec_scalar(input: &[f32]) -> f32 {
    input
        .iter()
        .fold(input[0], |max, &x| if x > max { x } else { max })
}

fn min_max_vec_scalar(input: &[f32]) -> (f32, f32) {
    let mut min = input[0];
    let mut max = input[0];
    for &x in &input[1..] {
        if x < min {
            min = x;
        }
        if x > max {
            max = x;
        }
    }
    (min, max)
}

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

fn norm_l1_scalar(input: &[f32]) -> f32 {
    input.iter().map(|&x| x.abs()).sum()
}

fn norm_l2_squared_scalar(input: &[f32]) -> f32 {
    input.iter().map(|&x| x * x).sum()
}

// ============================================================================
// SSE2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn sum_vec_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        sum = _mm_add_ps(sum, chunk);
        i += 4;
    }

    // Horizontal sum
    let temp = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    // Handle remaining elements
    while i < input.len() {
        final_sum += input[i];
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn product_vec_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut product = _mm_set1_ps(1.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        product = _mm_mul_ps(product, chunk);
        i += 4;
    }

    // Horizontal product
    let temp = _mm_mul_ps(product, _mm_movehl_ps(product, product));
    let result = _mm_mul_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_product = _mm_cvtss_f32(result);

    while i < input.len() {
        final_product *= input[i];
        i += 1;
    }

    final_product
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn min_vec_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_val = _mm_load1_ps(&input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        min_val = _mm_min_ps(min_val, chunk);
        i += 4;
    }

    // Horizontal min
    let temp = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
    let result = _mm_min_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_min = _mm_cvtss_f32(result);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        i += 1;
    }

    final_min
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn max_vec_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut max_val = _mm_load1_ps(&input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        max_val = _mm_max_ps(max_val, chunk);
        i += 4;
    }

    // Horizontal max
    let temp = _mm_max_ps(max_val, _mm_movehl_ps(max_val, max_val));
    let result = _mm_max_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_max = _mm_cvtss_f32(result);

    while i < input.len() {
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    final_max
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn min_max_vec_sse2(input: &[f32]) -> (f32, f32) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_val = _mm_load1_ps(&input[0]);
    let mut max_val = _mm_load1_ps(&input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        min_val = _mm_min_ps(min_val, chunk);
        max_val = _mm_max_ps(max_val, chunk);
        i += 4;
    }

    // Horizontal reductions
    let min_temp = _mm_min_ps(min_val, _mm_movehl_ps(min_val, min_val));
    let min_result = _mm_min_ps(min_temp, _mm_shuffle_ps(min_temp, min_temp, 0x01));
    let mut final_min = _mm_cvtss_f32(min_result);

    let max_temp = _mm_max_ps(max_val, _mm_movehl_ps(max_val, max_val));
    let max_result = _mm_max_ps(max_temp, _mm_shuffle_ps(max_temp, max_temp, 0x01));
    let mut final_max = _mm_cvtss_f32(max_result);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    (final_min, final_max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_chunk = _mm_loadu_ps(a.as_ptr().add(i));
        let b_chunk = _mm_loadu_ps(b.as_ptr().add(i));
        let product = _mm_mul_ps(a_chunk, b_chunk);
        sum = _mm_add_ps(sum, product);
        i += 4;
    }

    // Horizontal sum
    let temp = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < a.len() {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn norm_l1_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        let abs_chunk = _mm_and_ps(chunk, abs_mask);
        sum = _mm_add_ps(sum, abs_chunk);
        i += 4;
    }

    // Horizontal sum
    let temp = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < input.len() {
        final_sum += input[i].abs();
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn norm_l2_squared_sse2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = _mm_loadu_ps(input.as_ptr().add(i));
        let squared = _mm_mul_ps(chunk, chunk);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    // Horizontal sum
    let temp = _mm_add_ps(sum, _mm_movehl_ps(sum, sum));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < input.len() {
        final_sum += input[i] * input[i];
        i += 1;
    }

    final_sum
}

// ============================================================================
// AVX2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn sum_vec_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        sum = _mm256_add_ps(sum, chunk);
        i += 8;
    }

    // Extract and sum both 128-bit lanes
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let temp = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < input.len() {
        final_sum += input[i];
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn product_vec_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut product = _mm256_set1_ps(1.0);
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        product = _mm256_mul_ps(product, chunk);
        i += 8;
    }

    // Extract and multiply both 128-bit lanes
    let prod128 = _mm_mul_ps(
        _mm256_extractf128_ps(product, 0),
        _mm256_extractf128_ps(product, 1),
    );
    let temp = _mm_mul_ps(prod128, _mm_movehl_ps(prod128, prod128));
    let result = _mm_mul_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_product = _mm_cvtss_f32(result);

    while i < input.len() {
        final_product *= input[i];
        i += 1;
    }

    final_product
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn min_vec_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_val = _mm256_broadcast_ss(&input[0]);
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        min_val = _mm256_min_ps(min_val, chunk);
        i += 8;
    }

    // Extract and min both 128-bit lanes
    let min128 = _mm_min_ps(
        _mm256_extractf128_ps(min_val, 0),
        _mm256_extractf128_ps(min_val, 1),
    );
    let temp = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    let result = _mm_min_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_min = _mm_cvtss_f32(result);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        i += 1;
    }

    final_min
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn max_vec_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut max_val = _mm256_broadcast_ss(&input[0]);
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        max_val = _mm256_max_ps(max_val, chunk);
        i += 8;
    }

    // Extract and max both 128-bit lanes
    let max128 = _mm_max_ps(
        _mm256_extractf128_ps(max_val, 0),
        _mm256_extractf128_ps(max_val, 1),
    );
    let temp = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let result = _mm_max_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_max = _mm_cvtss_f32(result);

    while i < input.len() {
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    final_max
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn min_max_vec_avx2(input: &[f32]) -> (f32, f32) {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut min_val = _mm256_broadcast_ss(&input[0]);
    let mut max_val = _mm256_broadcast_ss(&input[0]);
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        min_val = _mm256_min_ps(min_val, chunk);
        max_val = _mm256_max_ps(max_val, chunk);
        i += 8;
    }

    // Extract and reduce both lanes
    let min128 = _mm_min_ps(
        _mm256_extractf128_ps(min_val, 0),
        _mm256_extractf128_ps(min_val, 1),
    );
    let min_temp = _mm_min_ps(min128, _mm_movehl_ps(min128, min128));
    let min_result = _mm_min_ps(min_temp, _mm_shuffle_ps(min_temp, min_temp, 0x01));
    let mut final_min = _mm_cvtss_f32(min_result);

    let max128 = _mm_max_ps(
        _mm256_extractf128_ps(max_val, 0),
        _mm256_extractf128_ps(max_val, 1),
    );
    let max_temp = _mm_max_ps(max128, _mm_movehl_ps(max128, max128));
    let max_result = _mm_max_ps(max_temp, _mm_shuffle_ps(max_temp, max_temp, 0x01));
    let mut final_max = _mm_cvtss_f32(max_result);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    (final_min, final_max)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_chunk = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_chunk = _mm256_loadu_ps(b.as_ptr().add(i));
        let product = _mm256_mul_ps(a_chunk, b_chunk);
        sum = _mm256_add_ps(sum, product);
        i += 8;
    }

    // Extract and sum both 128-bit lanes
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let temp = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < a.len() {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn norm_l1_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        let abs_chunk = _mm256_and_ps(chunk, abs_mask);
        sum = _mm256_add_ps(sum, abs_chunk);
        i += 8;
    }

    // Extract and sum both 128-bit lanes
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let temp = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < input.len() {
        final_sum += input[i].abs();
        i += 1;
    }

    final_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn norm_l2_squared_avx2(input: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= input.len() {
        let chunk = _mm256_loadu_ps(input.as_ptr().add(i));
        let squared = _mm256_mul_ps(chunk, chunk);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    // Extract and sum both 128-bit lanes
    let sum128 = _mm_add_ps(_mm256_extractf128_ps(sum, 0), _mm256_extractf128_ps(sum, 1));
    let temp = _mm_add_ps(sum128, _mm_movehl_ps(sum128, sum128));
    let result = _mm_add_ps(temp, _mm_shuffle_ps(temp, temp, 0x01));
    let mut final_sum = _mm_cvtss_f32(result);

    while i < input.len() {
        final_sum += input[i] * input[i];
        i += 1;
    }

    final_sum
}

// ============================================================================
// AVX512 implementations (x86/x86_64) - simplified for brevity
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn sum_vec_avx512(input: &[f32]) -> f32 {
    // For brevity, using AVX2 fallback - would implement proper AVX512 horizontal reductions
    sum_vec_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn product_vec_avx512(input: &[f32]) -> f32 {
    product_vec_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn min_vec_avx512(input: &[f32]) -> f32 {
    min_vec_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn max_vec_avx512(input: &[f32]) -> f32 {
    max_vec_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn min_max_vec_avx512(input: &[f32]) -> (f32, f32) {
    min_max_vec_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    dot_product_avx2(a, b)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn norm_l1_avx512(input: &[f32]) -> f32 {
    norm_l1_avx2(input)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn norm_l2_squared_avx512(input: &[f32]) -> f32 {
    norm_l2_squared_avx2(input)
}

// ============================================================================
// NEON implementations (ARM AArch64) - simplified for brevity
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn sum_vec_neon(input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        sum = vaddq_f32(sum, chunk);
        i += 4;
    }

    // Horizontal sum using pairwise addition
    let sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let sum1 = vpadd_f32(sum2, sum2);
    let mut final_sum = vget_lane_f32(sum1, 0);

    while i < input.len() {
        final_sum += input[i];
        i += 1;
    }

    final_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn product_vec_neon(input: &[f32]) -> f32 {
    // Fallback to scalar for simplicity
    product_vec_scalar(input)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn min_vec_neon(input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut min_val = vdupq_n_f32(input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        min_val = vminq_f32(min_val, chunk);
        i += 4;
    }

    // Horizontal min using pairwise min
    let min2 = vpmin_f32(vget_low_f32(min_val), vget_high_f32(min_val));
    let min1 = vpmin_f32(min2, min2);
    let mut final_min = vget_lane_f32(min1, 0);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        i += 1;
    }

    final_min
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn max_vec_neon(input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut max_val = vdupq_n_f32(input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        max_val = vmaxq_f32(max_val, chunk);
        i += 4;
    }

    // Horizontal max using pairwise max
    let max2 = vpmax_f32(vget_low_f32(max_val), vget_high_f32(max_val));
    let max1 = vpmax_f32(max2, max2);
    let mut final_max = vget_lane_f32(max1, 0);

    while i < input.len() {
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    final_max
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn min_max_vec_neon(input: &[f32]) -> (f32, f32) {
    use core::arch::aarch64::*;

    let mut min_val = vdupq_n_f32(input[0]);
    let mut max_val = vdupq_n_f32(input[0]);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        min_val = vminq_f32(min_val, chunk);
        max_val = vmaxq_f32(max_val, chunk);
        i += 4;
    }

    // Horizontal reductions
    let min2 = vpmin_f32(vget_low_f32(min_val), vget_high_f32(min_val));
    let min1 = vpmin_f32(min2, min2);
    let mut final_min = vget_lane_f32(min1, 0);

    let max2 = vpmax_f32(vget_low_f32(max_val), vget_high_f32(max_val));
    let max1 = vpmax_f32(max2, max2);
    let mut final_max = vget_lane_f32(max1, 0);

    while i < input.len() {
        if input[i] < final_min {
            final_min = input[i];
        }
        if input[i] > final_max {
            final_max = input[i];
        }
        i += 1;
    }

    (final_min, final_max)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_chunk = vld1q_f32(a.as_ptr().add(i));
        let b_chunk = vld1q_f32(b.as_ptr().add(i));
        let product = vmulq_f32(a_chunk, b_chunk);
        sum = vaddq_f32(sum, product);
        i += 4;
    }

    // Horizontal sum using pairwise addition
    let sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let sum1 = vpadd_f32(sum2, sum2);
    let mut final_sum = vget_lane_f32(sum1, 0);

    while i < a.len() {
        final_sum += a[i] * b[i];
        i += 1;
    }

    final_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn norm_l1_neon(input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        let abs_chunk = vabsq_f32(chunk);
        sum = vaddq_f32(sum, abs_chunk);
        i += 4;
    }

    // Horizontal sum using pairwise addition
    let sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let sum1 = vpadd_f32(sum2, sum2);
    let mut final_sum = vget_lane_f32(sum1, 0);

    while i < input.len() {
        final_sum += input[i].abs();
        i += 1;
    }

    final_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn norm_l2_squared_neon(input: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= input.len() {
        let chunk = vld1q_f32(input.as_ptr().add(i));
        let squared = vmulq_f32(chunk, chunk);
        sum = vaddq_f32(sum, squared);
        i += 4;
    }

    // Horizontal sum using pairwise addition
    let sum2 = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let sum1 = vpadd_f32(sum2, sum2);
    let mut final_sum = vget_lane_f32(sum1, 0);

    while i < input.len() {
        final_sum += input[i] * input[i];
        i += 1;
    }

    final_sum
}

#[allow(non_snake_case)]
#[cfg(all(test, not(feature = "no-std")))]
mod tests {
    use super::*;

    #[cfg(feature = "no-std")]
    use alloc::{vec, vec::Vec};

    const EPSILON: f32 = 1e-6;

    #[test]
    fn test_sum_vec() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = sum_vec(&input);
        assert_eq!(result, 10.0);

        // Test with empty vector
        let empty: Vec<f32> = vec![];
        assert_eq!(sum_vec(&empty), 0.0);

        // Test with negative numbers
        let negative = vec![-1.0, -2.0, 3.0, 4.0];
        assert_eq!(sum_vec(&negative), 4.0);
    }

    #[test]
    fn test_product_vec() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = product_vec(&input);
        assert_eq!(result, 24.0);

        // Test with empty vector
        let empty: Vec<f32> = vec![];
        assert_eq!(product_vec(&empty), 1.0);

        // Test with zeros
        let with_zero = vec![1.0, 2.0, 0.0, 4.0];
        assert_eq!(product_vec(&with_zero), 0.0);
    }

    #[test]
    fn test_min_vec() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = min_vec(&input);
        assert_eq!(result, 1.0);

        // Test with negative numbers
        let negative = vec![-1.0, -5.0, 2.0, -3.0];
        assert_eq!(min_vec(&negative), -5.0);
    }

    #[test]
    fn test_max_vec() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let result = max_vec(&input);
        assert_eq!(result, 5.0);

        // Test with negative numbers
        let negative = vec![-1.0, -5.0, 2.0, -3.0];
        assert_eq!(max_vec(&negative), 2.0);
    }

    #[test]
    fn test_min_max_vec() {
        let input = vec![3.0, 1.0, 4.0, 1.0, 5.0];
        let (min_val, max_val) = min_max_vec(&input);
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 5.0);

        // Test with single element
        let single = vec![42.0];
        let (min_single, max_single) = min_max_vec(&single);
        assert_eq!(min_single, 42.0);
        assert_eq!(max_single, 42.0);
    }

    #[test]
    fn test_mean_vec() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = mean_vec(&input);
        assert_eq!(result, 3.0);

        // Test with non-integer mean
        let decimal = vec![1.0, 2.0, 3.0];
        let decimal_mean = mean_vec(&decimal);
        assert_eq!(decimal_mean, 2.0);
    }

    #[test]
    fn test_dot_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = dot_product(&a, &b);
        assert_eq!(result, 32.0); // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        assert_eq!(dot_product(&empty_a, &empty_b), 0.0);

        // Test orthogonal vectors
        let ortho_a = vec![1.0, 0.0, 0.0];
        let ortho_b = vec![0.0, 1.0, 0.0];
        assert_eq!(dot_product(&ortho_a, &ortho_b), 0.0);
    }

    #[test]
    fn test_norm_l1() {
        let input = vec![-1.0, 2.0, -3.0, 4.0];
        let result = norm_l1(&input);
        assert_eq!(result, 10.0); // |−1| + |2| + |−3| + |4| = 1 + 2 + 3 + 4 = 10

        // Test with empty vector
        let empty: Vec<f32> = vec![];
        assert_eq!(norm_l1(&empty), 0.0);

        // Test with all positive
        let positive = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(norm_l1(&positive), 10.0);
    }

    #[test]
    fn test_norm_l2_squared() {
        let input = vec![1.0, 2.0, 3.0, 4.0];
        let result = norm_l2_squared(&input);
        assert_eq!(result, 30.0); // 1² + 2² + 3² + 4² = 1 + 4 + 9 + 16 = 30

        // Test with empty vector
        let empty: Vec<f32> = vec![];
        assert_eq!(norm_l2_squared(&empty), 0.0);
    }

    #[test]
    fn test_norm_l2() {
        let input = vec![3.0, 4.0];
        let result = norm_l2(&input);
        assert_eq!(result, 5.0); // sqrt(3² + 4²) = sqrt(9 + 16) = sqrt(25) = 5

        // Test with unit vector
        let unit = vec![1.0, 0.0, 0.0];
        assert!((norm_l2(&unit) - 1.0).abs() < EPSILON);
    }

    #[test]
    fn test_variance_vec() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = variance_vec(&input);
        assert!((result - 2.0).abs() < EPSILON); // Population variance = 2.0

        // Test with constant values (variance should be 0)
        let constant = vec![5.0, 5.0, 5.0, 5.0];
        assert!(variance_vec(&constant).abs() < EPSILON);
    }

    #[test]
    fn test_std_dev_vec() {
        let input = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = std_dev_vec(&input);
        assert!((result - 2.0_f32.sqrt()).abs() < EPSILON); // sqrt(2) ≈ 1.41421356

        // Test with constant values (std dev should be 0)
        let constant = vec![5.0, 5.0, 5.0, 5.0];
        assert!(std_dev_vec(&constant).abs() < EPSILON);
    }

    #[test]
    fn test_special_values() {
        // Test with infinity
        let with_inf = vec![1.0, f32::INFINITY, 3.0];
        assert_eq!(sum_vec(&with_inf), f32::INFINITY);
        assert_eq!(max_vec(&with_inf), f32::INFINITY);

        // Test with NaN
        let with_nan = vec![1.0, f32::NAN, 3.0];
        assert!(sum_vec(&with_nan).is_nan());
        assert!(min_vec(&with_nan).is_nan());
    }

    #[test]
    fn test_large_vectors() {
        let size = 10000;
        let input: Vec<f32> = (1..=size).map(|i| i as f32).collect();

        let expected_sum = (size * (size + 1) / 2) as f32;
        let actual_sum = sum_vec(&input);
        assert!((actual_sum - expected_sum).abs() < 1e-3);

        assert_eq!(min_vec(&input), 1.0);
        assert_eq!(max_vec(&input), size as f32);
    }

    #[test]
    #[should_panic(expected = "Input vector must not be empty")]
    fn test_min_vec_empty() {
        let empty: Vec<f32> = vec![];
        min_vec(&empty);
    }

    #[test]
    #[should_panic(expected = "Input vector must not be empty")]
    fn test_max_vec_empty() {
        let empty: Vec<f32> = vec![];
        max_vec(&empty);
    }

    #[test]
    #[should_panic(expected = "Input vectors must have the same length")]
    fn test_dot_product_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        dot_product(&a, &b);
    }

    #[test]
    fn test_mathematical_properties() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        // Test that dot product is commutative: a·b = b·a
        assert_eq!(dot_product(&a, &b), dot_product(&b, &a));

        // Test Cauchy-Schwarz inequality: |a·b| ≤ ||a|| ||b||
        let dot_ab = dot_product(&a, &b).abs();
        let norm_a = norm_l2(&a);
        let norm_b = norm_l2(&b);
        assert!(dot_ab <= norm_a * norm_b + EPSILON);

        // Test triangle inequality: ||a + b|| ≤ ||a|| + ||b||
        let sum_ab: Vec<f32> = a.iter().zip(b.iter()).map(|(&x, &y)| x + y).collect();
        let norm_sum = norm_l2(&sum_ab);
        assert!(norm_sum <= norm_a + norm_b + EPSILON);
    }
}
