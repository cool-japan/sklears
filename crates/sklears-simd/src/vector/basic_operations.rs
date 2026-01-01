//! # Basic SIMD Vector Operations
//!
//! Fundamental vector operations optimized with SIMD instructions including
//! dot products, norms, distance calculations, and basic linear algebra operations.
//!
//! ## Features
//!
//! - **Dot Product**: SIMD-optimized dot product with platform-specific implementations
//! - **Vector Norms**: L1, L2, and infinity norms with high performance
//! - **Distance Metrics**: Euclidean, Manhattan, and cosine distance/similarity
//! - **Advanced Operations**: Cross product and outer product for linear algebra
//! - **Multi-Platform**: SSE2, AVX2, AVX512, and NEON optimizations
//! - **Scalar Fallbacks**: Automatic fallback to scalar operations when needed
//!
//! ## Implementation Details
//!
//! All functions automatically detect the best available SIMD instruction set
//! and provide graceful fallback to scalar implementations. The functions are
//! designed to handle arbitrary vector lengths efficiently by processing
//! SIMD-sized chunks and handling remainders appropriately.

#[cfg(feature = "no-std")]
use alloc::vec;
#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

// Import ARM64 feature detection macro
#[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
use std::arch::is_aarch64_feature_detected;

/// SIMD-optimized dot product computation
///
/// Computes the dot product of two vectors using the best available SIMD
/// instruction set. Automatically falls back to scalar implementation if
/// no SIMD support is available.
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
/// use sklears_simd::vector::basic_operations::dot_product;
///
/// let a = vec![1.0, 2.0, 3.0, 4.0];
/// let b = vec![5.0, 6.0, 7.0, 8.0];
/// let result = dot_product(&a, &b);
/// assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 70
/// ```
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

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

/// SIMD-optimized L2 norm (Euclidean norm) computation
///
/// Computes the L2 norm (||x||₂) of a vector using SIMD-optimized dot product.
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// The L2 norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::norm_l2;
///
/// let x = vec![3.0, 4.0];
/// let result = norm_l2(&x);
/// assert_eq!(result, 5.0); // sqrt(3² + 4²) = 5
/// ```
pub fn norm_l2(x: &[f32]) -> f32 {
    dot_product(x, x).sqrt()
}

/// SIMD-optimized L1 norm (Manhattan norm) computation
///
/// Computes the L1 norm (||x||₁) of a vector using SIMD instructions
/// for absolute value computation and summation.
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// The L1 norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::norm_l1;
///
/// let x = vec![-3.0, 4.0, -5.0];
/// let result = norm_l1(&x);
/// assert_eq!(result, 12.0); // |−3| + |4| + |−5| = 12
/// ```
pub fn norm_l1(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { norm_l1_avx2(x) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { norm_l1_sse2(x) };
        }
    }

    #[cfg(all(target_arch = "aarch64", not(feature = "no-std")))]
    {
        if is_aarch64_feature_detected!("neon") {
            return unsafe { norm_l1_neon(x) };
        }
    }

    norm_l1_scalar(x)
}

/// SIMD-optimized infinity norm computation
///
/// Computes the L∞ norm (||x||∞) of a vector, which is the maximum
/// absolute value of all elements.
///
/// # Arguments
/// * `x` - Input vector
///
/// # Returns
/// The infinity norm as a single f32 value
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::norm_inf;
///
/// let x = vec![-3.0, 4.0, -5.0, 2.0];
/// let result = norm_inf(&x);
/// assert_eq!(result, 5.0); // max(|−3|, |4|, |−5|, |2|) = 5
/// ```
pub fn norm_inf(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { norm_inf_avx2(x) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { norm_inf_sse2(x) };
        }
    }

    norm_inf_scalar(x)
}

/// SIMD-optimized Euclidean distance computation
///
/// Computes the Euclidean distance between two vectors:
/// ||a - b||₂ = sqrt(Σ(aᵢ - bᵢ)²)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
/// The Euclidean distance as a single f32 value
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::euclidean_distance;
///
/// let a = vec![1.0, 2.0, 3.0];
/// let b = vec![4.0, 5.0, 6.0];
/// let result = euclidean_distance(&a, &b);
/// // sqrt((1-4)² + (2-5)² + (3-6)²) = sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
/// assert!((result - 5.196).abs() < 0.01);
/// ```
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if a.is_empty() {
        return 0.0;
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { euclidean_distance_avx2(a, b) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { euclidean_distance_sse2(a, b) };
        }
    }

    euclidean_distance_scalar(a, b)
}

/// SIMD-optimized cosine similarity computation
///
/// Computes the cosine similarity between two vectors:
/// cos_sim(a, b) = (a · b) / (||a||₂ * ||b||₂)
///
/// # Arguments
/// * `a` - First vector
/// * `b` - Second vector (must have same length as `a`)
///
/// # Returns
/// The cosine similarity as a single f32 value between -1.0 and 1.0
///
/// # Panics
/// Panics if the vectors have different lengths
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::cosine_similarity;
///
/// let a = vec![1.0, 0.0, 0.0];
/// let b = vec![0.0, 1.0, 0.0];
/// let result = cosine_similarity(&a, &b);
/// assert_eq!(result, 0.0); // Orthogonal vectors have cosine similarity of 0
/// ```
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vectors must have the same length");

    if a.is_empty() {
        return 1.0; // Convention: empty vectors are perfectly similar
    }

    let dot_ab = dot_product(a, b);
    let norm_a = norm_l2(a);
    let norm_b = norm_l2(b);

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0; // Zero vectors are orthogonal to everything
    }

    dot_ab / (norm_a * norm_b)
}

/// SIMD-optimized cross product for 3D vectors
///
/// Computes the cross product a × b for two 3-dimensional vectors.
///
/// # Arguments
/// * `a` - First 3D vector
/// * `b` - Second 3D vector
///
/// # Returns
/// The cross product as a 3-element vector, or an error if input vectors
/// are not exactly 3 elements long
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::cross_product;
///
/// let a = vec![1.0, 0.0, 0.0];  // Unit vector along x-axis
/// let b = vec![0.0, 1.0, 0.0];  // Unit vector along y-axis
/// let result = cross_product(&a, &b).unwrap();
/// assert_eq!(result, vec![0.0, 0.0, 1.0]);  // Unit vector along z-axis
/// ```
pub fn cross_product(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != 3 || b.len() != 3 {
        return Err("Cross product requires exactly 3-dimensional vectors");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("sse2") {
            return Ok(unsafe { cross_product_sse2(a, b) });
        }
    }

    Ok(cross_product_scalar(a, b))
}

/// SIMD-optimized outer product computation
///
/// Computes the outer product of two vectors, resulting in a matrix
/// where element (i,j) = a\[i\] * b\[j\].
///
/// # Arguments
/// * `a` - First vector (m elements)
/// * `b` - Second vector (n elements)
///
/// # Returns
/// An m×n matrix represented as `Vec<Vec<f32>>`
///
/// # Examples
/// ```rust
/// use sklears_simd::vector::basic_operations::outer_product;
///
/// let a = vec![1.0, 2.0];
/// let b = vec![3.0, 4.0, 5.0];
/// let result = outer_product(&a, &b);
/// // Expected: [[3.0, 4.0, 5.0], [6.0, 8.0, 10.0]]
/// assert_eq!(result[0], vec![3.0, 4.0, 5.0]);
/// assert_eq!(result[1], vec![6.0, 8.0, 10.0]);
/// ```
pub fn outer_product(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = b.len();

    if m == 0 || n == 0 {
        return vec![];
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if crate::simd_feature_detected!("avx2") {
            return unsafe { outer_product_avx2(a, b) };
        } else if crate::simd_feature_detected!("sse2") {
            return unsafe { outer_product_sse2(a, b) };
        }
    }

    outer_product_scalar(a, b)
}

// ============================================================================
// Scalar implementations (fallbacks)
// ============================================================================

fn dot_product_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

fn norm_l1_scalar(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).sum()
}

fn norm_inf_scalar(x: &[f32]) -> f32 {
    x.iter().map(|&v| v.abs()).fold(0.0f32, |a, b| a.max(b))
}

fn euclidean_distance_scalar(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn cross_product_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    vec![
        a[1] * b[2] - a[2] * b[1], // i component
        a[2] * b[0] - a[0] * b[2], // j component
        a[0] * b[1] - a[1] * b[0], // k component
    ]
}

fn outer_product_scalar(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let m = a.len();
    let n = b.len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        for (j, &b_val) in b.iter().enumerate().take(n) {
            result[i][j] = a[i] * b_val;
        }
    }

    result
}

// ============================================================================
// SSE2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn dot_product_sse2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let prod = _mm_mul_ps(a_vec, b_vec);
        sum = _mm_add_ps(sum, prod);
        i += 4;
    }

    // Horizontal sum of the 4 elements in sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < a.len() {
        scalar_sum += a[i] * b[i];
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn norm_l1_sse2(x: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Create mask for absolute value (clear sign bit)
    let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= x.len() {
        let x_vec = _mm_loadu_ps(x.as_ptr().add(i));
        let abs_vec = _mm_and_ps(x_vec, abs_mask);
        sum = _mm_add_ps(sum, abs_vec);
        i += 4;
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < x.len() {
        scalar_sum += x[i].abs();
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn norm_inf_sse2(x: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let abs_mask = _mm_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut max_vec = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= x.len() {
        let x_vec = _mm_loadu_ps(x.as_ptr().add(i));
        let abs_vec = _mm_and_ps(x_vec, abs_mask);
        max_vec = _mm_max_ps(max_vec, abs_vec);
        i += 4;
    }

    // Find maximum of the 4 elements
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result[0].max(result[1]).max(result[2]).max(result[3]);

    // Handle remaining elements
    while i < x.len() {
        max_val = max_val.max(x[i].abs());
        i += 1;
    }

    max_val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn euclidean_distance_sse2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm_setzero_ps();
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let diff = _mm_sub_ps(a_vec, b_vec);
        let squared = _mm_mul_ps(diff, diff);
        sum = _mm_add_ps(sum, squared);
        i += 4;
    }

    // Horizontal sum
    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    // Handle remaining elements
    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cross_product_sse2(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Load vectors: a = [a0, a1, a2, 0], b = [b0, b1, b2, 0]
    let a_vec = _mm_set_ps(0.0, a[2], a[1], a[0]);
    let b_vec = _mm_set_ps(0.0, b[2], b[1], b[0]);

    // Create shuffled versions for cross product computation
    // a_yzx = [a1, a2, a0, 0]
    let a_yzx = _mm_shuffle_ps(a_vec, a_vec, 0b00_01_10_01);
    // b_zxy = [b2, b0, b1, 0]
    let b_zxy = _mm_shuffle_ps(b_vec, b_vec, 0b00_10_00_10);

    // a_zxy = [a2, a0, a1, 0]
    let a_zxy = _mm_shuffle_ps(a_vec, a_vec, 0b00_10_00_10);
    // b_yzx = [b1, b2, b0, 0]
    let b_yzx = _mm_shuffle_ps(b_vec, b_vec, 0b00_01_10_01);

    // Compute cross product: a_yzx * b_zxy - a_zxy * b_yzx
    let prod1 = _mm_mul_ps(a_yzx, b_zxy);
    let prod2 = _mm_mul_ps(a_zxy, b_yzx);
    let result_vec = _mm_sub_ps(prod1, prod2);

    // Extract result
    let mut output = [0.0f32; 4];
    _mm_storeu_ps(output.as_mut_ptr(), result_vec);

    vec![output[0], output[1], output[2]]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn outer_product_sse2(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let m = a.len();
    let n = b.len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        let a_broadcast = _mm_set1_ps(a[i]);
        let mut j = 0;

        while j + 4 <= n {
            let b_vec = _mm_loadu_ps(b.as_ptr().add(j));
            let prod = _mm_mul_ps(a_broadcast, b_vec);
            _mm_storeu_ps(result[i].as_mut_ptr().add(j), prod);
            j += 4;
        }

        // Handle remaining elements
        while j < n {
            result[i][j] = a[i] * b[j];
            j += 1;
        }
    }

    result
}

// ============================================================================
// AVX2 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dot_product_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    // Process 8 elements at a time
    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);
        sum = _mm256_add_ps(sum, prod);
        i += 8;
    }

    // Horizontal sum of the 8 elements in sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < a.len() {
        scalar_sum += a[i] * b[i];
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn norm_l1_avx2(x: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= x.len() {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let abs_vec = _mm256_and_ps(x_vec, abs_mask);
        sum = _mm256_add_ps(sum, abs_vec);
        i += 8;
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < x.len() {
        scalar_sum += x[i].abs();
        i += 1;
    }

    scalar_sum
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn norm_inf_avx2(x: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let abs_mask = _mm256_set1_ps(f32::from_bits(0x7FFFFFFF));
    let mut max_vec = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= x.len() {
        let x_vec = _mm256_loadu_ps(x.as_ptr().add(i));
        let abs_vec = _mm256_and_ps(x_vec, abs_mask);
        max_vec = _mm256_max_ps(max_vec, abs_vec);
        i += 8;
    }

    // Find maximum of the 8 elements
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), max_vec);
    let mut max_val = result.iter().fold(0.0f32, |a, &b| a.max(b));

    // Handle remaining elements
    while i < x.len() {
        max_val = max_val.max(x[i].abs());
        i += 1;
    }

    max_val
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn euclidean_distance_avx2(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm256_setzero_ps();
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let diff = _mm256_sub_ps(a_vec, b_vec);
        let squared = _mm256_mul_ps(diff, diff);
        sum = _mm256_add_ps(sum, squared);
        i += 8;
    }

    // Horizontal sum
    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    // Handle remaining elements
    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn outer_product_avx2(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let m = a.len();
    let n = b.len();
    let mut result = vec![vec![0.0; n]; m];

    for i in 0..m {
        let a_broadcast = _mm256_set1_ps(a[i]);
        let mut j = 0;

        while j + 8 <= n {
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(j));
            let prod = _mm256_mul_ps(a_broadcast, b_vec);
            _mm256_storeu_ps(result[i].as_mut_ptr().add(j), prod);
            j += 8;
        }

        // Handle remaining elements
        while j < n {
            result[i][j] = a[i] * b[j];
            j += 1;
        }
    }

    result
}

// ============================================================================
// AVX512 implementations (x86/x86_64)
// ============================================================================

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx512f")]
unsafe fn dot_product_avx512(a: &[f32], b: &[f32]) -> f32 {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut sum = _mm512_setzero_ps();
    let mut i = 0;

    // Process 16 elements at a time
    while i + 16 <= a.len() {
        let a_vec = _mm512_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm512_loadu_ps(b.as_ptr().add(i));
        sum = _mm512_fmadd_ps(a_vec, b_vec, sum); // Fused multiply-add
        i += 16;
    }

    // Horizontal sum of the 16 elements in sum
    let scalar_sum = _mm512_reduce_add_ps(sum);

    // Handle remaining elements
    let mut remaining_sum = 0.0f32;
    while i < a.len() {
        remaining_sum += a[i] * b[i];
        i += 1;
    }

    scalar_sum + remaining_sum
}

// ============================================================================
// NEON implementations (ARM AArch64)
// ============================================================================

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn dot_product_neon(a: &[f32], b: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    // Process 4 elements at a time
    while i + 4 <= a.len() {
        let a_vec = vld1q_f32(a.as_ptr().add(i));
        let b_vec = vld1q_f32(b.as_ptr().add(i));
        sum = vfmaq_f32(sum, a_vec, b_vec); // Fused multiply-add
        i += 4;
    }

    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let final_sum = vpadd_f32(sum_pair, sum_pair);
    let mut scalar_sum = vget_lane_f32(final_sum, 0);

    // Handle remaining elements
    while i < a.len() {
        scalar_sum += a[i] * b[i];
        i += 1;
    }

    scalar_sum
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn norm_l1_neon(x: &[f32]) -> f32 {
    use core::arch::aarch64::*;

    let mut sum = vdupq_n_f32(0.0);
    let mut i = 0;

    while i + 4 <= x.len() {
        let x_vec = vld1q_f32(x.as_ptr().add(i));
        let abs_vec = vabsq_f32(x_vec);
        sum = vaddq_f32(sum, abs_vec);
        i += 4;
    }

    // Horizontal sum
    let sum_pair = vpadd_f32(vget_low_f32(sum), vget_high_f32(sum));
    let final_sum = vpadd_f32(sum_pair, sum_pair);
    let mut scalar_sum = vget_lane_f32(final_sum, 0);

    // Handle remaining elements
    while i < x.len() {
        scalar_sum += x[i].abs();
        i += 1;
    }

    scalar_sum
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
        assert_eq!(result, 70.0); // 1*5 + 2*6 + 3*7 + 4*8 = 5 + 12 + 21 + 32 = 70

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        assert_eq!(dot_product(&empty_a, &empty_b), 0.0);

        // Test with single element
        let single_a = vec![3.0];
        let single_b = vec![4.0];
        assert_eq!(dot_product(&single_a, &single_b), 12.0);
    }

    #[test]
    fn test_norms() {
        let x = vec![3.0, 4.0];

        // L2 norm
        let norm2 = norm_l2(&x);
        assert_eq!(norm2, 5.0); // sqrt(3² + 4²) = sqrt(25) = 5

        // L1 norm
        let norm1 = norm_l1(&x);
        assert_eq!(norm1, 7.0); // |3| + |4| = 7

        // L∞ norm
        let norm_inf_val = norm_inf(&x);
        assert_eq!(norm_inf_val, 4.0); // max(|3|, |4|) = 4

        // Test with negative values
        let y = vec![-3.0, 4.0, -5.0];
        assert_eq!(norm_l1(&y), 12.0); // |-3| + |4| + |-5| = 12
        assert_eq!(norm_inf(&y), 5.0); // max(|-3|, |4|, |-5|) = 5

        // Test with empty vector
        let empty: Vec<f32> = vec![];
        assert_eq!(norm_l2(&empty), 0.0);
        assert_eq!(norm_l1(&empty), 0.0);
        assert_eq!(norm_inf(&empty), 0.0);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let result = euclidean_distance(&a, &b);
        // sqrt((1-4)² + (2-5)² + (3-6)²) = sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
        assert!((result - 5.196).abs() < 0.01);

        // Test with identical vectors
        let identical = euclidean_distance(&a, &a);
        assert_eq!(identical, 0.0);

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        assert_eq!(euclidean_distance(&empty_a, &empty_b), 0.0);
    }

    #[test]
    fn test_cosine_similarity() {
        // Test orthogonal vectors
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        let result = cosine_similarity(&a, &b);
        assert!((result - 0.0).abs() < f32::EPSILON);

        // Test identical vectors
        let identical = cosine_similarity(&a, &a);
        assert!((identical - 1.0).abs() < f32::EPSILON);

        // Test opposite vectors
        let opposite = vec![-1.0, 0.0, 0.0];
        let opposite_sim = cosine_similarity(&a, &opposite);
        assert!((opposite_sim - (-1.0)).abs() < f32::EPSILON);

        // Test with zero vector
        let zero = vec![0.0, 0.0, 0.0];
        let zero_sim = cosine_similarity(&a, &zero);
        assert_eq!(zero_sim, 0.0);

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_b: Vec<f32> = vec![];
        assert_eq!(cosine_similarity(&empty_a, &empty_b), 1.0);
    }

    #[test]
    fn test_cross_product() {
        // Test unit vectors
        let i = vec![1.0, 0.0, 0.0];
        let j = vec![0.0, 1.0, 0.0];
        let result = cross_product(&i, &j).unwrap();
        assert_eq!(result, vec![0.0, 0.0, 1.0]); // i × j = k

        // Test with general vectors
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let cross = cross_product(&a, &b).unwrap();
        // Expected: (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        assert_eq!(cross, vec![-3.0, 6.0, -3.0]);

        // Test error for wrong dimensions
        let wrong_dim = vec![1.0, 2.0];
        assert!(cross_product(&wrong_dim, &j).is_err());
    }

    #[test]
    fn test_outer_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let result = outer_product(&a, &b);

        // Expected: [[1*3, 1*4, 1*5], [2*3, 2*4, 2*5]] = [[3, 4, 5], [6, 8, 10]]
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 3);
        assert_eq!(result[0], vec![3.0, 4.0, 5.0]);
        assert_eq!(result[1], vec![6.0, 8.0, 10.0]);

        // Test with empty vectors
        let empty_a: Vec<f32> = vec![];
        let empty_result = outer_product(&empty_a, &b);
        assert!(empty_result.is_empty());

        let empty_b: Vec<f32> = vec![];
        let empty_result2 = outer_product(&a, &empty_b);
        assert!(empty_result2.is_empty());
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_dot_product_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        dot_product(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_euclidean_distance_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        euclidean_distance(&a, &b);
    }

    #[test]
    #[should_panic(expected = "Vectors must have the same length")]
    fn test_cosine_similarity_dimension_mismatch() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0];
        cosine_similarity(&a, &b);
    }
}
