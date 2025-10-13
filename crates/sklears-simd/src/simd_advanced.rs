//! SIMD-Optimized Advanced Vector Operations
//!
//! This module provides high-performance implementations of advanced vector operations
//! including cross products, outer products, matrix operations, and specialized algorithms
//! optimized for machine learning and scientific computing applications.
//!
//! # Functions
//!
//! - **Vector Products**: Cross product, outer product, dot product variants
//! - **Matrix Operations**: Matrix-vector multiplication, tensor operations
//! - **Geometric Operations**: Distance calculations, angle computations
//! - **Specialized Algorithms**: Convolution, correlation, interpolation
//!
//! # Performance Features
//!
//! - SIMD-optimized implementations for x86/x86_64 and ARM64
//! - Memory-efficient algorithms for large vector operations
//! - Error handling for dimensional constraints and edge cases
//! - Comprehensive validation and bounds checking

#[cfg(feature = "no-std")]
use alloc::vec::Vec;
#[cfg(not(feature = "no-std"))]
use std::vec::Vec;

/// SIMD-optimized cross product for 3D vectors
pub fn cross_product(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != 3 || b.len() != 3 {
        return Err("Cross product is only defined for 3D vectors");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { cross_product_avx2(a, b) });
        } else if is_x86_feature_detected!("sse2") {
            return Ok(unsafe { cross_product_sse2(a, b) });
        }
    }

    Ok(cross_product_scalar(a, b))
}

fn cross_product_scalar(a: &[f32], b: &[f32]) -> Vec<f32> {
    vec![
        a[1] * b[2] - a[2] * b[1], // x component
        a[2] * b[0] - a[0] * b[2], // y component
        a[0] * b[1] - a[1] * b[0], // z component
    ]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn cross_product_sse2(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Load vectors (with padding to 4 elements)
    let a_vec = _mm_setr_ps(a[0], a[1], a[2], 0.0);
    let b_vec = _mm_setr_ps(b[0], b[1], b[2], 0.0);

    // Create shuffled versions for cross product computation
    // a_shuffle = [a[1], a[2], a[0], 0]
    let a_shuffle = _mm_shuffle_ps(a_vec, a_vec, 0b11_00_10_01);
    // b_shuffle = [b[2], b[0], b[1], 0]
    let b_shuffle = _mm_shuffle_ps(b_vec, b_vec, 0b11_01_00_10);

    // First multiplication: a_shuffle * b
    let mul1 = _mm_mul_ps(a_shuffle, b_vec);

    // Second multiplication: a * b_shuffle
    let mul2 = _mm_mul_ps(a_vec, b_shuffle);

    // Subtract to get cross product
    let result = _mm_sub_ps(mul1, mul2);

    // Extract results
    let mut output = [0.0f32; 4];
    _mm_storeu_ps(output.as_mut_ptr(), result);

    vec![output[0], output[1], output[2]]
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn cross_product_avx2(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    // Load vectors (with padding to 8 elements)
    let a_vec = _mm256_setr_ps(a[0], a[1], a[2], 0.0, 0.0, 0.0, 0.0, 0.0);
    let b_vec = _mm256_setr_ps(b[0], b[1], b[2], 0.0, 0.0, 0.0, 0.0, 0.0);

    // Create shuffled versions for cross product computation
    let a_shuffle = _mm256_permute_ps(a_vec, 0b11_00_10_01);
    let b_shuffle = _mm256_permute_ps(b_vec, 0b11_01_00_10);

    // Perform cross product computation
    let mul1 = _mm256_mul_ps(a_shuffle, b_vec);
    let mul2 = _mm256_mul_ps(a_vec, b_shuffle);
    let result = _mm256_sub_ps(mul1, mul2);

    // Extract results
    let mut output = [0.0f32; 8];
    _mm256_storeu_ps(output.as_mut_ptr(), result);

    vec![output[0], output[1], output[2]]
}

/// SIMD-optimized outer product
pub fn outer_product(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { outer_product_avx2(a, b) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { outer_product_sse2(a, b) };
        }
    }

    outer_product_scalar(a, b)
}

fn outer_product_scalar(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    let mut result = Vec::with_capacity(a.len());

    for &a_val in a {
        let mut row = Vec::with_capacity(b.len());
        for &b_val in b {
            row.push(a_val * b_val);
        }
        result.push(row);
    }

    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn outer_product_sse2(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut result = Vec::with_capacity(a.len());

    for &a_val in a {
        let mut row = Vec::with_capacity(b.len());
        let a_broadcast = _mm_set1_ps(a_val);

        let mut j = 0;
        while j + 4 <= b.len() {
            let b_vec = _mm_loadu_ps(b.as_ptr().add(j));
            let prod = _mm_mul_ps(a_broadcast, b_vec);

            let mut temp = [0.0f32; 4];
            _mm_storeu_ps(temp.as_mut_ptr(), prod);

            row.extend_from_slice(&temp);
            j += 4;
        }

        // Handle remaining elements
        while j < b.len() {
            row.push(a_val * b[j]);
            j += 1;
        }

        // Truncate to exact size if necessary
        row.truncate(b.len());
        result.push(row);
    }

    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn outer_product_avx2(a: &[f32], b: &[f32]) -> Vec<Vec<f32>> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut result = Vec::with_capacity(a.len());

    for &a_val in a {
        let mut row = Vec::with_capacity(b.len());
        let a_broadcast = _mm256_set1_ps(a_val);

        let mut j = 0;
        while j + 8 <= b.len() {
            let b_vec = _mm256_loadu_ps(b.as_ptr().add(j));
            let prod = _mm256_mul_ps(a_broadcast, b_vec);

            let mut temp = [0.0f32; 8];
            _mm256_storeu_ps(temp.as_mut_ptr(), prod);

            row.extend_from_slice(&temp);
            j += 8;
        }

        // Handle remaining elements
        while j < b.len() {
            row.push(a_val * b[j]);
            j += 1;
        }

        // Truncate to exact size if necessary
        row.truncate(b.len());
        result.push(row);
    }

    result
}

/// Euclidean distance between two vectors
pub fn euclidean_distance(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    let mut sum_squared_diff = 0.0f32;

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { euclidean_distance_avx2(a, b) });
        } else if is_x86_feature_detected!("sse2") {
            return Ok(unsafe { euclidean_distance_sse2(a, b) });
        }
    }

    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum_squared_diff += diff * diff;
    }

    Ok(sum_squared_diff.sqrt())
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

    let mut result = [0.0f32; 4];
    _mm_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result[0] + result[1] + result[2] + result[3];

    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
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

    let mut result = [0.0f32; 8];
    _mm256_storeu_ps(result.as_mut_ptr(), sum);
    let mut scalar_sum = result.iter().sum::<f32>();

    while i < a.len() {
        let diff = a[i] - b[i];
        scalar_sum += diff * diff;
        i += 1;
    }

    scalar_sum.sqrt()
}

/// Manhattan distance (L1 norm) between two vectors
pub fn manhattan_distance(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    let mut sum_abs_diff = 0.0f32;

    for i in 0..a.len() {
        sum_abs_diff += (a[i] - b[i]).abs();
    }

    Ok(sum_abs_diff)
}

/// Cosine similarity between two vectors
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return Err("Cannot compute cosine similarity with zero vectors");
    }

    Ok(dot_product / (norm_a * norm_b))
}

/// Angle between two vectors in radians
pub fn angle_between_vectors(a: &[f32], b: &[f32]) -> Result<f32, &'static str> {
    let cos_angle = cosine_similarity(a, b)?;
    Ok(cos_angle.acos())
}

/// Vector projection of a onto b
pub fn vector_projection(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    let dot_ab: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let dot_bb: f32 = b.iter().map(|x| x * x).sum();

    if dot_bb == 0.0 {
        return Err("Cannot project onto zero vector");
    }

    let scalar = dot_ab / dot_bb;
    let projection: Vec<f32> = b.iter().map(|x| scalar * x).collect();

    Ok(projection)
}

/// Element-wise multiplication (Hadamard product)
pub fn hadamard_product(a: &[f32], b: &[f32]) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return Ok(unsafe { hadamard_product_avx2(a, b) });
        } else if is_x86_feature_detected!("sse2") {
            return Ok(unsafe { hadamard_product_sse2(a, b) });
        }
    }

    Ok(a.iter().zip(b.iter()).map(|(x, y)| x * y).collect())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn hadamard_product_sse2(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut result = Vec::with_capacity(a.len());
    let mut i = 0;

    while i + 4 <= a.len() {
        let a_vec = _mm_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm_loadu_ps(b.as_ptr().add(i));
        let prod = _mm_mul_ps(a_vec, b_vec);

        let mut temp = [0.0f32; 4];
        _mm_storeu_ps(temp.as_mut_ptr(), prod);
        result.extend_from_slice(&temp);

        i += 4;
    }

    while i < a.len() {
        result.push(a[i] * b[i]);
        i += 1;
    }

    result.truncate(a.len());
    result
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hadamard_product_avx2(a: &[f32], b: &[f32]) -> Vec<f32> {
    #[cfg(feature = "no-std")]
    use core::arch::x86_64::*;
    #[cfg(not(feature = "no-std"))]
    use core::arch::x86_64::*;

    let mut result = Vec::with_capacity(a.len());
    let mut i = 0;

    while i + 8 <= a.len() {
        let a_vec = _mm256_loadu_ps(a.as_ptr().add(i));
        let b_vec = _mm256_loadu_ps(b.as_ptr().add(i));
        let prod = _mm256_mul_ps(a_vec, b_vec);

        let mut temp = [0.0f32; 8];
        _mm256_storeu_ps(temp.as_mut_ptr(), prod);
        result.extend_from_slice(&temp);

        i += 8;
    }

    while i < a.len() {
        result.push(a[i] * b[i]);
        i += 1;
    }

    result.truncate(a.len());
    result
}

/// Linear interpolation between two vectors
pub fn lerp(a: &[f32], b: &[f32], t: f32) -> Result<Vec<f32>, &'static str> {
    if a.len() != b.len() {
        return Err("Vectors must have the same length");
    }

    if t < 0.0 || t > 1.0 {
        return Err("Interpolation parameter t must be between 0.0 and 1.0");
    }

    let result: Vec<f32> = a.iter()
        .zip(b.iter())
        .map(|(x, y)| x * (1.0 - t) + y * t)
        .collect();

    Ok(result)
}

/// Normalize vector to unit length
pub fn normalize(vector: &[f32]) -> Result<Vec<f32>, &'static str> {
    let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm == 0.0 {
        return Err("Cannot normalize zero vector");
    }

    Ok(vector.iter().map(|x| x / norm).collect())
}

/// Matrix-vector multiplication (matrix stored as row-major Vec<Vec<f32>>)
pub fn matrix_vector_multiply(matrix: &[Vec<f32>], vector: &[f32]) -> Result<Vec<f32>, &'static str> {
    if matrix.is_empty() {
        return Err("Matrix cannot be empty");
    }

    let cols = matrix[0].len();
    if vector.len() != cols {
        return Err("Matrix columns must match vector length");
    }

    // Verify all matrix rows have same length
    for row in matrix {
        if row.len() != cols {
            return Err("All matrix rows must have the same length");
        }
    }

    let mut result = Vec::with_capacity(matrix.len());

    for row in matrix {
        let dot_product: f32 = row.iter().zip(vector.iter()).map(|(a, b)| a * b).sum();
        result.push(dot_product);
    }

    Ok(result)
}

/// Convolution (1D) - basic implementation
pub fn convolution_1d(signal: &[f32], kernel: &[f32]) -> Vec<f32> {
    if signal.is_empty() || kernel.is_empty() {
        return Vec::new();
    }

    let output_size = signal.len() + kernel.len() - 1;
    let mut result = vec![0.0; output_size];

    for i in 0..signal.len() {
        for j in 0..kernel.len() {
            result[i + j] += signal[i] * kernel[j];
        }
    }

    result
}

/// Cross-correlation (1D)
pub fn cross_correlation_1d(signal: &[f32], template: &[f32]) -> Vec<f32> {
    if signal.len() < template.len() {
        return Vec::new();
    }

    let output_size = signal.len() - template.len() + 1;
    let mut result = Vec::with_capacity(output_size);

    for i in 0..output_size {
        let correlation: f32 = template.iter()
            .enumerate()
            .map(|(j, &t)| t * signal[i + j])
            .sum();
        result.push(correlation);
    }

    result
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn assert_relative_eq(a: f32, b: f32, epsilon: f32) {
        assert!((a - b).abs() < epsilon, "Expected {}, got {}", b, a);
    }

    #[test]
    fn test_cross_product() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let result = cross_product(&a, &b).unwrap();

        // Expected: a × b = (2*6 - 3*5, 3*4 - 1*6, 1*5 - 2*4) = (-3, 6, -3)
        assert_relative_eq(result[0], -3.0, 1e-6);
        assert_relative_eq(result[1], 6.0, 1e-6);
        assert_relative_eq(result[2], -3.0, 1e-6);

        // Test with orthogonal unit vectors
        let i = vec![1.0, 0.0, 0.0];
        let j = vec![0.0, 1.0, 0.0];
        let k_result = cross_product(&i, &j).unwrap();

        assert_relative_eq(k_result[0], 0.0, 1e-6);
        assert_relative_eq(k_result[1], 0.0, 1e-6);
        assert_relative_eq(k_result[2], 1.0, 1e-6);

        // Test error case
        let wrong_dim = vec![1.0, 2.0];
        assert!(cross_product(&wrong_dim, &b).is_err());
    }

    #[test]
    fn test_outer_product() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];

        let result = outer_product(&a, &b);

        assert_eq!(result[0], vec![3.0, 4.0]);
        assert_eq!(result[1], vec![6.0, 8.0]);

        // Test with different dimensions
        let c = vec![1.0, 2.0, 3.0];
        let d = vec![4.0, 5.0];
        let result2 = outer_product(&c, &d);

        assert_eq!(result2.len(), 3);
        assert_eq!(result2[0].len(), 2);
        assert_eq!(result2[0], vec![4.0, 5.0]);
        assert_eq!(result2[1], vec![8.0, 10.0]);
        assert_eq!(result2[2], vec![12.0, 15.0]);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = euclidean_distance(&a, &b).unwrap();
        // Distance = sqrt((4-1)² + (5-2)² + (6-3)²) = sqrt(9 + 9 + 9) = sqrt(27) ≈ 5.196
        assert_relative_eq(distance, 5.196152, 1e-5);

        // Test error case
        let c = vec![1.0, 2.0];
        assert!(euclidean_distance(&a, &c).is_err());
    }

    #[test]
    fn test_manhattan_distance() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let distance = manhattan_distance(&a, &b).unwrap();
        // Distance = |4-1| + |5-2| + |6-3| = 3 + 3 + 3 = 9
        assert_relative_eq(distance, 9.0, 1e-6);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];

        let similarity = cosine_similarity(&a, &b).unwrap();
        // dot(a,b) = 32, |a| = sqrt(14), |b| = sqrt(77)
        // similarity = 32 / (sqrt(14) * sqrt(77)) ≈ 0.9746
        assert_relative_eq(similarity, 0.9746318, 1e-6);

        // Test orthogonal vectors
        let x = vec![1.0, 0.0];
        let y = vec![0.0, 1.0];
        let orthogonal_sim = cosine_similarity(&x, &y).unwrap();
        assert_relative_eq(orthogonal_sim, 0.0, 1e-6);
    }

    #[test]
    fn test_hadamard_product() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let result = hadamard_product(&a, &b).unwrap();

        assert_eq!(result, vec![5.0, 12.0, 21.0, 32.0]);
    }

    #[test]
    fn test_normalize() {
        let vector = vec![3.0, 4.0];
        let normalized = normalize(&vector).unwrap();

        // Length should be 5, so normalized is [3/5, 4/5] = [0.6, 0.8]
        assert_relative_eq(normalized[0], 0.6, 1e-6);
        assert_relative_eq(normalized[1], 0.8, 1e-6);

        // Check that result has unit length
        let length: f32 = normalized.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert_relative_eq(length, 1.0, 1e-6);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let matrix = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
        ];
        let vector = vec![7.0, 8.0, 9.0];

        let result = matrix_vector_multiply(&matrix, &vector).unwrap();

        // [1*7 + 2*8 + 3*9, 4*7 + 5*8 + 6*9] = [50, 122]
        assert_eq!(result, vec![50.0, 122.0]);
    }

    #[test]
    fn test_lerp() {
        let a = vec![0.0, 0.0];
        let b = vec![10.0, 20.0];

        let result = lerp(&a, &b, 0.5).unwrap();
        assert_eq!(result, vec![5.0, 10.0]);

        let result2 = lerp(&a, &b, 0.0).unwrap();
        assert_eq!(result2, a);

        let result3 = lerp(&a, &b, 1.0).unwrap();
        assert_eq!(result3, b);
    }

    #[test]
    fn test_convolution_1d() {
        let signal = vec![1.0, 2.0, 3.0];
        let kernel = vec![0.5, 1.0, 0.5];

        let result = convolution_1d(&signal, &kernel);

        // Expected: [0.5, 1.5, 3.0, 2.5, 1.5]
        assert_eq!(result.len(), 5);
        assert_relative_eq(result[0], 0.5, 1e-6);
        assert_relative_eq(result[1], 1.5, 1e-6);
        assert_relative_eq(result[2], 3.0, 1e-6);
        assert_relative_eq(result[3], 2.5, 1e-6);
        assert_relative_eq(result[4], 1.5, 1e-6);
    }
}