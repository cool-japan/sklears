//! Distance and similarity metrics

use crate::simd::SimdDistanceOps;
use scirs2_core::ndarray::Array1;
use sklears_core::types::Float;
use std::any::TypeId;

/// Compute Euclidean distance between two points
pub fn euclidean_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Use SIMD optimization for f32 if available
    euclidean_distance_optimized(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Internal optimized Euclidean distance implementation
#[inline]
fn euclidean_distance_optimized(a: &[Float], b: &[Float]) -> Float {
    // Use SIMD optimization for f32 if available
    if TypeId::of::<Float>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(a) };
        let b_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(b) };
        let result_f32 = SimdDistanceOps::euclidean_distance_f32(a_f32, b_f32);
        return result_f32 as Float;
    } else if TypeId::of::<Float>() == TypeId::of::<f64>() {
        // Convert to f32 for SIMD, then back to f64
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        let result_f32 = SimdDistanceOps::euclidean_distance_f32(&a_f32, &b_f32);
        return (result_f32 as f64) as Float;
    }

    // Fallback to scalar implementation for other types
    euclidean_distance_scalar(a, b)
}

/// Scalar fallback for Euclidean distance
#[inline]
fn euclidean_distance_scalar(a: &[Float], b: &[Float]) -> Float {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Compute Manhattan (L1) distance between two points
pub fn manhattan_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Use SIMD optimization for f32 if available
    manhattan_distance_optimized(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Internal optimized Manhattan distance implementation
#[inline]
fn manhattan_distance_optimized(a: &[Float], b: &[Float]) -> Float {
    // Use SIMD optimization for f32 if available
    if TypeId::of::<Float>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(a) };
        let b_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(b) };
        let result_f32 = SimdDistanceOps::manhattan_distance_f32(a_f32, b_f32);
        return result_f32 as Float;
    } else if TypeId::of::<Float>() == TypeId::of::<f64>() {
        // Convert to f32 for SIMD, then back to f64
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        let result_f32 = SimdDistanceOps::manhattan_distance_f32(&a_f32, &b_f32);
        return (result_f32 as f64) as Float;
    }

    // Fallback to scalar implementation for other types
    manhattan_distance_scalar(a, b)
}

/// Scalar fallback for Manhattan distance
#[inline]
fn manhattan_distance_scalar(a: &[Float], b: &[Float]) -> Float {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

/// Compute Minkowski distance between two points
pub fn minkowski_distance(a: &Array1<Float>, b: &Array1<Float>, p: Float) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    debug_assert!(p > 0.0, "p must be positive");

    if (p - 1.0).abs() < Float::EPSILON {
        manhattan_distance(a, b)
    } else if (p - 2.0).abs() < Float::EPSILON {
        euclidean_distance(a, b)
    } else {
        let mut sum = 0.0;
        for (x, y) in a.iter().zip(b.iter()) {
            sum += (x - y).abs().powf(p);
        }
        sum.powf(1.0 / p)
    }
}

/// Compute cosine similarity between two vectors
pub fn cosine_similarity(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Use optimized implementation that calculates 1 - cosine_distance
    1.0 - cosine_distance_optimized(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Internal optimized cosine distance implementation
#[inline]
fn cosine_distance_optimized(a: &[Float], b: &[Float]) -> Float {
    // Use SIMD optimization for f32 if available
    if TypeId::of::<Float>() == TypeId::of::<f32>() {
        let a_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(a) };
        let b_f32 = unsafe { std::mem::transmute::<&[Float], &[f32]>(b) };
        let cosine_sim = SimdDistanceOps::cosine_similarity_f32(a_f32, b_f32);
        let result_f32 = 1.0 - cosine_sim;
        return result_f32 as Float;
    } else if TypeId::of::<Float>() == TypeId::of::<f64>() {
        // Convert to f32 for SIMD, then back to f64
        let a_f32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b_f32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        let cosine_sim = SimdDistanceOps::cosine_similarity_f32(&a_f32, &b_f32);
        let result_f32 = 1.0 - cosine_sim;
        return (result_f32 as f64) as Float;
    }

    // Fallback to scalar implementation for other types
    cosine_distance_scalar(a, b)
}

/// Scalar fallback for cosine distance
#[inline]
fn cosine_distance_scalar(a: &[Float], b: &[Float]) -> Float {
    let dot_product: Float = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: Float = a.iter().map(|x| x * x).sum::<Float>().sqrt();
    let norm_b: Float = b.iter().map(|x| x * x).sum::<Float>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0 // Maximum distance for zero vectors
    } else {
        1.0 - (dot_product / (norm_a * norm_b))
    }
}

/// Compute cosine distance (1 - cosine similarity)
pub fn cosine_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    // Use SIMD optimization for f32 if available
    cosine_distance_optimized(a.as_slice().unwrap(), b.as_slice().unwrap())
}

// Scalar f32 versions for high-performance applications

/// Euclidean distance for f32 arrays (using SIMD optimization)
pub fn euclidean_distance_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    SimdDistanceOps::euclidean_distance_f32(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Manhattan distance for f32 arrays (using SIMD optimization)
pub fn manhattan_distance_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    SimdDistanceOps::manhattan_distance_f32(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Cosine distance for f32 arrays (using SIMD optimization)
pub fn cosine_distance_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    1.0 - SimdDistanceOps::cosine_similarity_f32(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Cosine similarity for f32 arrays (using SIMD optimization)
pub fn cosine_similarity_f32(a: &Array1<f32>, b: &Array1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");
    SimdDistanceOps::cosine_similarity_f32(a.as_slice().unwrap(), b.as_slice().unwrap())
}

/// Scalar implementation for f32 Euclidean distance
#[inline]
#[allow(dead_code)]
fn euclidean_distance_scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = x - y;
        sum += diff * diff;
    }
    sum.sqrt()
}

/// Scalar implementation for f32 Manhattan distance
#[inline]
#[allow(dead_code)]
fn manhattan_distance_scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        sum += (x - y).abs();
    }
    sum
}

/// Scalar implementation for f32 cosine distance
#[inline]
#[allow(dead_code)]
fn cosine_distance_scalar_f32(a: &[f32], b: &[f32]) -> f32 {
    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0 // Maximum distance for zero vectors
    } else {
        1.0 - (dot_product / (norm_a * norm_b))
    }
}

/// Compute Hamming distance between two arrays
/// Hamming distance counts the number of positions where elements differ
/// Useful for binary/categorical data and error correction
pub fn hamming_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let mut count = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        if (x - y).abs() > Float::EPSILON {
            count += 1.0;
        }
    }
    count
}

/// Compute normalized Hamming distance (0 to 1 range)
/// Normalizes Hamming distance by the vector length
pub fn hamming_distance_normalized(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    if a.is_empty() {
        return 0.0;
    }

    hamming_distance(a, b) / (a.len() as Float)
}

/// Compute Jaccard similarity between two binary arrays
/// Jaccard similarity = |intersection| / |union|
/// Works with binary (0/1) or boolean-like data
pub fn jaccard_similarity(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let mut intersection = 0.0;
    let mut union = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        let x_nonzero = x.abs() > Float::EPSILON;
        let y_nonzero = y.abs() > Float::EPSILON;

        if x_nonzero && y_nonzero {
            intersection += 1.0;
        }
        if x_nonzero || y_nonzero {
            union += 1.0;
        }
    }

    if union > 0.0 {
        intersection / union
    } else {
        1.0 // Both vectors are zero, perfect similarity
    }
}

/// Compute Jaccard distance (1 - Jaccard similarity)
pub fn jaccard_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    1.0 - jaccard_similarity(a, b)
}

/// Compute Canberra distance between two arrays
/// Canberra distance is sensitive to small changes near zero
/// More robust to outliers than Euclidean distance
pub fn canberra_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let mut sum = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let numerator = (x - y).abs();
        let denominator = x.abs() + y.abs();

        if denominator > Float::EPSILON {
            sum += numerator / denominator;
        }
        // If both x and y are zero, the contribution is 0 (0/0 = 0 by convention)
    }
    sum
}

/// Compute Chebyshev distance (L-infinity norm) between two arrays
/// Chebyshev distance is the maximum absolute difference between components
/// Also known as supremum metric or maximum metric
pub fn chebyshev_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let mut max_diff = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let diff = (x - y).abs();
        if diff > max_diff {
            max_diff = diff;
        }
    }
    max_diff
}

/// Compute Braycurtis distance between two arrays
/// Braycurtis distance is the sum of absolute differences divided by sum of absolute values
/// Useful for ecological data and compositional data analysis
pub fn braycurtis_distance(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    debug_assert_eq!(a.len(), b.len(), "Arrays must have the same length");

    let mut numerator = 0.0;
    let mut denominator = 0.0;

    for (x, y) in a.iter().zip(b.iter()) {
        numerator += (x - y).abs();
        denominator += x.abs() + y.abs();
    }

    if denominator > Float::EPSILON {
        numerator / denominator
    } else {
        0.0 // Both vectors are zero
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(euclidean_distance(&a, &b), 5.0, epsilon = 1e-10);

        let c = array![1.0, 1.0];
        assert_abs_diff_eq!(euclidean_distance(&c, &c), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_manhattan_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];
        assert_abs_diff_eq!(manhattan_distance(&a, &b), 7.0, epsilon = 1e-10);

        let c = array![1.0, 1.0];
        assert_abs_diff_eq!(manhattan_distance(&c, &c), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_minkowski_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];

        // p=1 should equal Manhattan distance
        assert_abs_diff_eq!(minkowski_distance(&a, &b, 1.0), 7.0, epsilon = 1e-10);

        // p=2 should equal Euclidean distance
        assert_abs_diff_eq!(minkowski_distance(&a, &b, 2.0), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert_abs_diff_eq!(cosine_similarity(&a, &b), 0.0, epsilon = 1e-6);

        let c = array![1.0, 1.0];
        let d = array![1.0, 1.0];
        assert_abs_diff_eq!(cosine_similarity(&c, &d), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance() {
        let a = array![1.0, 0.0];
        let b = array![0.0, 1.0];
        assert_abs_diff_eq!(cosine_distance(&a, &b), 1.0, epsilon = 1e-6);

        let c = array![1.0, 1.0];
        let d = array![1.0, 1.0];
        assert_abs_diff_eq!(cosine_distance(&c, &d), 0.0, epsilon = 1e-6);
    }

    // Tests for SIMD-optimized f32 functions
    #[test]
    fn test_euclidean_distance_f32() {
        let a = array![0.0f32, 0.0f32];
        let b = array![3.0f32, 4.0f32];
        assert_abs_diff_eq!(euclidean_distance_f32(&a, &b), 5.0f32, epsilon = 1e-6);

        let c = array![1.0f32, 1.0f32];
        assert_abs_diff_eq!(euclidean_distance_f32(&c, &c), 0.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_manhattan_distance_f32() {
        let a = array![0.0f32, 0.0f32];
        let b = array![3.0f32, 4.0f32];
        assert_abs_diff_eq!(manhattan_distance_f32(&a, &b), 7.0f32, epsilon = 1e-6);

        let c = array![1.0f32, 1.0f32];
        assert_abs_diff_eq!(manhattan_distance_f32(&c, &c), 0.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_distance_f32() {
        let a = array![1.0f32, 0.0f32];
        let b = array![0.0f32, 1.0f32];
        assert_abs_diff_eq!(cosine_distance_f32(&a, &b), 1.0f32, epsilon = 1e-6);

        let c = array![1.0f32, 1.0f32];
        let d = array![1.0f32, 1.0f32];
        assert_abs_diff_eq!(cosine_distance_f32(&c, &d), 0.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_cosine_similarity_f32() {
        let a = array![1.0f32, 0.0f32];
        let b = array![0.0f32, 1.0f32];
        assert_abs_diff_eq!(cosine_similarity_f32(&a, &b), 0.0f32, epsilon = 1e-6);

        let c = array![1.0f32, 1.0f32];
        let d = array![1.0f32, 1.0f32];
        assert_abs_diff_eq!(cosine_similarity_f32(&c, &d), 1.0f32, epsilon = 1e-6);
    }

    #[test]
    fn test_hamming_distance() {
        // Test basic Hamming distance
        let a = array![1.0, 0.0, 1.0, 1.0];
        let b = array![1.0, 1.0, 0.0, 1.0];
        assert_abs_diff_eq!(hamming_distance(&a, &b), 2.0, epsilon = 1e-10);

        // Test identical vectors
        let c = array![1.0, 1.0, 0.0];
        assert_abs_diff_eq!(hamming_distance(&c, &c), 0.0, epsilon = 1e-10);

        // Test completely different vectors
        let d = array![1.0, 1.0, 1.0];
        let e = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(hamming_distance(&d, &e), 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hamming_distance_normalized() {
        let a = array![1.0, 0.0, 1.0, 1.0];
        let b = array![1.0, 1.0, 0.0, 1.0];
        assert_abs_diff_eq!(hamming_distance_normalized(&a, &b), 0.5, epsilon = 1e-10);

        // Test empty vectors
        let empty_a = array![];
        let empty_b = array![];
        assert_abs_diff_eq!(
            hamming_distance_normalized(&empty_a, &empty_b),
            0.0,
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_jaccard_similarity() {
        // Test basic Jaccard similarity
        let a = array![1.0, 1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 1.0, 0.0];
        // Intersection: 1 element (index 0), Union: 3 elements (indices 0, 1, 2)
        assert_abs_diff_eq!(jaccard_similarity(&a, &b), 1.0 / 3.0, epsilon = 1e-10);

        // Test identical non-zero vectors
        let c = array![1.0, 1.0, 1.0];
        assert_abs_diff_eq!(jaccard_similarity(&c, &c), 1.0, epsilon = 1e-10);

        // Test disjoint sets
        let d = array![1.0, 0.0, 0.0];
        let e = array![0.0, 1.0, 0.0];
        assert_abs_diff_eq!(jaccard_similarity(&d, &e), 0.0, epsilon = 1e-10);

        // Test zero vectors (should return 1.0)
        let zero_a = array![0.0, 0.0, 0.0];
        let zero_b = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(jaccard_similarity(&zero_a, &zero_b), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_jaccard_distance() {
        let a = array![1.0, 1.0, 0.0, 0.0];
        let b = array![1.0, 0.0, 1.0, 0.0];
        // Jaccard similarity is 1/3, so distance should be 2/3
        assert_abs_diff_eq!(jaccard_distance(&a, &b), 2.0 / 3.0, epsilon = 1e-10);

        // Test identical vectors (distance should be 0)
        let c = array![1.0, 1.0, 1.0];
        assert_abs_diff_eq!(jaccard_distance(&c, &c), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_canberra_distance() {
        // Test basic Canberra distance
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 4.0];
        // |1-2|/(|1|+|2|) + |2-3|/(|2|+|3|) + |3-4|/(|3|+|4|) = 1/3 + 1/5 + 1/7
        let expected = 1.0 / 3.0 + 1.0 / 5.0 + 1.0 / 7.0;
        assert_abs_diff_eq!(canberra_distance(&a, &b), expected, epsilon = 1e-10);

        // Test identical vectors
        let c = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(canberra_distance(&c, &c), 0.0, epsilon = 1e-10);

        // Test zero elements (should be handled gracefully)
        let d = array![0.0, 1.0, 2.0];
        let e = array![0.0, 2.0, 3.0];
        // First element contributes 0, second: |1-2|/(|1|+|2|) = 1/3, third: |2-3|/(|2|+|3|) = 1/5
        let expected_zero = 0.0 + 1.0 / 3.0 + 1.0 / 5.0;
        assert_abs_diff_eq!(canberra_distance(&d, &e), expected_zero, epsilon = 1e-10);
    }

    #[test]
    fn test_chebyshev_distance() {
        // Test basic Chebyshev distance (maximum absolute difference)
        let a = array![1.0, 5.0, 3.0];
        let b = array![2.0, 1.0, 6.0];
        // Differences: |1-2| = 1, |5-1| = 4, |3-6| = 3, max = 4
        assert_abs_diff_eq!(chebyshev_distance(&a, &b), 4.0, epsilon = 1e-10);

        // Test identical vectors
        let c = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(chebyshev_distance(&c, &c), 0.0, epsilon = 1e-10);

        // Test single element difference
        let d = array![5.0, 0.0, 0.0];
        let e = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(chebyshev_distance(&d, &e), 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_braycurtis_distance() {
        // Test basic Braycurtis distance
        let a = array![1.0, 2.0, 3.0];
        let b = array![2.0, 3.0, 1.0];
        // Numerator: |1-2| + |2-3| + |3-1| = 1 + 1 + 2 = 4
        // Denominator: |1| + |2| + |2| + |3| + |3| + |1| = 1 + 2 + 2 + 3 + 3 + 1 = 12
        // Distance: 4/12 = 1/3
        assert_abs_diff_eq!(braycurtis_distance(&a, &b), 1.0 / 3.0, epsilon = 1e-10);

        // Test identical vectors
        let c = array![1.0, 2.0, 3.0];
        assert_abs_diff_eq!(braycurtis_distance(&c, &c), 0.0, epsilon = 1e-10);

        // Test zero vectors (should return 0.0)
        let zero_a = array![0.0, 0.0, 0.0];
        let zero_b = array![0.0, 0.0, 0.0];
        assert_abs_diff_eq!(braycurtis_distance(&zero_a, &zero_b), 0.0, epsilon = 1e-10);

        // Test maximum distance case
        let d = array![1.0, 0.0];
        let e = array![0.0, 1.0];
        // Numerator: |1-0| + |0-1| = 2, Denominator: 1 + 0 + 0 + 1 = 2, Distance: 2/2 = 1.0
        assert_abs_diff_eq!(braycurtis_distance(&d, &e), 1.0, epsilon = 1e-10);
    }
}
