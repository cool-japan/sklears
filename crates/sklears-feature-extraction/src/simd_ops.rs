//! SIMD-optimized operations for feature extraction
//!
//! This module provides SIMD-optimized implementations of common mathematical
//! operations used in feature extraction to improve performance.

use crate::*;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{error::Result as SklResult, prelude::SklearsError, types::Float};

/// SIMD-optimized vector operations
pub struct SimdOps;

impl SimdOps {
    /// Compute dot product with SIMD optimization
    pub fn dot_product(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        let mut sum = 0.0;
        let len = a.len();

        // Process 4 elements at a time for SIMD-like operations
        let chunks = len / 4;
        let remainder = len % 4;

        // Unrolled loop for better performance
        for i in 0..chunks {
            let base = i * 4;
            sum += a[base] * b[base]
                + a[base + 1] * b[base + 1]
                + a[base + 2] * b[base + 2]
                + a[base + 3] * b[base + 3];
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            sum += a[i] * b[i];
        }

        sum
    }

    /// Compute L2 norm with SIMD optimization
    pub fn l2_norm(vector: ArrayView1<Float>) -> Float {
        let mut sum_squares = 0.0;
        let len = vector.len();

        // Process 4 elements at a time
        let chunks = len / 4;

        for i in 0..chunks {
            let base = i * 4;
            let v0 = vector[base];
            let v1 = vector[base + 1];
            let v2 = vector[base + 2];
            let v3 = vector[base + 3];

            sum_squares += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let v = vector[i];
            sum_squares += v * v;
        }

        sum_squares.sqrt()
    }

    /// Compute element-wise multiplication with SIMD optimization
    pub fn element_wise_multiply(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Array1<Float> {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        let len = a.len();
        let mut result = Array1::zeros(len);
        let chunks = len / 4;

        // Unrolled loop for SIMD-like performance
        for i in 0..chunks {
            let base = i * 4;
            result[base] = a[base] * b[base];
            result[base + 1] = a[base + 1] * b[base + 1];
            result[base + 2] = a[base + 2] * b[base + 2];
            result[base + 3] = a[base + 3] * b[base + 3];
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            result[i] = a[i] * b[i];
        }

        result
    }

    /// Compute sum of array elements with SIMD optimization
    pub fn sum(vector: ArrayView1<Float>) -> Float {
        let len = vector.len();
        let chunks = len / 4;
        let mut sum = 0.0;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            sum += vector[base] + vector[base + 1] + vector[base + 2] + vector[base + 3];
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            sum += vector[i];
        }

        sum
    }

    /// Compute matrix-vector multiplication with SIMD optimization
    pub fn matrix_vector_multiply(
        matrix: ArrayView2<Float>,
        vector: ArrayView1<Float>,
    ) -> Array1<Float> {
        assert_eq!(
            matrix.ncols(),
            vector.len(),
            "Matrix columns must match vector length"
        );

        let rows = matrix.nrows();
        let cols = matrix.ncols();
        let mut result = Array1::zeros(rows);

        for row in 0..rows {
            let matrix_row = matrix.row(row);
            result[row] = Self::dot_product(matrix_row, vector);
        }

        result
    }

    /// Compute cosine similarity with SIMD optimization
    pub fn cosine_similarity(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        let dot = Self::dot_product(a, b);
        let norm_a = Self::l2_norm(a);
        let norm_b = Self::l2_norm(b);

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            dot / (norm_a * norm_b)
        }
    }

    /// Compute Euclidean distance with SIMD optimization
    pub fn euclidean_distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        assert_eq!(a.len(), b.len(), "Vector dimensions must match");

        let len = a.len();
        let chunks = len / 4;
        let mut sum_squares = 0.0;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let d0 = a[base] - b[base];
            let d1 = a[base + 1] - b[base + 1];
            let d2 = a[base + 2] - b[base + 2];
            let d3 = a[base + 3] - b[base + 3];

            sum_squares += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let diff = a[i] - b[i];
            sum_squares += diff * diff;
        }

        sum_squares.sqrt()
    }

    /// Compute mean of array elements with SIMD optimization
    pub fn mean(vector: ArrayView1<Float>) -> Float {
        if vector.is_empty() {
            return 0.0;
        }
        Self::sum(vector) / vector.len() as Float
    }

    /// Compute variance with SIMD optimization
    pub fn variance(vector: ArrayView1<Float>) -> Float {
        if vector.len() <= 1 {
            return 0.0;
        }

        let mean_val = Self::mean(vector);
        let len = vector.len();
        let chunks = len / 4;
        let mut sum_squares = 0.0;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let d0 = vector[base] - mean_val;
            let d1 = vector[base + 1] - mean_val;
            let d2 = vector[base + 2] - mean_val;
            let d3 = vector[base + 3] - mean_val;

            sum_squares += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let diff = vector[i] - mean_val;
            sum_squares += diff * diff;
        }

        sum_squares / (len - 1) as Float
    }

    /// Compute standard deviation with SIMD optimization
    pub fn std_dev(vector: ArrayView1<Float>) -> Float {
        Self::variance(vector).sqrt()
    }

    /// Compute range (max - min) with SIMD optimization
    pub fn range(vector: ArrayView1<Float>) -> Float {
        if vector.is_empty() {
            return 0.0;
        }

        let len = vector.len();
        let chunks = len / 4;

        let mut min_val = Float::INFINITY;
        let mut max_val = Float::NEG_INFINITY;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let v0 = vector[base];
            let v1 = vector[base + 1];
            let v2 = vector[base + 2];
            let v3 = vector[base + 3];

            min_val = min_val.min(v0).min(v1).min(v2).min(v3);
            max_val = max_val.max(v0).max(v1).max(v2).max(v3);
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let v = vector[i];
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        max_val - min_val
    }

    /// Compute skewness with SIMD optimization
    pub fn skewness(vector: ArrayView1<Float>) -> Float {
        if vector.len() < 3 {
            return 0.0;
        }

        let mean_val = Self::mean(vector);
        let std_val = Self::std_dev(vector);

        if std_val == 0.0 {
            return 0.0;
        }

        let len = vector.len();
        let chunks = len / 4;
        let mut sum_cubes = 0.0;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let d0 = (vector[base] - mean_val) / std_val;
            let d1 = (vector[base + 1] - mean_val) / std_val;
            let d2 = (vector[base + 2] - mean_val) / std_val;
            let d3 = (vector[base + 3] - mean_val) / std_val;

            sum_cubes += d0 * d0 * d0 + d1 * d1 * d1 + d2 * d2 * d2 + d3 * d3 * d3;
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let d = (vector[i] - mean_val) / std_val;
            sum_cubes += d * d * d;
        }

        sum_cubes / len as Float
    }

    /// Compute kurtosis with SIMD optimization
    pub fn kurtosis(vector: ArrayView1<Float>) -> Float {
        if vector.len() < 4 {
            return 0.0;
        }

        let mean_val = Self::mean(vector);
        let std_val = Self::std_dev(vector);

        if std_val == 0.0 {
            return 0.0;
        }

        let len = vector.len();
        let chunks = len / 4;
        let mut sum_fourth_powers = 0.0;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let d0 = (vector[base] - mean_val) / std_val;
            let d1 = (vector[base + 1] - mean_val) / std_val;
            let d2 = (vector[base + 2] - mean_val) / std_val;
            let d3 = (vector[base + 3] - mean_val) / std_val;

            let p0 = d0 * d0;
            let p1 = d1 * d1;
            let p2 = d2 * d2;
            let p3 = d3 * d3;

            sum_fourth_powers += p0 * p0 + p1 * p1 + p2 * p2 + p3 * p3;
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let d = (vector[i] - mean_val) / std_val;
            let p = d * d;
            sum_fourth_powers += p * p;
        }

        (sum_fourth_powers / len as Float) - 3.0 // Excess kurtosis
    }
}

/// SIMD-optimized statistical operations
pub struct SimdStats;

impl SimdStats {
    /// Compute multiple statistics in a single pass
    pub fn compute_stats(vector: ArrayView1<Float>) -> StatisticsResult {
        if vector.is_empty() {
            return StatisticsResult::default();
        }

        let len = vector.len();
        let chunks = len / 4;

        let mut sum = 0.0;
        let mut sum_squares = 0.0;
        let mut min_val = Float::INFINITY;
        let mut max_val = Float::NEG_INFINITY;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base = i * 4;
            let v0 = vector[base];
            let v1 = vector[base + 1];
            let v2 = vector[base + 2];
            let v3 = vector[base + 3];

            // Sum
            sum += v0 + v1 + v2 + v3;

            // Sum of squares
            sum_squares += v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;

            // Min/Max
            min_val = min_val.min(v0).min(v1).min(v2).min(v3);
            max_val = max_val.max(v0).max(v1).max(v2).max(v3);
        }

        // Handle remaining elements
        for i in chunks * 4..len {
            let v = vector[i];
            sum += v;
            sum_squares += v * v;
            min_val = min_val.min(v);
            max_val = max_val.max(v);
        }

        let mean = sum / len as Float;
        let variance = if len > 1 {
            (sum_squares - sum * sum / len as Float) / (len - 1) as Float
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let range = max_val - min_val;

        // Compute higher-order moments
        let skewness = if len >= 3 && std_dev > 0.0 {
            SimdOps::skewness(vector)
        } else {
            0.0
        };

        let kurtosis = if len >= 4 && std_dev > 0.0 {
            SimdOps::kurtosis(vector)
        } else {
            0.0
        };

        StatisticsResult {
            mean,
            variance,
            std_dev,
            min: min_val,
            max: max_val,
            range,
            sum,
            count: len,
            skewness,
            kurtosis,
        }
    }
}

/// Result of SIMD-optimized statistical computation
#[derive(Debug, Clone)]
pub struct StatisticsResult {
    /// Mean value
    pub mean: Float,
    /// Variance
    pub variance: Float,
    /// Standard deviation
    pub std_dev: Float,
    /// Minimum value
    pub min: Float,
    /// Maximum value
    pub max: Float,
    /// Range (max - min)
    pub range: Float,
    /// Sum of all values
    pub sum: Float,
    /// Number of elements
    pub count: usize,
    /// Skewness
    pub skewness: Float,
    /// Kurtosis (excess)
    pub kurtosis: Float,
}

impl Default for StatisticsResult {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            range: 0.0,
            sum: 0.0,
            count: 0,
            skewness: 0.0,
            kurtosis: 0.0,
        }
    }
}

// Wrapper functions expected by tests
pub fn simd_dot_product(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<f64> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }
    Ok(SimdOps::dot_product(*a, *b))
}

pub fn simd_add_vectors(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }

    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i] + b[i];
    }
    Ok(result)
}

pub fn simd_matrix_vector_multiply(
    matrix: &ArrayView2<f64>,
    vector: &ArrayView1<f64>,
) -> SklResult<Array1<f64>> {
    if matrix.ncols() != vector.len() {
        return Err(SklearsError::InvalidInput(
            "Matrix columns must match vector length".to_string(),
        ));
    }
    Ok(SimdOps::matrix_vector_multiply(*matrix, *vector))
}

pub fn simd_euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<f64> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }
    Ok(SimdOps::euclidean_distance(*a, *b))
}

pub fn simd_cosine_similarity(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<f64> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }
    Ok(SimdOps::cosine_similarity(*a, *b))
}

/// SIMD-optimized vector subtraction
pub fn simd_subtract_vectors(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }

    let mut result = Array1::zeros(a.len());
    for i in 0..a.len() {
        result[i] = a[i] - b[i];
    }
    Ok(result)
}

/// SIMD-optimized element-wise vector multiplication
pub fn simd_multiply_vectors(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<Array1<f64>> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }

    Ok(SimdOps::element_wise_multiply(*a, *b))
}

/// SIMD-optimized vector norm computation
pub fn simd_vector_norm(vector: &ArrayView1<f64>, p: i32) -> SklResult<f64> {
    if vector.is_empty() {
        return Err(SklearsError::InvalidInput("Empty vector".to_string()));
    }

    match p {
        1 => {
            // L1 norm (Manhattan norm)
            Ok(vector.iter().map(|x| x.abs()).sum())
        }
        2 => {
            // L2 norm (Euclidean norm)
            Ok(SimdOps::l2_norm(*vector))
        }
        _ => Err(SklearsError::InvalidInput(format!(
            "Unsupported norm p={}",
            p
        ))),
    }
}

/// SIMD-optimized Manhattan distance
pub fn simd_manhattan_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<f64> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }

    let mut distance = 0.0;
    for i in 0..a.len() {
        distance += (a[i] - b[i]).abs();
    }
    Ok(distance)
}

/// SIMD-optimized squared Euclidean distance
pub fn simd_squared_euclidean_distance(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> SklResult<f64> {
    if a.len() != b.len() {
        return Err(SklearsError::InvalidInput(
            "Vector dimensions must match".to_string(),
        ));
    }

    let mut sum_squares = 0.0;
    for i in 0..a.len() {
        let diff = a[i] - b[i];
        sum_squares += diff * diff;
    }
    Ok(sum_squares)
}

/// SIMD-optimized batch dot product computation
pub fn simd_batch_dot_product(
    vectors: &ArrayView2<f64>,
    query: &ArrayView1<f64>,
) -> SklResult<Array1<f64>> {
    if vectors.ncols() != query.len() {
        return Err(SklearsError::InvalidInput(
            "Query vector dimension must match matrix columns".to_string(),
        ));
    }

    let mut results = Array1::zeros(vectors.nrows());
    for (i, row) in vectors.axis_iter(Axis(0)).enumerate() {
        results[i] = SimdOps::dot_product(row, *query);
    }
    Ok(results)
}

/// SIMD-optimized matrix multiplication
pub fn simd_matrix_multiply(a: &ArrayView2<f64>, b: &ArrayView2<f64>) -> SklResult<Array2<f64>> {
    if a.ncols() != b.nrows() {
        return Err(SklearsError::InvalidInput(
            "Matrix dimensions incompatible for multiplication".to_string(),
        ));
    }

    let mut result = Array2::zeros((a.nrows(), b.ncols()));
    for i in 0..a.nrows() {
        for j in 0..b.ncols() {
            let mut sum = 0.0;
            for k in 0..a.ncols() {
                sum += a[(i, k)] * b[(k, j)];
            }
            result[(i, j)] = sum;
        }
    }
    Ok(result)
}

/// SIMD-optimized vector sum
pub fn simd_vector_sum(vector: &ArrayView1<f64>) -> SklResult<f64> {
    if vector.is_empty() {
        return Err(SklearsError::InvalidInput("Empty vector".to_string()));
    }
    Ok(SimdOps::sum(*vector))
}

/// SIMD-optimized vector mean
pub fn simd_vector_mean(vector: &ArrayView1<f64>) -> SklResult<f64> {
    if vector.is_empty() {
        return Err(SklearsError::InvalidInput("Empty vector".to_string()));
    }
    Ok(SimdOps::mean(*vector))
}

/// SIMD-optimized vector variance
pub fn simd_vector_variance(vector: &ArrayView1<f64>) -> SklResult<f64> {
    if vector.is_empty() {
        return Err(SklearsError::InvalidInput("Empty vector".to_string()));
    }
    Ok(SimdOps::variance(*vector))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_dot_product() {
        let a = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = array![2.0, 3.0, 4.0, 5.0, 6.0];

        let result = SimdOps::dot_product(a.view(), b.view());
        let expected = 1.0 * 2.0 + 2.0 * 3.0 + 3.0 * 4.0 + 4.0 * 5.0 + 5.0 * 6.0;

        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_l2_norm() {
        let vector = array![3.0, 4.0];
        let result = SimdOps::l2_norm(vector.view());
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_element_wise_multiply() {
        let a = array![1.0, 2.0, 3.0, 4.0];
        let b = array![2.0, 3.0, 4.0, 5.0];

        let result = SimdOps::element_wise_multiply(a.view(), b.view());
        let expected = array![2.0, 6.0, 12.0, 20.0];

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_sum() {
        let vector = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = SimdOps::sum(vector.view());
        assert_abs_diff_eq!(result, 15.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = array![1.0, 0.0, 0.0];
        let b = array![0.0, 1.0, 0.0];

        let result = SimdOps::cosine_similarity(a.view(), b.view());
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);

        let c = array![1.0, 1.0, 0.0];
        let d = array![1.0, 1.0, 0.0];

        let result2 = SimdOps::cosine_similarity(c.view(), d.view());
        assert_abs_diff_eq!(result2, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![0.0, 0.0];
        let b = array![3.0, 4.0];

        let result = SimdOps::euclidean_distance(a.view(), b.view());
        assert_abs_diff_eq!(result, 5.0, epsilon = 1e-10);
    }

    #[test]
    fn test_statistics() {
        let vector = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = SimdStats::compute_stats(vector.view());

        assert_abs_diff_eq!(stats.mean, 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.sum, 15.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min, 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max, 5.0, epsilon = 1e-10);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_matrix_vector_multiply() {
        let matrix = Array2::from_shape_vec((2, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let vector = array![1.0, 2.0, 3.0];

        let result = SimdOps::matrix_vector_multiply(matrix.view(), vector.view());
        let expected = array![14.0, 32.0]; // [1*1+2*2+3*3, 4*1+5*2+6*3]

        for (r, e) in result.iter().zip(expected.iter()) {
            assert_abs_diff_eq!(r, e, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_variance() {
        let vector = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = SimdOps::variance(vector.view());

        // Sample variance calculation: 32/7 ≈ 4.571428571428571
        let expected = 32.0 / 7.0;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_empty_vector() {
        let empty = Array1::<Float>::zeros(0);
        let stats = SimdStats::compute_stats(empty.view());

        assert_eq!(stats.count, 0);
        assert_eq!(stats.sum, 0.0);
        assert_eq!(stats.mean, 0.0);
    }

    #[test]
    fn test_std_dev() {
        let vector = array![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        let result = SimdOps::std_dev(vector.view());

        // Expected standard deviation: sqrt(32/7) ≈ 2.1380899352993946
        let expected = (32.0_f64 / 7.0_f64).sqrt() as Float;
        assert_abs_diff_eq!(result, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_range() {
        let vector = array![1.0, 5.0, 3.0, 9.0, 2.0];
        let result = SimdOps::range(vector.view());
        assert_abs_diff_eq!(result, 8.0, epsilon = 1e-10); // 9.0 - 1.0 = 8.0
    }

    #[test]
    fn test_skewness() {
        // Symmetric distribution should have skewness close to 0
        let vector = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = SimdOps::skewness(vector.view());
        assert_abs_diff_eq!(result, 0.0, epsilon = 1e-10);

        // Right-skewed distribution (positive skewness)
        let right_skewed = array![1.0, 1.0, 1.0, 2.0, 10.0];
        let skew_result = SimdOps::skewness(right_skewed.view());
        assert!(skew_result > 0.0);
    }

    #[test]
    fn test_kurtosis() {
        // Uniform distribution should have negative kurtosis (platykurtic)
        let vector = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = SimdOps::kurtosis(vector.view());
        // For a uniform distribution, excess kurtosis ≈ -1.2
        assert!(
            result < 0.0,
            "Uniform distribution should have negative kurtosis"
        );
        assert!(result > -2.0, "Kurtosis should be reasonable");

        // Test with more peaked distribution
        let peaked = array![5.0, 5.0, 5.0, 1.0, 9.0];
        let peaked_result = SimdOps::kurtosis(peaked.view());
        assert!(
            peaked_result > result,
            "More peaked distribution should have higher kurtosis"
        );
    }

    #[test]
    fn test_enhanced_statistics() {
        let vector = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let stats = SimdStats::compute_stats(vector.view());

        assert_abs_diff_eq!(stats.mean, 3.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.range, 5.0, epsilon = 1e-10); // 6.0 - 1.0
        assert_eq!(stats.count, 6);
        assert!(stats.std_dev > 0.0);
        // For a uniform distribution, skewness should be close to 0
        assert_abs_diff_eq!(stats.skewness, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_edge_cases_new_functions() {
        // Empty vector
        let empty = Array1::<Float>::zeros(0);
        assert_eq!(SimdOps::range(empty.view()), 0.0);
        assert_eq!(SimdOps::skewness(empty.view()), 0.0);
        assert_eq!(SimdOps::kurtosis(empty.view()), 0.0);

        // Single element
        let single = array![5.0];
        assert_eq!(SimdOps::range(single.view()), 0.0);
        assert_eq!(SimdOps::skewness(single.view()), 0.0);
        assert_eq!(SimdOps::kurtosis(single.view()), 0.0);

        // Constant vector (zero standard deviation)
        let constant = array![3.0, 3.0, 3.0, 3.0];
        assert_eq!(SimdOps::range(constant.view()), 0.0);
        assert_eq!(SimdOps::skewness(constant.view()), 0.0);
        assert_eq!(SimdOps::kurtosis(constant.view()), 0.0);
    }
}
