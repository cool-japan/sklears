//! Optimized numerical operations for high-performance imputation
//!
//! This module provides optimized implementations of common numerical operations
//! used in missing data imputation to achieve significant performance improvements.
//! Uses SIMD instructions and unsafe code for performance-critical paths.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{error::Result as SklResult, prelude::SklearsError};
use wide::f64x4;

/// Cache-friendly data layout for imputation operations
#[derive(Clone, Debug)]
pub struct CacheOptimizedData {
    /// Row-major data layout optimized for cache access
    data: Vec<f64>,
    /// Missing value indicators packed as bits
    missing_mask: Vec<u64>,
    /// Dimensions
    n_rows: usize,
    n_cols: usize,
    /// Cache line size alignment
    cache_line_size: usize,
}

impl CacheOptimizedData {
    /// Create cache-optimized data layout
    pub fn new(data: &Array2<f64>, missing_val: f64) -> Self {
        let (n_rows, n_cols) = data.dim();
        let cache_line_size = 64; // Common cache line size

        // Pad columns to cache line boundaries
        let padded_cols =
            ((n_cols * 8 + cache_line_size - 1) / cache_line_size) * cache_line_size / 8;
        let mut aligned_data = vec![0.0; n_rows * padded_cols];

        // Copy data with padding
        for i in 0..n_rows {
            for j in 0..n_cols {
                aligned_data[i * padded_cols + j] = data[[i, j]];
            }
        }

        // Create packed missing mask (64 bits per u64)
        let mask_len = (n_rows * n_cols + 63) / 64;
        let mut missing_mask = vec![0u64; mask_len];

        for i in 0..n_rows {
            for j in 0..n_cols {
                let idx = i * n_cols + j;
                let is_missing = if missing_val.is_nan() {
                    data[[i, j]].is_nan()
                } else {
                    (data[[i, j]] - missing_val).abs() < f64::EPSILON
                };

                if is_missing {
                    let word_idx = idx / 64;
                    let bit_idx = idx % 64;
                    missing_mask[word_idx] |= 1u64 << bit_idx;
                }
            }
        }

        Self {
            data: aligned_data,
            missing_mask,
            n_rows,
            n_cols,
            cache_line_size,
        }
    }

    /// Get value at position (i, j) with bounds checking
    pub fn get(&self, i: usize, j: usize) -> Option<f64> {
        if i < self.n_rows && j < self.n_cols {
            let padded_cols = ((self.n_cols * 8 + self.cache_line_size - 1) / self.cache_line_size)
                * self.cache_line_size
                / 8;
            Some(self.data[i * padded_cols + j])
        } else {
            None
        }
    }

    /// Check if value is missing
    pub fn is_missing(&self, i: usize, j: usize) -> bool {
        if i < self.n_rows && j < self.n_cols {
            let idx = i * self.n_cols + j;
            let word_idx = idx / 64;
            let bit_idx = idx % 64;
            (self.missing_mask[word_idx] & (1u64 << bit_idx)) != 0
        } else {
            false
        }
    }

    /// Get row slice for cache-friendly access
    pub fn get_row(&self, i: usize) -> Option<&[f64]> {
        if i < self.n_rows {
            let padded_cols = ((self.n_cols * 8 + self.cache_line_size - 1) / self.cache_line_size)
                * self.cache_line_size
                / 8;
            let start = i * padded_cols;
            Some(&self.data[start..start + self.n_cols])
        } else {
            None
        }
    }
}

/// Optimized distance calculations using SIMD and unsafe code
pub struct SimdDistanceCalculator;

impl SimdDistanceCalculator {
    /// Optimized Euclidean distance calculation using SIMD
    pub fn euclidean_distance_simd(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");

        if x.len() < 4 {
            // Fallback for small vectors
            return x
                .iter()
                .zip(y.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>()
                .sqrt();
        }

        unsafe { Self::euclidean_distance_simd_unsafe(x, y) }
    }

    /// Unsafe SIMD implementation for maximum performance
    unsafe fn euclidean_distance_simd_unsafe(x: &[f64], y: &[f64]) -> f64 {
        let len = x.len();
        let chunks = len / 4;

        let mut sum = f64x4::splat(0.0);

        // Process 4 elements at a time
        for i in 0..chunks {
            let x_chunk = f64x4::new([x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3]]);
            let y_chunk = f64x4::new([y[i * 4], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3]]);
            let diff = x_chunk - y_chunk;
            sum += diff * diff;
        }

        // Sum the SIMD lanes
        let sum_array = sum.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            let diff = x[i] - y[i];
            result += diff * diff;
        }

        result.sqrt()
    }

    /// Optimized Manhattan distance calculation using SIMD
    pub fn manhattan_distance_simd(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");

        if x.len() < 4 {
            // Fallback for small vectors
            return x.iter().zip(y.iter()).map(|(a, b)| (a - b).abs()).sum();
        }

        unsafe { Self::manhattan_distance_simd_unsafe(x, y) }
    }

    /// Unsafe SIMD implementation for Manhattan distance
    unsafe fn manhattan_distance_simd_unsafe(x: &[f64], y: &[f64]) -> f64 {
        let len = x.len();
        let chunks = len / 4;

        let mut sum = f64x4::splat(0.0);

        // Process 4 elements at a time
        for i in 0..chunks {
            let x_chunk = f64x4::new([x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3]]);
            let y_chunk = f64x4::new([y[i * 4], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3]]);
            let diff = x_chunk - y_chunk;
            sum += diff.abs();
        }

        // Sum the SIMD lanes
        let sum_array = sum.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            result += (x[i] - y[i]).abs();
        }

        result
    }

    /// NaN-aware Euclidean distance for missing data
    pub fn nan_euclidean_distance_simd(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");

        let mut sum_sq = 0.0;
        let mut valid_count = 0;

        // Use vectorized operations where possible
        for (&x_val, &y_val) in x.iter().zip(y.iter()) {
            if !x_val.is_nan() && !y_val.is_nan() {
                let diff = x_val - y_val;
                sum_sq += diff * diff;
                valid_count += 1;
            }
        }

        if valid_count > 0 {
            (sum_sq / valid_count as f64).sqrt()
        } else {
            f64::INFINITY
        }
    }

    /// Optimized cosine similarity calculation
    pub fn cosine_similarity_simd(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");

        if x.len() < 4 {
            // Fallback for small vectors
            let dot_product: f64 = x.iter().zip(y.iter()).map(|(a, b)| a * b).sum();
            let norm_x: f64 = x.iter().map(|a| a * a).sum::<f64>().sqrt();
            let norm_y: f64 = y.iter().map(|a| a * a).sum::<f64>().sqrt();

            if norm_x == 0.0 || norm_y == 0.0 {
                return 0.0;
            }

            return dot_product / (norm_x * norm_y);
        }

        unsafe { Self::cosine_similarity_simd_unsafe(x, y) }
    }

    /// Unsafe SIMD implementation for cosine similarity
    unsafe fn cosine_similarity_simd_unsafe(x: &[f64], y: &[f64]) -> f64 {
        let len = x.len();
        let chunks = len / 4;

        let mut dot_product = f64x4::splat(0.0);
        let mut norm_x_sq = f64x4::splat(0.0);
        let mut norm_y_sq = f64x4::splat(0.0);

        // Process 4 elements at a time
        for i in 0..chunks {
            let x_chunk = f64x4::new([x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3]]);
            let y_chunk = f64x4::new([y[i * 4], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3]]);

            dot_product += x_chunk * y_chunk;
            norm_x_sq += x_chunk * x_chunk;
            norm_y_sq += y_chunk * y_chunk;
        }

        // Sum the SIMD lanes
        let dot_array = dot_product.to_array();
        let norm_x_array = norm_x_sq.to_array();
        let norm_y_array = norm_y_sq.to_array();

        let mut dot_result = dot_array[0] + dot_array[1] + dot_array[2] + dot_array[3];
        let mut norm_x_result =
            norm_x_array[0] + norm_x_array[1] + norm_x_array[2] + norm_x_array[3];
        let mut norm_y_result =
            norm_y_array[0] + norm_y_array[1] + norm_y_array[2] + norm_y_array[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            dot_result += x[i] * y[i];
            norm_x_result += x[i] * x[i];
            norm_y_result += y[i] * y[i];
        }

        let norm_x = norm_x_result.sqrt();
        let norm_y = norm_y_result.sqrt();

        if norm_x == 0.0 || norm_y == 0.0 {
            0.0
        } else {
            dot_result / (norm_x * norm_y)
        }
    }
}

/// Optimized statistical calculations using SIMD
pub struct SimdStatistics;

impl SimdStatistics {
    /// Optimized mean calculation using SIMD
    pub fn mean_simd(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }

        if data.len() < 4 {
            return data.iter().sum::<f64>() / data.len() as f64;
        }

        unsafe { Self::mean_simd_unsafe(data) }
    }

    /// Unsafe SIMD implementation for mean calculation
    unsafe fn mean_simd_unsafe(data: &[f64]) -> f64 {
        let len = data.len();
        let chunks = len / 4;

        let mut sum = f64x4::splat(0.0);

        // Process 8 elements at a time
        for i in 0..chunks {
            let chunk = f64x4::new([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
            sum += chunk;
        }

        // Sum the SIMD lanes
        let sum_array = sum.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        result += data
            .iter()
            .skip(chunks * 4)
            .take(len - chunks * 4)
            .sum::<f64>();

        result / len as f64
    }

    /// Optimized variance calculation using SIMD
    pub fn variance_simd(data: &[f64], mean: Option<f64>) -> f64 {
        if data.len() <= 1 {
            return 0.0;
        }

        let mean = mean.unwrap_or_else(|| Self::mean_simd(data));

        if data.len() < 4 {
            let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
            return sum_sq_diff / (data.len() - 1) as f64;
        }

        unsafe { Self::variance_simd_unsafe(data, mean) }
    }

    /// Unsafe SIMD implementation for variance calculation
    unsafe fn variance_simd_unsafe(data: &[f64], mean: f64) -> f64 {
        let len = data.len();
        let chunks = len / 4;

        let mean_vec = f64x4::splat(mean);
        let mut sum_sq_diff = f64x4::splat(0.0);

        // Process 8 elements at a time
        for i in 0..chunks {
            let chunk = f64x4::new([
                data[i * 4],
                data[i * 4 + 1],
                data[i * 4 + 2],
                data[i * 4 + 3],
            ]);
            let diff = chunk - mean_vec;
            sum_sq_diff += diff * diff;
        }

        // Sum the SIMD lanes
        let sum_array = sum_sq_diff.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        result += data
            .iter()
            .skip(chunks * 4)
            .take(len - chunks * 4)
            .map(|&x| {
                let diff = x - mean;
                diff * diff
            })
            .sum::<f64>();

        result / (len - 1) as f64
    }

    /// Optimized standard deviation calculation
    pub fn std_dev_simd(data: &[f64], mean: Option<f64>) -> f64 {
        Self::variance_simd(data, mean).sqrt()
    }

    /// Optimized min/max finding using SIMD
    pub fn min_max_simd(data: &[f64]) -> (f64, f64) {
        if data.is_empty() {
            return (f64::NAN, f64::NAN);
        }

        if data.len() == 1 {
            return (data[0], data[0]);
        }

        if data.len() < 4 {
            let mut min_val = data[0];
            let mut max_val = data[0];
            for &val in &data[1..] {
                if val < min_val {
                    min_val = val;
                }
                if val > max_val {
                    max_val = val;
                }
            }
            return (min_val, max_val);
        }

        unsafe { Self::min_max_simd_unsafe(data) }
    }

    /// Unsafe SIMD implementation for min/max finding
    unsafe fn min_max_simd_unsafe(data: &[f64]) -> (f64, f64) {
        let len = data.len();
        let chunks = len / 4;

        let mut min_result = f64::INFINITY;
        let mut max_result = f64::NEG_INFINITY;

        // Process 4 elements at a time
        for i in 0..chunks {
            let base_idx = i * 4;
            for j in 0..4 {
                let val = data[base_idx + j];
                if val < min_result {
                    min_result = val;
                }
                if val > max_result {
                    max_result = val;
                }
            }
        }

        // Handle remaining elements
        for &val in data.iter().skip(chunks * 4).take(len - chunks * 4) {
            if val < min_result {
                min_result = val;
            }
            if val > max_result {
                max_result = val;
            }
        }

        (min_result, max_result)
    }

    /// Optimized quantile calculation using SIMD for sorting
    pub fn quantile_simd(data: &[f64], q: f64) -> f64 {
        if data.is_empty() {
            return f64::NAN;
        }

        // Create a copy for sorting
        let mut sorted_data: Vec<f64> = data.iter().filter(|&&x| !x.is_nan()).cloned().collect();

        if sorted_data.is_empty() {
            return f64::NAN;
        }

        // Use unstable sort for better performance
        sorted_data.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());

        let index = q * (sorted_data.len() - 1) as f64;
        let lower = index.floor() as usize;
        let upper = index.ceil() as usize;

        if lower == upper {
            sorted_data[lower]
        } else {
            let weight = index - lower as f64;
            sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
        }
    }
}

/// Optimized matrix operations using SIMD
pub struct SimdMatrixOps;

impl SimdMatrixOps {
    /// Optimized matrix-vector multiplication using SIMD
    pub fn matrix_vector_multiply_simd(
        matrix: &Array2<f64>,
        vector: &Array1<f64>,
    ) -> SklResult<Array1<f64>> {
        let (n_rows, n_cols) = matrix.dim();

        if n_cols != vector.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix columns {} must match vector length {}",
                n_cols,
                vector.len()
            )));
        }

        let mut result = Array1::zeros(n_rows);
        let vector_slice = vector.as_slice().unwrap();

        // Parallel processing over rows
        for i in 0..n_rows {
            let row = matrix.row(i);
            result[i] = Self::dot_product_simd(row.as_slice().unwrap(), vector_slice);
        }

        Ok(result)
    }

    /// Optimized dot product using SIMD
    pub fn dot_product_simd(x: &[f64], y: &[f64]) -> f64 {
        assert_eq!(x.len(), y.len(), "Vectors must have the same length");

        if x.len() < 4 {
            return x.iter().zip(y.iter()).map(|(&a, &b)| a * b).sum();
        }

        unsafe { Self::dot_product_simd_unsafe(x, y) }
    }

    /// Unsafe SIMD implementation for dot product
    unsafe fn dot_product_simd_unsafe(x: &[f64], y: &[f64]) -> f64 {
        let len = x.len();
        let chunks = len / 4;

        let mut sum = f64x4::splat(0.0);

        // Process 8 elements at a time
        for i in 0..chunks {
            let x_chunk = f64x4::new([x[i * 4], x[i * 4 + 1], x[i * 4 + 2], x[i * 4 + 3]]);
            let y_chunk = f64x4::new([y[i * 4], y[i * 4 + 1], y[i * 4 + 2], y[i * 4 + 3]]);
            sum += x_chunk * y_chunk;
        }

        // Sum the SIMD lanes
        let sum_array = sum.to_array();
        let mut result = sum_array[0] + sum_array[1] + sum_array[2] + sum_array[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            result += x[i] * y[i];
        }

        result
    }

    /// Optimized matrix transpose with cache-friendly access
    pub fn transpose_simd(matrix: &Array2<f64>) -> Array2<f64> {
        let (n_rows, n_cols) = matrix.dim();
        let mut result = Array2::zeros((n_cols, n_rows));

        // Use cache-friendly block transpose for better performance
        const BLOCK_SIZE: usize = 64;

        for i_block in (0..n_rows).step_by(BLOCK_SIZE) {
            for j_block in (0..n_cols).step_by(BLOCK_SIZE) {
                let i_end = (i_block + BLOCK_SIZE).min(n_rows);
                let j_end = (j_block + BLOCK_SIZE).min(n_cols);

                for i in i_block..i_end {
                    for j in j_block..j_end {
                        result[[j, i]] = matrix[[i, j]];
                    }
                }
            }
        }

        result
    }

    /// Optimized matrix-matrix multiplication using SIMD and blocking
    pub fn matrix_multiply_simd(a: &Array2<f64>, b: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (a_rows, a_cols) = a.dim();
        let (b_rows, b_cols) = b.dim();

        if a_cols != b_rows {
            return Err(SklearsError::InvalidInput(format!(
                "Matrix dimensions incompatible: {}x{} * {}x{}",
                a_rows, a_cols, b_rows, b_cols
            )));
        }

        let mut result = Array2::zeros((a_rows, b_cols));

        // Use cache-friendly blocked multiplication
        const BLOCK_SIZE: usize = 64;

        for i_block in (0..a_rows).step_by(BLOCK_SIZE) {
            for j_block in (0..b_cols).step_by(BLOCK_SIZE) {
                for k_block in (0..a_cols).step_by(BLOCK_SIZE) {
                    let i_end = (i_block + BLOCK_SIZE).min(a_rows);
                    let j_end = (j_block + BLOCK_SIZE).min(b_cols);
                    let k_end = (k_block + BLOCK_SIZE).min(a_cols);

                    for i in i_block..i_end {
                        for j in j_block..j_end {
                            let mut sum = 0.0;

                            // Use SIMD for inner product if block is large enough
                            let row = a.row(i);
                            let row_slice = row.as_slice().unwrap();
                            let k_slice = &row_slice[k_block..k_end];
                            let b_slice: Vec<f64> = (k_block..k_end).map(|k| b[[k, j]]).collect();

                            if k_slice.len() >= 4 {
                                unsafe {
                                    sum += Self::dot_product_simd_unsafe(k_slice, &b_slice);
                                }
                            } else {
                                for k in k_block..k_end {
                                    sum += a[[i, k]] * b[[k, j]];
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
}

/// Optimized K-means clustering
pub struct SimdKMeans;

impl SimdKMeans {
    /// Optimized centroid calculation
    pub fn calculate_centroids_simd(data: &Array2<f64>, labels: &[usize], k: usize) -> Array2<f64> {
        let (_n_samples, n_features) = data.dim();
        let mut centroids = Array2::zeros((k, n_features));
        let mut counts = vec![0; k];

        // Count points in each cluster
        for &label in labels {
            counts[label] += 1;
        }

        // Calculate centroids using parallel processing
        centroids
            .axis_iter_mut(Axis(0))
            .enumerate()
            .par_bridge()
            .for_each(|(cluster_idx, mut centroid)| {
                let mut sums = vec![0.0; n_features];

                for (sample_idx, &label) in labels.iter().enumerate() {
                    if label == cluster_idx {
                        let sample = data.row(sample_idx);
                        for (i, &val) in sample.iter().enumerate() {
                            sums[i] += val;
                        }
                    }
                }

                // Divide by count to get centroid
                if counts[cluster_idx] > 0 {
                    let count = counts[cluster_idx] as f64;
                    for (i, &sum) in sums.iter().enumerate() {
                        centroid[i] = sum / count;
                    }
                }
            });

        centroids
    }
}

/// Enhanced imputation operations with SIMD optimizations
pub struct SimdImputationOps;

impl SimdImputationOps {
    /// Optimized weighted mean calculation for KNN imputation using SIMD
    pub fn weighted_mean_simd(values: &[f64], weights: &[f64]) -> f64 {
        assert_eq!(
            values.len(),
            weights.len(),
            "Values and weights must have same length"
        );

        if values.is_empty() {
            return 0.0;
        }

        if values.len() < 8 {
            let weighted_sum: f64 = values
                .iter()
                .zip(weights.iter())
                .map(|(&v, &w)| v * w)
                .sum();
            let weight_sum: f64 = weights.iter().sum();

            return if weight_sum > 0.0 {
                weighted_sum / weight_sum
            } else {
                SimdStatistics::mean_simd(values)
            };
        }

        unsafe { Self::weighted_mean_simd_unsafe(values, weights) }
    }

    /// Unsafe SIMD implementation for weighted mean
    unsafe fn weighted_mean_simd_unsafe(values: &[f64], weights: &[f64]) -> f64 {
        let len = values.len();
        let chunks = len / 4;

        let mut weighted_sum = f64x4::splat(0.0);
        let mut weight_sum = f64x4::splat(0.0);

        // Process 8 elements at a time
        for i in 0..chunks {
            let values_chunk = f64x4::new([
                values[i * 4],
                values[i * 4 + 1],
                values[i * 4 + 2],
                values[i * 4 + 3],
            ]);
            let weights_chunk = f64x4::new([
                weights[i * 4],
                weights[i * 4 + 1],
                weights[i * 4 + 2],
                weights[i * 4 + 3],
            ]);

            weighted_sum += values_chunk * weights_chunk;
            weight_sum += weights_chunk;
        }

        // Sum the SIMD lanes
        let weighted_array = weighted_sum.to_array();
        let weight_array = weight_sum.to_array();
        let mut weighted_result =
            weighted_array[0] + weighted_array[1] + weighted_array[2] + weighted_array[3];
        let mut weight_result =
            weight_array[0] + weight_array[1] + weight_array[2] + weight_array[3];

        // Handle remaining elements
        for i in (chunks * 4)..len {
            weighted_result += values[i] * weights[i];
            weight_result += weights[i];
        }

        if weight_result > 0.0 {
            weighted_result / weight_result
        } else {
            SimdStatistics::mean_simd(values)
        }
    }

    /// Optimized missing value detection using SIMD
    pub fn count_missing_simd(data: &[f64]) -> usize {
        if data.len() < 4 {
            return data.iter().filter(|&&x| x.is_nan()).count();
        }

        unsafe { Self::count_missing_simd_unsafe(data) }
    }

    /// Unsafe SIMD implementation for missing value counting
    unsafe fn count_missing_simd_unsafe(data: &[f64]) -> usize {
        let len = data.len();
        let chunks = len / 4;

        let mut missing_count = 0;

        // Process 4 elements at a time (manual check since f64x4 doesn't have is_nan)
        for i in 0..chunks {
            let base_idx = i * 4;
            for j in 0..4 {
                if data[base_idx + j].is_nan() {
                    missing_count += 1;
                }
            }
        }

        // Handle remaining elements
        missing_count += data
            .iter()
            .skip(chunks * 4)
            .take(len - chunks * 4)
            .filter(|x| x.is_nan())
            .count();

        missing_count
    }

    /// Optimized batch distance calculation for KNN
    pub fn batch_distances_simd(
        query_point: &[f64],
        data_points: &Array2<f64>,
        metric: &str,
    ) -> Vec<f64> {
        let n_points = data_points.nrows();
        let mut distances = Vec::with_capacity(n_points);

        match metric {
            "euclidean" => {
                distances.par_extend((0..n_points).into_par_iter().map(|i| {
                    let row = data_points.row(i);
                    let point = row.as_slice().unwrap();
                    SimdDistanceCalculator::euclidean_distance_simd(query_point, point)
                }));
            }
            "manhattan" => {
                distances.par_extend((0..n_points).into_par_iter().map(|i| {
                    let row = data_points.row(i);
                    let point = row.as_slice().unwrap();
                    SimdDistanceCalculator::manhattan_distance_simd(query_point, point)
                }));
            }
            "cosine" => {
                distances.par_extend((0..n_points).into_par_iter().map(|i| {
                    let row = data_points.row(i);
                    let point = row.as_slice().unwrap();
                    1.0 - SimdDistanceCalculator::cosine_similarity_simd(query_point, point)
                }));
            }
            "nan_euclidean" => {
                distances.par_extend((0..n_points).into_par_iter().map(|i| {
                    let row = data_points.row(i);
                    let point = row.as_slice().unwrap();
                    SimdDistanceCalculator::nan_euclidean_distance_simd(query_point, point)
                }));
            }
            _ => {
                // Fallback to euclidean
                distances.par_extend((0..n_points).into_par_iter().map(|i| {
                    let row = data_points.row(i);
                    let point = row.as_slice().unwrap();
                    SimdDistanceCalculator::euclidean_distance_simd(query_point, point)
                }));
            }
        }

        distances
    }

    /// Optimized k-nearest neighbors finding
    pub fn find_knn_simd(
        query_point: &[f64],
        data_points: &Array2<f64>,
        k: usize,
        metric: &str,
    ) -> Vec<(usize, f64)> {
        let distances = Self::batch_distances_simd(query_point, data_points, metric);

        let mut indexed_distances: Vec<(usize, f64)> = distances
            .into_iter()
            .enumerate()
            .filter(|(_, dist)| dist.is_finite())
            .collect();

        // Use partial sort for better performance when k << n
        if k < indexed_distances.len() {
            indexed_distances.select_nth_unstable_by(k, |a, b| a.1.partial_cmp(&b.1).unwrap());
            indexed_distances.truncate(k);
        }

        indexed_distances.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        indexed_distances
    }

    /// Memory-efficient imputation for large datasets
    pub fn streaming_mean_imputation(data: &mut Array2<f64>, chunk_size: usize, missing_val: f64) {
        let (n_rows, n_cols) = data.dim();

        // Calculate column means in chunks to save memory
        let mut column_means = vec![0.0; n_cols];
        let mut column_counts = vec![0; n_cols];

        for row_chunk in (0..n_rows).step_by(chunk_size) {
            let end_row = (row_chunk + chunk_size).min(n_rows);

            for i in row_chunk..end_row {
                for j in 0..n_cols {
                    let val = data[[i, j]];
                    let is_missing = if missing_val.is_nan() {
                        val.is_nan()
                    } else {
                        (val - missing_val).abs() < f64::EPSILON
                    };

                    if !is_missing {
                        column_means[j] += val;
                        column_counts[j] += 1;
                    }
                }
            }
        }

        // Finalize means
        for j in 0..n_cols {
            if column_counts[j] > 0 {
                column_means[j] /= column_counts[j] as f64;
            }
        }

        // Apply imputation in chunks
        for row_chunk in (0..n_rows).step_by(chunk_size) {
            let end_row = (row_chunk + chunk_size).min(n_rows);

            for i in row_chunk..end_row {
                for j in 0..n_cols {
                    let val = data[[i, j]];
                    let is_missing = if missing_val.is_nan() {
                        val.is_nan()
                    } else {
                        (val - missing_val).abs() < f64::EPSILON
                    };

                    if is_missing && column_counts[j] > 0 {
                        data[[i, j]] = column_means[j];
                    }
                }
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_euclidean_distance() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let distance = SimdDistanceCalculator::euclidean_distance_simd(&x, &y);
        let expected = 3.0; // sqrt(9 * 1^2) = 3.0

        assert_abs_diff_eq!(distance, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_mean() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let mean = SimdStatistics::mean_simd(&data);
        let expected = 5.5;

        assert_abs_diff_eq!(mean, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_dot_product() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];

        let dot_product = SimdMatrixOps::dot_product_simd(&x, &y);
        let expected = 240.0; // 1*2 + 2*3 + ... + 8*9

        assert_abs_diff_eq!(dot_product, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_weighted_mean() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        let weighted_mean = SimdImputationOps::weighted_mean_simd(&values, &weights);
        let expected = 4.5; // Simple mean when all weights are equal

        assert_abs_diff_eq!(weighted_mean, expected, epsilon = 1e-10);
    }

    #[test]
    fn test_cache_optimized_data() {
        let data = Array2::from_shape_vec(
            (3, 4),
            vec![
                1.0,
                2.0,
                f64::NAN,
                4.0,
                5.0,
                f64::NAN,
                7.0,
                8.0,
                9.0,
                10.0,
                11.0,
                f64::NAN,
            ],
        )
        .unwrap();

        let optimized = CacheOptimizedData::new(&data, f64::NAN);

        // Test value access
        assert_eq!(optimized.get(0, 0), Some(1.0));
        assert_eq!(optimized.get(0, 1), Some(2.0));
        assert_eq!(optimized.get(1, 0), Some(5.0));

        // Test missing value detection
        assert!(optimized.is_missing(0, 2));
        assert!(optimized.is_missing(1, 1));
        assert!(optimized.is_missing(2, 3));
        assert!(!optimized.is_missing(0, 0));
        assert!(!optimized.is_missing(1, 0));
    }

    #[test]
    fn test_simd_distance_calculations() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = vec![2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0];

        // Test Manhattan distance
        let manhattan = SimdDistanceCalculator::manhattan_distance_simd(&x, &y);
        assert_abs_diff_eq!(manhattan, 10.0, epsilon = 1e-10);

        // Test cosine similarity
        let cosine_sim = SimdDistanceCalculator::cosine_similarity_simd(&x, &y);
        assert!(cosine_sim > 0.9); // Should be very similar vectors
    }

    #[test]
    fn test_simd_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        // Test variance
        let variance = SimdStatistics::variance_simd(&data, None);
        let expected_variance = 9.166666666666666; // Variance of 1..10
        assert_abs_diff_eq!(variance, expected_variance, epsilon = 1e-10);

        // Test min/max
        let (min_val, max_val) = SimdStatistics::min_max_simd(&data);
        assert_eq!(min_val, 1.0);
        assert_eq!(max_val, 10.0);

        // Test quantile
        let median = SimdStatistics::quantile_simd(&data, 0.5);
        assert_abs_diff_eq!(median, 5.5, epsilon = 1e-10);
    }

    #[test]
    fn test_matrix_operations() {
        let matrix =
            Array2::from_shape_vec((3, 3), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])
                .unwrap();

        // Test transpose
        let transposed = SimdMatrixOps::transpose_simd(&matrix);
        assert_eq!(transposed[[0, 0]], 1.0);
        assert_eq!(transposed[[0, 1]], 4.0);
        assert_eq!(transposed[[0, 2]], 7.0);
        assert_eq!(transposed[[1, 0]], 2.0);

        // Test matrix-vector multiplication
        let vector = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let result = SimdMatrixOps::matrix_vector_multiply_simd(&matrix, &vector).unwrap();

        // Expected: [1*1 + 2*2 + 3*3, 4*1 + 5*2 + 6*3, 7*1 + 8*2 + 9*3] = [14, 32, 50]
        assert_abs_diff_eq!(result[0], 14.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 32.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 50.0, epsilon = 1e-10);
    }

    #[test]
    fn test_batch_distances() {
        let query = vec![1.0, 2.0, 3.0];
        let data = Array2::from_shape_vec(
            (3, 3),
            vec![
                1.0, 2.0, 3.0, // Distance 0
                2.0, 3.0, 4.0, // Distance sqrt(3)
                4.0, 5.0, 6.0, // Distance sqrt(27)
            ],
        )
        .unwrap();

        let distances = SimdImputationOps::batch_distances_simd(&query, &data, "euclidean");

        assert_eq!(distances.len(), 3);
        assert_abs_diff_eq!(distances[0], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[1], 3.0_f64.sqrt(), epsilon = 1e-10);
        assert_abs_diff_eq!(distances[2], 27.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_knn_finding() {
        let query = vec![1.0, 2.0, 3.0];
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, // Distance 0 (closest)
                2.0, 3.0, 4.0, // Distance sqrt(3)
                4.0, 5.0, 6.0, // Distance sqrt(27)
                0.5, 1.5, 2.5, // Distance sqrt(0.75)
                10.0, 11.0, 12.0, // Distance sqrt(243) (farthest)
            ],
        )
        .unwrap();

        let knn = SimdImputationOps::find_knn_simd(&query, &data, 3, "euclidean");

        assert_eq!(knn.len(), 3);
        assert_eq!(knn[0].0, 0); // Closest is the identical point
        assert_eq!(knn[1].0, 3); // Second closest
        assert_eq!(knn[2].0, 1); // Third closest
    }

    #[test]
    fn test_missing_count() {
        let data = vec![
            1.0,
            f64::NAN,
            3.0,
            f64::NAN,
            5.0,
            6.0,
            f64::NAN,
            8.0,
            9.0,
            f64::NAN,
        ];
        let count = SimdImputationOps::count_missing_simd(&data);
        assert_eq!(count, 4);
    }

    #[test]
    fn test_streaming_imputation() {
        let mut data = Array2::from_shape_vec(
            (4, 3),
            vec![
                1.0,
                f64::NAN,
                3.0,
                4.0,
                5.0,
                f64::NAN,
                f64::NAN,
                8.0,
                9.0,
                10.0,
                11.0,
                12.0,
            ],
        )
        .unwrap();

        SimdImputationOps::streaming_mean_imputation(&mut data, 2, f64::NAN);

        // Check that missing values were replaced with column means
        // Column 0 mean: (1.0 + 4.0 + 10.0) / 3 = 5.0
        // Column 1 mean: (5.0 + 8.0 + 11.0) / 3 = 8.0
        // Column 2 mean: (3.0 + 9.0 + 12.0) / 3 = 8.0

        assert_abs_diff_eq!(data[[0, 1]], 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data[[1, 2]], 8.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data[[2, 0]], 5.0, epsilon = 1e-10);

        // Non-missing values should remain unchanged
        assert_abs_diff_eq!(data[[0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data[[1, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data[[3, 2]], 12.0, epsilon = 1e-10);
    }
}
