//! Shared utilities and helper functions for feature engineering
//!
//! This module provides common utility functions used across the engineering
//! modules, including data validation, array operations, mathematical helpers,
//! and performance utilities.

use crate::*;
use scirs2_core::ndarray::{Array, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::prelude::SklearsError;
use sklears_core::types::Float;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

/// Validates input data arrays for feature engineering operations
///
/// This function performs comprehensive validation of input data to ensure
/// it meets the requirements for feature engineering algorithms.
pub fn validate_input_data(data: &ArrayView2<Float>) -> Result<(), SklearsError> {
    if data.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Input data cannot be empty".to_string(),
        ));
    }

    if data.nrows() == 0 {
        return Err(SklearsError::InvalidInput(
            "Input data must have at least one sample".to_string(),
        ));
    }

    if data.ncols() == 0 {
        return Err(SklearsError::InvalidInput(
            "Input data must have at least one feature".to_string(),
        ));
    }

    // Check for non-finite values
    let finite_count = data.iter().filter(|&&x| x.is_finite()).count();
    let total_count = data.len();

    if finite_count == 0 {
        return Err(SklearsError::InvalidInput(
            "Input data contains no finite values".to_string(),
        ));
    }

    let non_finite_ratio = (total_count - finite_count) as Float / total_count as Float;
    if non_finite_ratio > 0.5 {
        return Err(SklearsError::InvalidInput(format!(
            "Input data contains too many non-finite values ({:.1}%)",
            non_finite_ratio * 100.0
        )));
    }

    Ok(())
}

/// Validates feature names for consistency and uniqueness
pub fn validate_feature_names(names: &[String]) -> Result<(), SklearsError> {
    if names.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Feature names cannot be empty".to_string(),
        ));
    }

    // Check for duplicates
    let unique_names: HashSet<_> = names.iter().collect();
    if unique_names.len() != names.len() {
        return Err(SklearsError::InvalidInput(
            "Feature names must be unique".to_string(),
        ));
    }

    // Check for empty or whitespace-only names
    for name in names {
        if name.trim().is_empty() {
            return Err(SklearsError::InvalidInput(
                "Feature names cannot be empty or whitespace-only".to_string(),
            ));
        }
    }

    Ok(())
}

/// Generates automatic feature names with optional prefix
pub fn generate_feature_names(n_features: usize, prefix: Option<&str>) -> Vec<String> {
    let prefix = prefix.unwrap_or("feature");
    (0..n_features)
        .map(|i| format!("{}_{}", prefix, i))
        .collect()
}

/// Checks if an array contains only finite values
pub fn is_finite_array(data: &ArrayView1<Float>) -> bool {
    data.iter().all(|&x| x.is_finite())
}

/// Checks if an array contains any NaN values
pub fn has_nan_values(data: &ArrayView1<Float>) -> bool {
    data.iter().any(|&x| x.is_nan())
}

/// Checks if an array contains any infinite values
pub fn has_infinite_values(data: &ArrayView1<Float>) -> bool {
    data.iter().any(|&x| x.is_infinite())
}

/// Computes safe statistics for a column, handling non-finite values
pub fn safe_column_statistics(column: &ArrayView1<Float>) -> (Float, Float, Float, Float, usize) {
    let finite_values: Vec<Float> = column.iter().copied().filter(|x| x.is_finite()).collect();

    if finite_values.is_empty() {
        return (0.0, 0.0, 0.0, 0.0, 0);
    }

    let n = finite_values.len();
    let mean = finite_values.iter().sum::<Float>() / n as Float;

    let variance = finite_values
        .iter()
        .map(|&x| (x - mean).powi(2))
        .sum::<Float>() / n as Float;
    let std_dev = variance.sqrt();

    let min_val = finite_values
        .iter()
        .copied()
        .fold(Float::INFINITY, Float::min);
    let max_val = finite_values
        .iter()
        .copied()
        .fold(Float::NEG_INFINITY, Float::max);

    (mean, std_dev, min_val, max_val, n)
}

/// Normalizes a vector to unit length (L2 norm)
pub fn normalize_vector(mut vector: Array1<Float>) -> Array1<Float> {
    let norm = (vector.iter().map(|&x| x * x).sum::<Float>()).sqrt();
    if norm > Float::EPSILON {
        vector.mapv_inplace(|x| x / norm);
    }
    vector
}

/// Standardizes a vector to zero mean and unit variance
pub fn standardize_vector(mut vector: Array1<Float>) -> Array1<Float> {
    let (mean, std, _, _, _) = safe_column_statistics(&vector.view());
    if std > Float::EPSILON {
        vector.mapv_inplace(|x| (x - mean) / std);
    }
    vector
}

/// Clips values in an array to a specified range
pub fn clip_array(mut array: Array1<Float>, min_val: Float, max_val: Float) -> Array1<Float> {
    array.mapv_inplace(|x| x.max(min_val).min(max_val));
    array
}

/// Computes the percentile of a sorted array
pub fn percentile(sorted_data: &[Float], percentile: Float) -> Float {
    if sorted_data.is_empty() {
        return 0.0;
    }

    let index = (percentile / 100.0) * (sorted_data.len() - 1) as Float;
    let lower_index = index.floor() as usize;
    let upper_index = index.ceil() as usize;

    if lower_index == upper_index {
        sorted_data[lower_index]
    } else {
        let weight = index - lower_index as Float;
        sorted_data[lower_index] * (1.0 - weight) + sorted_data[upper_index] * weight
    }
}

/// Computes robust statistics (median, median absolute deviation)
pub fn robust_statistics(data: &ArrayView1<Float>) -> (Float, Float) {
    let mut finite_values: Vec<Float> = data.iter().copied().filter(|x| x.is_finite()).collect();

    if finite_values.is_empty() {
        return (0.0, 0.0);
    }

    finite_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = percentile(&finite_values, 50.0);

    // Median Absolute Deviation
    let deviations: Vec<Float> = finite_values.iter().map(|&x| (x - median).abs()).collect();
    let mut sorted_deviations = deviations;
    sorted_deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mad = percentile(&sorted_deviations, 50.0);

    (median, mad)
}

/// Generates a hash for deterministic randomization
pub fn hash_feature_name(name: &str, seed: u64) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    let mut hasher = DefaultHasher::new();
    name.hash(&mut hasher);
    seed.hash(&mut hasher);
    hasher.finish()
}

/// Creates evenly spaced values in a range
pub fn linspace(start: Float, end: Float, num_points: usize) -> Array1<Float> {
    if num_points == 0 {
        return Array1::zeros(0);
    }

    if num_points == 1 {
        return Array1::from_vec(vec![start]);
    }

    let step = (end - start) / (num_points - 1) as Float;
    Array1::from_shape_fn(num_points, |i| start + i as Float * step)
}

/// Creates logarithmically spaced values
pub fn logspace(start: Float, end: Float, num_points: usize, base: Float) -> Array1<Float> {
    let linear_space = linspace(start, end, num_points);
    linear_space.mapv(|x| base.powf(x))
}

/// Computes the cumulative sum of an array
pub fn cumsum(data: &ArrayView1<Float>) -> Array1<Float> {
    let mut result = Array1::zeros(data.len());
    let mut sum = 0.0;

    for (i, &value) in data.iter().enumerate() {
        sum += value;
        result[i] = sum;
    }

    result
}

/// Computes moving window statistics
pub fn moving_window_stats(
    data: &ArrayView1<Float>,
    window_size: usize,
) -> (Array1<Float>, Array1<Float>) {
    let n = data.len();
    if window_size > n || window_size == 0 {
        return (Array1::zeros(0), Array1::zeros(0));
    }

    let output_size = n - window_size + 1;
    let mut means = Array1::zeros(output_size);
    let mut stds = Array1::zeros(output_size);

    for i in 0..output_size {
        let window = data.slice(s![i..i + window_size]);
        let (mean, std, _, _, _) = safe_column_statistics(&window);
        means[i] = mean;
        stds[i] = std;
    }

    (means, stds)
}

/// Performance timer for benchmarking operations
pub struct PerformanceTimer {
    start_time: Instant,
    operation_name: String,
}

impl PerformanceTimer {
    /// Start timing an operation
    pub fn start(operation_name: String) -> Self {
        Self {
            start_time: Instant::now(),
            operation_name,
        }
    }

    /// Get elapsed time in milliseconds
    pub fn elapsed_ms(&self) -> f64 {
        self.start_time.elapsed().as_secs_f64() * 1000.0
    }

    /// Stop timing and log the result
    pub fn stop(self) -> f64 {
        let elapsed = self.elapsed_ms();
        println!("Operation '{}' completed in {:.2} ms", self.operation_name, elapsed);
        elapsed
    }
}

/// Memory usage estimator for arrays
pub fn estimate_memory_usage(shape: &[usize]) -> usize {
    let elements = shape.iter().product::<usize>();
    elements * std::mem::size_of::<Float>()
}

/// Checks if an operation would exceed memory limits
pub fn check_memory_limit(
    operation_memory: usize,
    memory_limit: Option<usize>,
) -> Result<(), SklearsError> {
    if let Some(limit) = memory_limit {
        if operation_memory > limit {
            return Err(SklearsError::InvalidInput(format!(
                "Operation would require {} bytes, exceeding limit of {} bytes",
                operation_memory, limit
            )));
        }
    }
    Ok(())
}

/// Splits data into chunks for batch processing
pub fn create_chunks(data_size: usize, chunk_size: usize) -> Vec<(usize, usize)> {
    let mut chunks = Vec::new();
    let mut start = 0;

    while start < data_size {
        let end = (start + chunk_size).min(data_size);
        chunks.push((start, end));
        start = end;
    }

    chunks
}

/// Safe division that handles division by zero
pub fn safe_divide(numerator: Float, denominator: Float) -> Float {
    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

/// Safe logarithm that handles zero and negative values
pub fn safe_log(value: Float) -> Float {
    if value <= 0.0 {
        Float::NEG_INFINITY
    } else {
        value.ln()
    }
}

/// Safe square root that handles negative values
pub fn safe_sqrt(value: Float) -> Float {
    if value < 0.0 {
        0.0
    } else {
        value.sqrt()
    }
}

/// Converts degrees to radians
pub fn degrees_to_radians(degrees: Float) -> Float {
    degrees * std::f64::consts::PI as Float / 180.0
}

/// Converts radians to degrees
pub fn radians_to_degrees(radians: Float) -> Float {
    radians * 180.0 / std::f64::consts::PI as Float
}

/// Calculates the Euclidean distance between two points
pub fn euclidean_distance(p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).powi(2))
        .sum::<Float>()
        .sqrt()
}

/// Calculates the Manhattan distance between two points
pub fn manhattan_distance(p1: &ArrayView1<Float>, p2: &ArrayView1<Float>) -> Float {
    p1.iter()
        .zip(p2.iter())
        .map(|(a, b)| (a - b).abs())
        .sum::<Float>()
}

/// Calculates the cosine similarity between two vectors
pub fn cosine_similarity(v1: &ArrayView1<Float>, v2: &ArrayView1<Float>) -> Float {
    let dot_product = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum::<Float>();
    let norm1 = v1.iter().map(|&x| x.powi(2)).sum::<Float>().sqrt();
    let norm2 = v2.iter().map(|&x| x.powi(2)).sum::<Float>().sqrt();

    safe_divide(dot_product, norm1 * norm2)
}

/// Creates a feature name mapping for transformed features
pub fn create_feature_mapping(
    original_names: &[String],
    transform_name: &str,
    n_output_features: usize,
) -> HashMap<String, String> {
    let mut mapping = HashMap::new();

    for (i, output_name) in generate_feature_names(n_output_features, Some(transform_name))
        .iter()
        .enumerate()
    {
        let input_idx = i % original_names.len();
        let original_name = &original_names[input_idx];
        mapping.insert(output_name.clone(), original_name.clone());
    }

    mapping
}

/// Validates that array dimensions are compatible for operations
pub fn validate_compatible_dimensions(
    arr1_shape: &[usize],
    arr2_shape: &[usize],
    operation: &str,
) -> Result<(), SklearsError> {
    match operation {
        "element_wise" => {
            if arr1_shape != arr2_shape {
                return Err(SklearsError::InvalidInput(format!(
                    "Arrays must have same shape for element-wise operations: {:?} vs {:?}",
                    arr1_shape, arr2_shape
                )));
            }
        }
        "matrix_multiply" => {
            if arr1_shape.len() != 2 || arr2_shape.len() != 2 {
                return Err(SklearsError::InvalidInput(
                    "Both arrays must be 2D for matrix multiplication".to_string(),
                ));
            }
            if arr1_shape[1] != arr2_shape[0] {
                return Err(SklearsError::InvalidInput(format!(
                    "Incompatible dimensions for matrix multiplication: ({}, {}) x ({}, {})",
                    arr1_shape[0], arr1_shape[1], arr2_shape[0], arr2_shape[1]
                )));
            }
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown operation for dimension validation: {}",
                operation
            )));
        }
    }

    Ok(())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_validate_input_data() {
        let valid_data = array![[1.0, 2.0], [3.0, 4.0]];
        assert!(validate_input_data(&valid_data.view()).is_ok());

        let empty_data = Array2::zeros((0, 0));
        assert!(validate_input_data(&empty_data.view()).is_err());
    }

    #[test]
    fn test_safe_column_statistics() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let (mean, std, min_val, max_val, count) = safe_column_statistics(&data.view());

        assert!((mean - 3.0).abs() < 1e-10);
        assert!(count == 5);
        assert!((min_val - 1.0).abs() < 1e-10);
        assert!((max_val - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_normalize_vector() {
        let data = array![3.0, 4.0];
        let normalized = normalize_vector(data);
        let norm = normalized.iter().map(|&x| x * x).sum::<Float>().sqrt();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_percentile() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((percentile(&data, 50.0) - 3.0).abs() < 1e-10);
        assert!((percentile(&data, 0.0) - 1.0).abs() < 1e-10);
        assert!((percentile(&data, 100.0) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_linspace() {
        let result = linspace(0.0, 10.0, 11);
        assert_eq!(result.len(), 11);
        assert!((result[0] - 0.0).abs() < 1e-10);
        assert!((result[10] - 10.0).abs() < 1e-10);
        assert!((result[5] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_safe_math_functions() {
        assert_eq!(safe_divide(10.0, 0.0), 0.0);
        assert_eq!(safe_sqrt(-1.0), 0.0);
        assert!(safe_log(0.0).is_infinite());
    }

    #[test]
    fn test_distance_functions() {
        let p1 = array![0.0, 0.0];
        let p2 = array![3.0, 4.0];

        let euclidean = euclidean_distance(&p1.view(), &p2.view());
        assert!((euclidean - 5.0).abs() < 1e-10);

        let manhattan = manhattan_distance(&p1.view(), &p2.view());
        assert!((manhattan - 7.0).abs() < 1e-10);
    }

    #[test]
    fn test_feature_name_generation() {
        let names = generate_feature_names(3, Some("test"));
        assert_eq!(names, vec!["test_0", "test_1", "test_2"]);

        let default_names = generate_feature_names(2, None);
        assert_eq!(default_names, vec!["feature_0", "feature_1"]);
    }

    #[test]
    fn test_validate_feature_names() {
        let valid_names = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        assert!(validate_feature_names(&valid_names).is_ok());

        let duplicate_names = vec!["a".to_string(), "b".to_string(), "a".to_string()];
        assert!(validate_feature_names(&duplicate_names).is_err());

        let empty_names: Vec<String> = vec![];
        assert!(validate_feature_names(&empty_names).is_err());
    }

    #[test]
    fn test_performance_timer() {
        let timer = PerformanceTimer::start("test_operation".to_string());
        std::thread::sleep(std::time::Duration::from_millis(10));
        let elapsed = timer.elapsed_ms();
        assert!(elapsed >= 10.0);
    }

    #[test]
    fn test_create_chunks() {
        let chunks = create_chunks(10, 3);
        assert_eq!(chunks, vec![(0, 3), (3, 6), (6, 9), (9, 10)]);

        let single_chunk = create_chunks(5, 10);
        assert_eq!(single_chunk, vec![(0, 5)]);
    }
}