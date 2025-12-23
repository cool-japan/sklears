//! Data preprocessing utilities for machine learning
//!
//! This module provides utilities for data cleaning, outlier detection,
//! data transformation, and quality assessment.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, Axis};
use scirs2_core::numeric::Float;
use std::collections::HashMap;

/// Data cleaning utilities
pub struct DataCleaner;

impl DataCleaner {
    /// Remove rows with missing values (NaN)
    pub fn drop_missing_rows<T>(data: &Array2<T>) -> UtilsResult<Array2<T>>
    where
        T: Float + Clone + std::iter::Sum,
    {
        let mut valid_rows = Vec::new();

        for (row_idx, row) in data.axis_iter(Axis(0)).enumerate() {
            if !row.iter().any(|&x| x.is_nan()) {
                valid_rows.push(row_idx);
            }
        }

        if valid_rows.is_empty() {
            return Err(UtilsError::EmptyInput);
        }

        let mut result = Array2::zeros((valid_rows.len(), data.ncols()));
        for (new_idx, &old_idx) in valid_rows.iter().enumerate() {
            result.row_mut(new_idx).assign(&data.row(old_idx));
        }

        Ok(result)
    }

    /// Fill missing values with specified value
    pub fn fill_missing<T>(data: &mut Array2<T>, fill_value: T)
    where
        T: Float + Clone + std::iter::Sum,
    {
        data.mapv_inplace(|x| if x.is_nan() { fill_value } else { x });
    }

    /// Fill missing values with column means
    pub fn fill_with_mean<T>(data: &mut Array2<T>) -> UtilsResult<()>
    where
        T: Float + Clone + std::iter::Sum,
    {
        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let valid_values: Vec<T> = col.iter().cloned().filter(|x| !x.is_nan()).collect();

            if !valid_values.is_empty() {
                let mean =
                    valid_values.iter().cloned().sum::<T>() / T::from(valid_values.len()).unwrap();

                for row_idx in 0..data.nrows() {
                    if data[[row_idx, col_idx]].is_nan() {
                        data[[row_idx, col_idx]] = mean;
                    }
                }
            }
        }
        Ok(())
    }

    /// Fill missing values with column medians
    pub fn fill_with_median<T>(data: &mut Array2<T>) -> UtilsResult<()>
    where
        T: Float + Clone + PartialOrd,
    {
        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let mut valid_values: Vec<T> = col.iter().cloned().filter(|x| !x.is_nan()).collect();

            if !valid_values.is_empty() {
                valid_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median = if valid_values.len() % 2 == 0 {
                    let mid = valid_values.len() / 2;
                    (valid_values[mid - 1] + valid_values[mid]) / T::from(2).unwrap()
                } else {
                    valid_values[valid_values.len() / 2]
                };

                for row_idx in 0..data.nrows() {
                    if data[[row_idx, col_idx]].is_nan() {
                        data[[row_idx, col_idx]] = median;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Outlier detection methods
pub struct OutlierDetector;

impl OutlierDetector {
    /// Detect outliers using Z-score method
    pub fn zscore_outliers<T>(data: &ArrayView1<T>, threshold: T) -> Vec<usize>
    where
        T: Float + Clone + std::iter::Sum,
    {
        let mean = data.iter().cloned().sum::<T>() / T::from(data.len()).unwrap();
        let variance =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<T>() / T::from(data.len()).unwrap();
        let std_dev = variance.sqrt();

        if std_dev == T::zero() {
            return Vec::new();
        }

        data.iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                let z_score = (value - mean).abs() / std_dev;
                if z_score > threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Detect outliers using IQR (Interquartile Range) method
    pub fn iqr_outliers<T>(data: &ArrayView1<T>, multiplier: T) -> Vec<usize>
    where
        T: Float + Clone + PartialOrd,
    {
        let mut sorted_data: Vec<T> = data.iter().cloned().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        if n < 4 {
            return Vec::new();
        }

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;
        let q1 = sorted_data[q1_idx];
        let q3 = sorted_data[q3_idx];
        let iqr = q3 - q1;

        let lower_bound = q1 - multiplier * iqr;
        let upper_bound = q3 + multiplier * iqr;

        data.iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                if value < lower_bound || value > upper_bound {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Detect outliers using modified Z-score method (using median)
    pub fn modified_zscore_outliers<T>(data: &ArrayView1<T>, threshold: T) -> Vec<usize>
    where
        T: Float + Clone + PartialOrd,
    {
        let mut sorted_data: Vec<T> = data.iter().cloned().collect();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = sorted_data.len();
        if n == 0 {
            return Vec::new();
        }

        let median = if n % 2 == 0 {
            (sorted_data[n / 2 - 1] + sorted_data[n / 2]) / T::from(2).unwrap()
        } else {
            sorted_data[n / 2]
        };

        // Calculate MAD (Median Absolute Deviation)
        let mut deviations: Vec<T> = data.iter().map(|&x| (x - median).abs()).collect();
        deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let mad = if deviations.len() % 2 == 0 {
            let mid = deviations.len() / 2;
            (deviations[mid - 1] + deviations[mid]) / T::from(2).unwrap()
        } else {
            deviations[deviations.len() / 2]
        };

        if mad == T::zero() {
            return Vec::new();
        }

        let mad_scaled = mad * T::from(1.4826).unwrap(); // Scale factor for normal distribution

        data.iter()
            .enumerate()
            .filter_map(|(idx, &value)| {
                let modified_z = T::from(0.6745).unwrap() * (value - median).abs() / mad_scaled;
                if modified_z > threshold {
                    Some(idx)
                } else {
                    None
                }
            })
            .collect()
    }
}

/// Feature scaling utilities
pub struct FeatureScaler;

impl FeatureScaler {
    /// Standard scaling (z-score normalization)
    pub fn standard_scale<T>(data: &Array2<T>) -> UtilsResult<(Array2<T>, Array1<T>, Array1<T>)>
    where
        T: Float + Clone + std::iter::Sum,
    {
        let mut scaled_data = data.clone();
        let mut means = Array1::zeros(data.ncols());
        let mut stds = Array1::zeros(data.ncols());

        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let mean = col.iter().cloned().sum::<T>() / T::from(col.len()).unwrap();
            let variance =
                col.iter().map(|&x| (x - mean).powi(2)).sum::<T>() / T::from(col.len()).unwrap();
            let std_dev = variance.sqrt();

            means[col_idx] = mean;
            stds[col_idx] = std_dev;

            if std_dev != T::zero() {
                for row_idx in 0..data.nrows() {
                    scaled_data[[row_idx, col_idx]] = (data[[row_idx, col_idx]] - mean) / std_dev;
                }
            }
        }

        Ok((scaled_data, means, stds))
    }

    /// Min-max scaling to [0, 1] range
    pub fn minmax_scale<T>(data: &Array2<T>) -> UtilsResult<(Array2<T>, Array1<T>, Array1<T>)>
    where
        T: Float + Clone + PartialOrd,
    {
        let mut scaled_data = data.clone();
        let mut mins = Array1::zeros(data.ncols());
        let mut maxs = Array1::zeros(data.ncols());

        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let min_val = col
                .iter()
                .cloned()
                .fold(col[0], |acc, x| if x < acc { x } else { acc });
            let max_val = col
                .iter()
                .cloned()
                .fold(col[0], |acc, x| if x > acc { x } else { acc });

            mins[col_idx] = min_val;
            maxs[col_idx] = max_val;

            let range = max_val - min_val;
            if range != T::zero() {
                for row_idx in 0..data.nrows() {
                    scaled_data[[row_idx, col_idx]] = (data[[row_idx, col_idx]] - min_val) / range;
                }
            }
        }

        Ok((scaled_data, mins, maxs))
    }

    /// Robust scaling using median and IQR
    pub fn robust_scale<T>(data: &Array2<T>) -> UtilsResult<(Array2<T>, Array1<T>, Array1<T>)>
    where
        T: Float + Clone + PartialOrd,
    {
        let mut scaled_data = data.clone();
        let mut medians = Array1::zeros(data.ncols());
        let mut iqrs = Array1::zeros(data.ncols());

        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let mut sorted_col: Vec<T> = col.iter().cloned().collect();
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let n = sorted_col.len();
            let median = if n % 2 == 0 {
                (sorted_col[n / 2 - 1] + sorted_col[n / 2]) / T::from(2).unwrap()
            } else {
                sorted_col[n / 2]
            };

            let q1_idx = n / 4;
            let q3_idx = 3 * n / 4;
            let q1 = sorted_col[q1_idx];
            let q3 = sorted_col[q3_idx];
            let iqr = q3 - q1;

            medians[col_idx] = median;
            iqrs[col_idx] = iqr;

            if iqr != T::zero() {
                for row_idx in 0..data.nrows() {
                    scaled_data[[row_idx, col_idx]] = (data[[row_idx, col_idx]] - median) / iqr;
                }
            }
        }

        Ok((scaled_data, medians, iqrs))
    }
}

/// Data quality assessment utilities
pub struct DataQualityAssessor;

impl DataQualityAssessor {
    /// Calculate missing value statistics
    pub fn missing_value_stats<T>(data: &Array2<T>) -> HashMap<String, f64>
    where
        T: Float,
    {
        let total_cells = data.len() as f64;
        let mut missing_count = 0;
        let mut missing_per_column = Vec::new();
        let mut missing_per_row = Vec::new();

        // Count missing values per column
        for col_idx in 0..data.ncols() {
            let col_missing = data.column(col_idx).iter().filter(|&&x| x.is_nan()).count();
            missing_per_column.push(col_missing as f64 / data.nrows() as f64);
            missing_count += col_missing;
        }

        // Count missing values per row
        for row_idx in 0..data.nrows() {
            let row_missing = data.row(row_idx).iter().filter(|&&x| x.is_nan()).count();
            missing_per_row.push(row_missing as f64 / data.ncols() as f64);
        }

        let mut stats = HashMap::new();
        stats.insert(
            "total_missing_ratio".to_string(),
            missing_count as f64 / total_cells,
        );
        stats.insert(
            "max_column_missing_ratio".to_string(),
            missing_per_column.iter().cloned().fold(0.0, f64::max),
        );
        stats.insert(
            "max_row_missing_ratio".to_string(),
            missing_per_row.iter().cloned().fold(0.0, f64::max),
        );
        stats.insert(
            "columns_with_missing".to_string(),
            missing_per_column.iter().filter(|&&x| x > 0.0).count() as f64,
        );
        stats.insert(
            "rows_with_missing".to_string(),
            missing_per_row.iter().filter(|&&x| x > 0.0).count() as f64,
        );

        stats
    }

    /// Calculate basic data quality metrics
    pub fn quality_metrics<T>(data: &Array2<T>) -> HashMap<String, f64>
    where
        T: Float + PartialOrd + std::iter::Sum + std::fmt::Display,
    {
        let mut metrics = HashMap::new();

        // Calculate completeness (non-missing ratio)
        let total_cells = data.len() as f64;
        let missing_count = data.iter().filter(|&&x| x.is_nan()).count() as f64;
        metrics.insert(
            "completeness".to_string(),
            1.0 - (missing_count / total_cells),
        );

        // Calculate uniformity (check for repeated values)
        let mut unique_counts = Vec::new();
        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let mut unique_values = std::collections::HashSet::new();
            for &value in col.iter() {
                if !value.is_nan() {
                    // Convert to string for hashing (approximation)
                    unique_values.insert(format!("{value:.6}"));
                }
            }
            let uniqueness = unique_values.len() as f64 / col.len() as f64;
            unique_counts.push(uniqueness);
        }

        let avg_uniqueness = unique_counts.iter().sum::<f64>() / unique_counts.len() as f64;
        metrics.insert("uniqueness".to_string(), avg_uniqueness);

        // Calculate consistency (low coefficient of variation)
        let mut cv_values = Vec::new();
        for col_idx in 0..data.ncols() {
            let col = data.column(col_idx);
            let valid_values: Vec<T> = col.iter().cloned().filter(|x| !x.is_nan()).collect();

            if valid_values.len() > 1 {
                let mean =
                    valid_values.iter().cloned().sum::<T>() / T::from(valid_values.len()).unwrap();
                let variance = valid_values.iter().map(|&x| (x - mean).powi(2)).sum::<T>()
                    / T::from(valid_values.len()).unwrap();
                let std_dev = variance.sqrt();

                if mean != T::zero() {
                    let cv = (std_dev / mean.abs()).to_f64().unwrap();
                    cv_values.push(cv);
                }
            }
        }

        if !cv_values.is_empty() {
            let avg_cv = cv_values.iter().sum::<f64>() / cv_values.len() as f64;
            metrics.insert("consistency".to_string(), 1.0 / (1.0 + avg_cv)); // Higher is better
        }

        metrics
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_drop_missing_rows() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, f64::NAN, 6.0],
            [7.0, 8.0, 9.0],
            [f64::NAN, 11.0, 12.0]
        ];

        let cleaned = DataCleaner::drop_missing_rows(&data).unwrap();
        assert_eq!(cleaned.nrows(), 2);
        assert_eq!(cleaned.row(0), array![1.0, 2.0, 3.0]);
        assert_eq!(cleaned.row(1), array![7.0, 8.0, 9.0]);
    }

    #[test]
    fn test_fill_missing_with_value() {
        let mut data = array![[1.0, 2.0], [f64::NAN, 4.0], [5.0, f64::NAN]];

        DataCleaner::fill_missing(&mut data, 0.0);

        assert_eq!(data, array![[1.0, 2.0], [0.0, 4.0], [5.0, 0.0]]);
    }

    #[test]
    fn test_fill_with_mean() {
        let mut data = array![[1.0, 2.0], [f64::NAN, 4.0], [5.0, f64::NAN]];

        DataCleaner::fill_with_mean(&mut data).unwrap();

        // Mean of first column (1, 5) = 3, mean of second column (2, 4) = 3
        assert_abs_diff_eq!(data[[1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(data[[2, 1]], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_zscore_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 100.0]; // 100 is clearly an outlier
        let outliers = OutlierDetector::zscore_outliers(&data.view(), 1.5);
        assert_eq!(outliers, vec![4]);
    }

    #[test]
    fn test_iqr_outliers() {
        let data = array![1.0, 2.0, 3.0, 4.0, 5.0, 100.0]; // 100 is an outlier
        let outliers = OutlierDetector::iqr_outliers(&data.view(), 1.5);
        assert_eq!(outliers, vec![5]);
    }

    #[test]
    fn test_standard_scaling() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let (scaled, _means, _stds) = FeatureScaler::standard_scale(&data).unwrap();

        // Check that scaled data has mean ~0 and std ~1
        for col_idx in 0..scaled.ncols() {
            let col = scaled.column(col_idx);
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            assert_abs_diff_eq!(mean, 0.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_minmax_scaling() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let (scaled, _mins, _maxs) = FeatureScaler::minmax_scale(&data).unwrap();

        // Check that scaled data is in [0, 1] range
        for col_idx in 0..scaled.ncols() {
            let col = scaled.column(col_idx);
            let min_val = col.iter().cloned().fold(col[0], f64::min);
            let max_val = col.iter().cloned().fold(col[0], f64::max);

            assert_abs_diff_eq!(min_val, 0.0, epsilon = 1e-10);
            assert_abs_diff_eq!(max_val, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_missing_value_stats() {
        let data = array![[1.0, 2.0, 3.0], [f64::NAN, 5.0, 6.0], [7.0, f64::NAN, 9.0]];

        let stats = DataQualityAssessor::missing_value_stats(&data);

        assert_abs_diff_eq!(stats["total_missing_ratio"], 2.0 / 9.0, epsilon = 1e-10);
        assert_eq!(stats["columns_with_missing"], 2.0);
        assert_eq!(stats["rows_with_missing"], 2.0);
    }

    #[test]
    fn test_quality_metrics() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let metrics = DataQualityAssessor::quality_metrics(&data);

        // All data is present, so completeness should be 1.0
        assert_abs_diff_eq!(metrics["completeness"], 1.0, epsilon = 1e-10);
        assert!(metrics.contains_key("uniqueness"));
        assert!(metrics.contains_key("consistency"));
    }
}
