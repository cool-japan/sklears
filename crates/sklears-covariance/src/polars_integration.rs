//! Polars DataFrame integration for covariance estimation
//!
//! This module provides seamless integration with Polars DataFrames, allowing
//! direct covariance estimation from DataFrames with automatic type handling,
//! column validation, and result formatting.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, NdFloat};
use scirs2_core::Distribution;
use sklears_core::error::{Result as SklResult, SklearsError};
use std::collections::HashMap;

/// Polars DataFrame wrapper for covariance estimation
///
/// This struct provides a bridge between Polars DataFrames and covariance estimators,
/// handling type conversions, column validation, and metadata preservation.
#[derive(Debug, Clone)]
pub struct CovarianceDataFrame {
    /// Column names from the original DataFrame
    pub column_names: Vec<String>,
    /// Numeric data matrix
    pub data: Array2<f64>,
    /// Missing value indicators
    pub missing_mask: Option<Array2<bool>>,
    /// Column metadata (data types, statistics, etc.)
    pub metadata: DataFrameMetadata,
}

/// Metadata about the DataFrame columns
#[derive(Debug, Clone)]
pub struct DataFrameMetadata {
    /// Original column data types
    pub column_types: HashMap<String, String>,
    /// Column statistics (mean, std, min, max, etc.)
    pub column_stats: HashMap<String, ColumnStatistics>,
    /// Number of missing values per column
    pub missing_counts: HashMap<String, usize>,
    /// Total number of rows (including missing)
    pub n_rows: usize,
    /// Number of numeric columns used for covariance
    pub n_features: usize,
}

/// Statistical summary for a column
#[derive(Debug, Clone)]
pub struct ColumnStatistics {
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// 25th percentile
    pub q25: f64,
    /// Median (50th percentile)
    pub median: f64,
    /// 75th percentile
    pub q75: f64,
    /// Number of non-missing values
    pub count: usize,
}

/// Result of covariance estimation with DataFrame context
#[derive(Debug, Clone)]
pub struct CovarianceResult<F: NdFloat> {
    /// Estimated covariance matrix
    pub covariance: Array2<F>,
    /// Precision matrix (inverse covariance) if available
    pub precision: Option<Array2<F>>,
    /// Feature names corresponding to matrix dimensions
    pub feature_names: Vec<String>,
    /// Original DataFrame metadata
    pub metadata: DataFrameMetadata,
    /// Estimator-specific metadata
    pub estimator_info: EstimatorInfo,
}

/// Information about the estimator used
#[derive(Debug, Clone)]
pub struct EstimatorInfo {
    /// Name of the estimator
    pub name: String,
    /// Estimator parameters
    pub parameters: HashMap<String, String>,
    /// Convergence information
    pub convergence: Option<ConvergenceInfo>,
    /// Performance metrics
    pub metrics: Option<PerformanceMetrics>,
}

/// Convergence information for iterative estimators
#[derive(Debug, Clone)]
pub struct ConvergenceInfo {
    /// Number of iterations performed
    pub n_iterations: usize,
    /// Whether convergence was achieved
    pub converged: bool,
    /// Final objective value
    pub objective_value: Option<f64>,
    /// Convergence tolerance achieved
    pub tolerance: Option<f64>,
}

/// Performance metrics for the estimation
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Computation time in milliseconds
    pub computation_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: Option<f64>,
    /// Condition number of the result
    pub condition_number: Option<f64>,
    /// Log-likelihood if applicable
    pub log_likelihood: Option<f64>,
}

impl CovarianceDataFrame {
    /// Create a new CovarianceDataFrame from a Polars-compatible data structure
    ///
    /// # Arguments
    /// * `data` - 2D array of numeric data
    /// * `column_names` - Names of the columns
    /// * `missing_mask` - Optional mask indicating missing values
    ///
    /// # Returns
    /// * `CovarianceDataFrame` instance ready for covariance estimation
    pub fn new(
        data: Array2<f64>,
        column_names: Vec<String>,
        missing_mask: Option<Array2<bool>>,
    ) -> SklResult<Self> {
        let (n_rows, n_features) = data.dim();

        if column_names.len() != n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Number of column names ({}) doesn't match number of features ({})",
                column_names.len(),
                n_features
            )));
        }

        // Compute basic statistics for each column
        let mut column_stats = HashMap::new();
        let mut missing_counts = HashMap::new();

        for (col_idx, col_name) in column_names.iter().enumerate() {
            let column_data = data.column(col_idx);
            let missing_count = if let Some(ref mask) = missing_mask {
                mask.column(col_idx).iter().filter(|&&x| x).count()
            } else {
                0
            };

            // Filter out missing values for statistics
            let valid_data: Vec<f64> = if let Some(ref mask) = missing_mask {
                column_data
                    .iter()
                    .zip(mask.column(col_idx).iter())
                    .filter_map(|(&val, &is_missing)| if !is_missing { Some(val) } else { None })
                    .collect()
            } else {
                column_data.to_vec()
            };

            if !valid_data.is_empty() {
                let stats = compute_column_statistics(&valid_data);
                column_stats.insert(col_name.clone(), stats);
            }

            missing_counts.insert(col_name.clone(), missing_count);
        }

        let metadata = DataFrameMetadata {
            column_types: column_names
                .iter()
                .map(|name| (name.clone(), "f64".to_string()))
                .collect(),
            column_stats,
            missing_counts,
            n_rows,
            n_features,
        };

        Ok(Self {
            column_names,
            data,
            missing_mask,
            metadata,
        })
    }

    /// Get the data as an ArrayView2 for use with estimators
    pub fn as_array_view(&self) -> ArrayView2<'_, f64> {
        self.data.view()
    }

    /// Get column names
    pub fn column_names(&self) -> &[String] {
        &self.column_names
    }

    /// Get the shape of the data
    pub fn shape(&self) -> (usize, usize) {
        self.data.dim()
    }

    /// Check if there are any missing values
    pub fn has_missing_values(&self) -> bool {
        self.missing_mask.is_some()
    }

    /// Get missing value ratio for each column
    pub fn missing_ratios(&self) -> HashMap<String, f64> {
        self.metadata
            .missing_counts
            .iter()
            .map(|(name, &count)| (name.clone(), count as f64 / self.metadata.n_rows as f64))
            .collect()
    }

    /// Filter out rows with missing values
    pub fn drop_missing(&self) -> SklResult<Self> {
        if let Some(ref mask) = self.missing_mask {
            // Find rows with no missing values
            let valid_rows: Vec<usize> = (0..self.metadata.n_rows)
                .filter(|&row_idx| !mask.row(row_idx).iter().any(|&x| x))
                .collect();

            if valid_rows.is_empty() {
                return Err(SklearsError::InvalidInput(
                    "No rows without missing values found".to_string(),
                ));
            }

            // Create new data array with only valid rows
            let new_data =
                Array2::from_shape_fn((valid_rows.len(), self.metadata.n_features), |(i, j)| {
                    self.data[[valid_rows[i], j]]
                });

            Self::new(new_data, self.column_names.clone(), None)
        } else {
            Ok(self.clone())
        }
    }

    /// Validate data for covariance estimation
    pub fn validate(&self) -> SklResult<()> {
        let (n_rows, n_features) = self.shape();

        if n_rows < 2 {
            return Err(SklearsError::InvalidInput(
                "At least 2 observations required for covariance estimation".to_string(),
            ));
        }

        if n_features < 1 {
            return Err(SklearsError::InvalidInput(
                "At least 1 feature required for covariance estimation".to_string(),
            ));
        }

        // Check for infinite or NaN values
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let column_data = self.data.column(col_idx);
            if column_data.iter().any(|&x| !x.is_finite()) {
                return Err(SklearsError::InvalidInput(format!(
                    "Column '{}' contains infinite or NaN values",
                    col_name
                )));
            }
        }

        // Check for constant columns
        for (col_idx, col_name) in self.column_names.iter().enumerate() {
            let column_data = self.data.column(col_idx);
            let first_val = column_data[0];
            if column_data.iter().all(|&x| (x - first_val).abs() < 1e-12) {
                return Err(SklearsError::InvalidInput(format!(
                    "Column '{}' is constant (zero variance)",
                    col_name
                )));
            }
        }

        Ok(())
    }

    /// Get summary statistics for the DataFrame
    pub fn describe(&self) -> DataFrameDescription {
        DataFrameDescription {
            shape: self.shape(),
            column_names: self.column_names.clone(),
            metadata: self.metadata.clone(),
        }
    }
}

/// Summary description of a DataFrame
#[derive(Debug, Clone)]
pub struct DataFrameDescription {
    /// Shape (rows, columns)
    pub shape: (usize, usize),
    /// Column names
    pub column_names: Vec<String>,
    /// Detailed metadata
    pub metadata: DataFrameMetadata,
}

impl<F: NdFloat> CovarianceResult<F> {
    /// Create a new covariance result
    pub fn new(
        covariance: Array2<F>,
        precision: Option<Array2<F>>,
        feature_names: Vec<String>,
        metadata: DataFrameMetadata,
        estimator_info: EstimatorInfo,
    ) -> Self {
        Self {
            covariance,
            precision,
            feature_names,
            metadata,
            estimator_info,
        }
    }

    /// Get covariance matrix with feature names
    pub fn covariance_with_names(&self) -> HashMap<(String, String), F> {
        let mut result = HashMap::new();
        for (i, name_i) in self.feature_names.iter().enumerate() {
            for (j, name_j) in self.feature_names.iter().enumerate() {
                result.insert((name_i.clone(), name_j.clone()), self.covariance[[i, j]]);
            }
        }
        result
    }

    /// Get correlation matrix from covariance matrix
    pub fn correlation(&self) -> SklResult<Array2<F>> {
        let n = self.covariance.nrows();
        if n != self.covariance.ncols() {
            return Err(SklearsError::InvalidInput(
                "Covariance matrix is not square".to_string(),
            ));
        }

        let mut correlation = Array2::zeros((n, n));

        // Extract standard deviations from diagonal
        let mut std_devs = Vec::with_capacity(n);
        for i in 0..n {
            let variance = self.covariance[[i, i]];
            if variance <= F::zero() {
                return Err(SklearsError::InvalidInput(format!(
                    "Feature {} has non-positive variance",
                    self.feature_names[i]
                )));
            }
            std_devs.push(variance.sqrt());
        }

        // Compute correlation coefficients
        for i in 0..n {
            for j in 0..n {
                correlation[[i, j]] = self.covariance[[i, j]] / (std_devs[i] * std_devs[j]);
            }
        }

        Ok(correlation)
    }

    /// Get variance array (diagonal of covariance matrix)
    pub fn variances(&self) -> Array1<F> {
        self.covariance.diag().to_owned()
    }

    /// Get standard deviation array
    pub fn standard_deviations(&self) -> Array1<F> {
        self.variances().mapv(|x| x.sqrt())
    }
}

/// Compute statistical summary for a column of data
fn compute_column_statistics(data: &[f64]) -> ColumnStatistics {
    let n = data.len();
    if n == 0 {
        return ColumnStatistics {
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            q25: 0.0,
            median: 0.0,
            q75: 0.0,
            count: 0,
        };
    }

    let mut sorted_data = data.to_vec();
    sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mean = data.iter().sum::<f64>() / n as f64;
    let variance = data.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64;
    let std_dev = variance.sqrt();

    let q25 = percentile(&sorted_data, 0.25);
    let median = percentile(&sorted_data, 0.5);
    let q75 = percentile(&sorted_data, 0.75);

    ColumnStatistics {
        mean,
        std_dev,
        min: sorted_data[0],
        max: sorted_data[n - 1],
        q25,
        median,
        q75,
        count: n,
    }
}

/// Compute percentile of sorted data
fn percentile(sorted_data: &[f64], p: f64) -> f64 {
    let n = sorted_data.len();
    if n == 0 {
        return 0.0;
    }
    if n == 1 {
        return sorted_data[0];
    }

    let index = p * (n - 1) as f64;
    let lower = index.floor() as usize;
    let upper = index.ceil() as usize;

    if lower == upper {
        sorted_data[lower]
    } else {
        let weight = index - lower as f64;
        sorted_data[lower] * (1.0 - weight) + sorted_data[upper] * weight
    }
}

/// Trait for estimators that can work directly with DataFrames
pub trait DataFrameEstimator<F: NdFloat> {
    /// Fit the estimator to a DataFrame and return enhanced results
    fn fit_dataframe(&self, df: &CovarianceDataFrame) -> SklResult<CovarianceResult<F>>;

    /// Get estimator name
    fn name(&self) -> &str;

    /// Get estimator parameters as a string map
    fn parameters(&self) -> HashMap<String, String>;
}

/// Utility functions for DataFrame integration
pub mod utils {
    use super::*;

    /// Convert a slice of slices to a CovarianceDataFrame
    pub fn from_slices(
        data: &[&[f64]],
        column_names: Vec<String>,
    ) -> SklResult<CovarianceDataFrame> {
        if data.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Empty data provided".to_string(),
            ));
        }

        let n_rows = data.len();
        let n_cols = data[0].len();

        // Validate all rows have same length
        for (i, row) in data.iter().enumerate() {
            if row.len() != n_cols {
                return Err(SklearsError::InvalidInput(format!(
                    "Row {} has {} columns, expected {}",
                    i,
                    row.len(),
                    n_cols
                )));
            }
        }

        let mut array_data = Array2::zeros((n_rows, n_cols));
        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                array_data[[i, j]] = val;
            }
        }

        CovarianceDataFrame::new(array_data, column_names, None)
    }

    /// Create sample data for testing
    pub fn create_sample_data(
        n_samples: usize,
        n_features: usize,
    ) -> SklResult<CovarianceDataFrame> {
        use scirs2_core::random::thread_rng;

        let mut rng = thread_rng();
        let mut data = Array2::zeros((n_samples, n_features));

        let normal = scirs2_core::StandardNormal;
        for i in 0..n_samples {
            for j in 0..n_features {
                data[[i, j]] = normal.sample(&mut rng);
            }
        }

        let column_names = (0..n_features).map(|i| format!("feature_{}", i)).collect();

        CovarianceDataFrame::new(data, column_names, None)
    }

    /// Standardize DataFrame columns (zero mean, unit variance)
    pub fn standardize(df: &CovarianceDataFrame) -> SklResult<CovarianceDataFrame> {
        let (n_rows, n_features) = df.shape();
        let mut standardized_data = Array2::zeros((n_rows, n_features));

        for j in 0..n_features {
            let column = df.data.column(j);
            let mean = column.sum() / n_rows as f64;
            let variance =
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (n_rows - 1) as f64;
            let std_dev = variance.sqrt();

            if std_dev < 1e-12 {
                return Err(SklearsError::InvalidInput(format!(
                    "Column '{}' has zero variance",
                    df.column_names[j]
                )));
            }

            for i in 0..n_rows {
                standardized_data[[i, j]] = (df.data[[i, j]] - mean) / std_dev;
            }
        }

        CovarianceDataFrame::new(
            standardized_data,
            df.column_names.clone(),
            df.missing_mask.clone(),
        )
    }

    /// Center DataFrame columns (zero mean)
    pub fn center(df: &CovarianceDataFrame) -> SklResult<CovarianceDataFrame> {
        let (n_rows, n_features) = df.shape();
        let mut centered_data = Array2::zeros((n_rows, n_features));

        for j in 0..n_features {
            let column = df.data.column(j);
            let mean = column.sum() / n_rows as f64;

            for i in 0..n_rows {
                centered_data[[i, j]] = df.data[[i, j]] - mean;
            }
        }

        CovarianceDataFrame::new(
            centered_data,
            df.column_names.clone(),
            df.missing_mask.clone(),
        )
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_covariance_dataframe_creation() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let column_names = vec!["feature1".to_string(), "feature2".to_string()];

        let df = CovarianceDataFrame::new(data, column_names, None).unwrap();

        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.column_names(), &["feature1", "feature2"]);
        assert!(!df.has_missing_values());
    }

    #[test]
    fn test_dataframe_validation() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let column_names = vec!["feature1".to_string(), "feature2".to_string()];

        let df = CovarianceDataFrame::new(data, column_names, None).unwrap();
        assert!(df.validate().is_ok());
    }

    #[test]
    fn test_missing_value_handling() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let column_names = vec!["feature1".to_string(), "feature2".to_string()];
        let missing_mask = Some(array![[false, false], [true, false], [false, false]]);

        let df = CovarianceDataFrame::new(data, column_names, missing_mask).unwrap();

        assert!(df.has_missing_values());
        let missing_ratios = df.missing_ratios();
        assert_eq!(missing_ratios["feature1"], 1.0 / 3.0);
        assert_eq!(missing_ratios["feature2"], 0.0);
    }

    #[test]
    fn test_column_statistics() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let stats = compute_column_statistics(&data);

        assert_eq!(stats.mean, 3.0);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 5.0);
        assert_eq!(stats.median, 3.0);
        assert_eq!(stats.count, 5);
    }

    #[test]
    fn test_percentile_calculation() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];

        assert_eq!(percentile(&data, 0.0), 1.0);
        assert_eq!(percentile(&data, 0.5), 3.0);
        assert_eq!(percentile(&data, 1.0), 5.0);
        assert_eq!(percentile(&data, 0.25), 2.0);
        assert_eq!(percentile(&data, 0.75), 4.0);
    }

    #[test]
    fn test_utils_from_slices() {
        let vec1 = vec![1.0, 2.0];
        let vec2 = vec![3.0, 4.0];
        let vec3 = vec![5.0, 6.0];
        let data = vec![vec1.as_slice(), vec2.as_slice(), vec3.as_slice()];
        let column_names = vec!["A".to_string(), "B".to_string()];

        let df = utils::from_slices(&data, column_names).unwrap();

        assert_eq!(df.shape(), (3, 2));
        assert_eq!(df.column_names(), &["A", "B"]);
    }

    #[test]
    fn test_standardization() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let column_names = vec!["A".to_string(), "B".to_string()];
        let df = CovarianceDataFrame::new(data, column_names, None).unwrap();

        let standardized = utils::standardize(&df).unwrap();

        // Check that means are approximately zero
        let col0_mean = standardized.data.column(0).sum() / 3.0;
        let col1_mean = standardized.data.column(1).sum() / 3.0;
        assert!(col0_mean.abs() < 1e-10);
        assert!(col1_mean.abs() < 1e-10);
    }

    #[test]
    fn test_centering() {
        let data = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];
        let column_names = vec!["A".to_string(), "B".to_string()];
        let df = CovarianceDataFrame::new(data, column_names, None).unwrap();

        let centered = utils::center(&df).unwrap();

        // Check that means are approximately zero
        let col0_mean = centered.data.column(0).sum() / 3.0;
        let col1_mean = centered.data.column(1).sum() / 3.0;
        assert!(col0_mean.abs() < 1e-10);
        assert!(col1_mean.abs() < 1e-10);
    }
}
