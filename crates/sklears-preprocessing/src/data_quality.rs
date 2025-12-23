//! Data Quality Validation Framework
//!
//! Comprehensive data quality checks for preprocessing pipelines.
//! Validates data before and after transformations to ensure correctness.

use scirs2_core::ndarray::Array2;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Data quality report with comprehensive statistics
#[derive(Debug, Clone)]
pub struct DataQualityReport {
    /// Number of samples
    pub n_samples: usize,
    /// Number of features
    pub n_features: usize,
    /// Missing value statistics per feature
    pub missing_stats: Vec<MissingStats>,
    /// Outlier statistics per feature
    pub outlier_stats: Vec<OutlierStats>,
    /// Distribution statistics per feature
    pub distribution_stats: Vec<DistributionStats>,
    /// Correlation warnings
    pub correlation_warnings: Vec<CorrelationWarning>,
    /// Data quality score (0-100)
    pub quality_score: f64,
    /// List of detected issues
    pub issues: Vec<QualityIssue>,
}

/// Missing value statistics for a feature
#[derive(Debug, Clone)]
pub struct MissingStats {
    pub feature_idx: usize,
    pub missing_count: usize,
    pub missing_percentage: f64,
}

/// Outlier statistics for a feature
#[derive(Debug, Clone)]
pub struct OutlierStats {
    pub feature_idx: usize,
    pub outlier_count: usize,
    pub outlier_percentage: f64,
    pub outlier_indices: Vec<usize>,
}

/// Distribution statistics for a feature
#[derive(Debug, Clone)]
pub struct DistributionStats {
    pub feature_idx: usize,
    pub mean: f64,
    pub std: f64,
    pub min: f64,
    pub max: f64,
    pub median: f64,
    pub q25: f64,
    pub q75: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub unique_count: usize,
    pub constant: bool,
}

/// Correlation warning between features
#[derive(Debug, Clone)]
pub struct CorrelationWarning {
    pub feature_i: usize,
    pub feature_j: usize,
    pub correlation: f64,
}

/// Data quality issue
#[derive(Debug, Clone)]
pub struct QualityIssue {
    pub severity: IssueSeverity,
    pub category: IssueCategory,
    pub description: String,
    pub affected_features: Vec<usize>,
}

/// Issue severity level
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueSeverity {
    Critical,
    Warning,
    Info,
}

/// Issue category
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IssueCategory {
    MissingValues,
    Outliers,
    ConstantFeatures,
    HighCorrelation,
    Duplicates,
    DataType,
    Range,
    Distribution,
}

/// Configuration for data quality validation
#[derive(Debug, Clone)]
pub struct DataQualityConfig {
    /// Missing value threshold for warnings (percentage)
    pub missing_threshold: f64,
    /// Outlier detection method
    pub outlier_method: OutlierMethod,
    /// Outlier threshold (std deviations or IQR multiplier)
    pub outlier_threshold: f64,
    /// Correlation threshold for warnings
    pub correlation_threshold: f64,
    /// Check for duplicate samples
    pub check_duplicates: bool,
    /// Check for constant features
    pub check_constant_features: bool,
    /// Check for near-constant features (variance threshold)
    pub near_constant_threshold: f64,
}

impl Default for DataQualityConfig {
    fn default() -> Self {
        Self {
            missing_threshold: 10.0,
            outlier_method: OutlierMethod::ZScore,
            outlier_threshold: 3.0,
            correlation_threshold: 0.95,
            check_duplicates: true,
            check_constant_features: true,
            near_constant_threshold: 1e-8,
        }
    }
}

/// Outlier detection method
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OutlierMethod {
    ZScore,
    IQR,
    ModifiedZScore,
}

/// Data quality validator
pub struct DataQualityValidator {
    config: DataQualityConfig,
}

impl DataQualityValidator {
    /// Create a new validator with default configuration
    pub fn new() -> Self {
        Self {
            config: DataQualityConfig::default(),
        }
    }

    /// Create a validator with custom configuration
    pub fn with_config(config: DataQualityConfig) -> Self {
        Self { config }
    }

    /// Validate data and generate quality report
    pub fn validate(&self, x: &Array2<f64>) -> Result<DataQualityReport, SklearsError> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput("Empty dataset".to_string()));
        }

        let mut issues = Vec::new();

        // Compute missing value statistics
        let missing_stats = self.compute_missing_stats(x);
        for stat in &missing_stats {
            if stat.missing_percentage > self.config.missing_threshold {
                issues.push(QualityIssue {
                    severity: if stat.missing_percentage > 50.0 {
                        IssueSeverity::Critical
                    } else {
                        IssueSeverity::Warning
                    },
                    category: IssueCategory::MissingValues,
                    description: format!(
                        "Feature {} has {:.2}% missing values",
                        stat.feature_idx, stat.missing_percentage
                    ),
                    affected_features: vec![stat.feature_idx],
                });
            }
        }

        // Compute outlier statistics
        let outlier_stats = self.compute_outlier_stats(x);
        for stat in &outlier_stats {
            if stat.outlier_percentage > 5.0 {
                issues.push(QualityIssue {
                    severity: IssueSeverity::Warning,
                    category: IssueCategory::Outliers,
                    description: format!(
                        "Feature {} has {:.2}% outliers",
                        stat.feature_idx, stat.outlier_percentage
                    ),
                    affected_features: vec![stat.feature_idx],
                });
            }
        }

        // Compute distribution statistics
        let distribution_stats = self.compute_distribution_stats(x);
        if self.config.check_constant_features {
            for stat in &distribution_stats {
                if stat.constant {
                    issues.push(QualityIssue {
                        severity: IssueSeverity::Warning,
                        category: IssueCategory::ConstantFeatures,
                        description: format!("Feature {} is constant", stat.feature_idx),
                        affected_features: vec![stat.feature_idx],
                    });
                } else if stat.std < self.config.near_constant_threshold {
                    issues.push(QualityIssue {
                        severity: IssueSeverity::Info,
                        category: IssueCategory::ConstantFeatures,
                        description: format!(
                            "Feature {} has very low variance: {}",
                            stat.feature_idx, stat.std
                        ),
                        affected_features: vec![stat.feature_idx],
                    });
                }
            }
        }

        // Compute correlations
        let correlation_warnings = self.compute_correlation_warnings(x);
        for warning in &correlation_warnings {
            issues.push(QualityIssue {
                severity: IssueSeverity::Info,
                category: IssueCategory::HighCorrelation,
                description: format!(
                    "Features {} and {} are highly correlated: {:.3}",
                    warning.feature_i, warning.feature_j, warning.correlation
                ),
                affected_features: vec![warning.feature_i, warning.feature_j],
            });
        }

        // Check for duplicates
        if self.config.check_duplicates {
            let duplicate_count = self.count_duplicate_samples(x);
            if duplicate_count > 0 {
                issues.push(QualityIssue {
                    severity: IssueSeverity::Info,
                    category: IssueCategory::Duplicates,
                    description: format!(
                        "Found {} duplicate samples ({:.2}%)",
                        duplicate_count,
                        (duplicate_count as f64 / n_samples as f64) * 100.0
                    ),
                    affected_features: vec![],
                });
            }
        }

        // Calculate quality score
        let quality_score = self.calculate_quality_score(&issues, n_samples, n_features);

        Ok(DataQualityReport {
            n_samples,
            n_features,
            missing_stats,
            outlier_stats,
            distribution_stats,
            correlation_warnings,
            quality_score,
            issues,
        })
    }

    /// Compute missing value statistics
    fn compute_missing_stats(&self, x: &Array2<f64>) -> Vec<MissingStats> {
        let n_samples = x.nrows();

        (0..x.ncols())
            .map(|j| {
                let col = x.column(j);
                let missing_count = col.iter().filter(|v| v.is_nan()).count();
                let missing_percentage = (missing_count as f64 / n_samples as f64) * 100.0;

                MissingStats {
                    feature_idx: j,
                    missing_count,
                    missing_percentage,
                }
            })
            .collect()
    }

    /// Compute outlier statistics
    fn compute_outlier_stats(&self, x: &Array2<f64>) -> Vec<OutlierStats> {
        (0..x.ncols())
            .map(|j| {
                let col = x.column(j);
                let values: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();

                if values.is_empty() {
                    return OutlierStats {
                        feature_idx: j,
                        outlier_count: 0,
                        outlier_percentage: 0.0,
                        outlier_indices: vec![],
                    };
                }

                let outlier_indices = match self.config.outlier_method {
                    OutlierMethod::ZScore => self.detect_outliers_zscore(&values, j, x.nrows()),
                    OutlierMethod::IQR => self.detect_outliers_iqr(&values, j, x.nrows()),
                    OutlierMethod::ModifiedZScore => {
                        self.detect_outliers_modified_zscore(&values, j, x.nrows())
                    }
                };

                let outlier_count = outlier_indices.len();
                let outlier_percentage = (outlier_count as f64 / x.nrows() as f64) * 100.0;

                OutlierStats {
                    feature_idx: j,
                    outlier_count,
                    outlier_percentage,
                    outlier_indices,
                }
            })
            .collect()
    }

    /// Detect outliers using Z-score method
    fn detect_outliers_zscore(&self, values: &[f64], _col_idx: usize, n_rows: usize) -> Vec<usize> {
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std = variance.sqrt();

        if std < 1e-10 {
            return vec![];
        }

        let mut outliers = Vec::new();
        let mut value_idx = 0;

        for i in 0..n_rows {
            if value_idx < values.len() {
                let z_score = (values[value_idx] - mean).abs() / std;
                if z_score > self.config.outlier_threshold {
                    outliers.push(i);
                }
                value_idx += 1;
            }
        }

        outliers
    }

    /// Detect outliers using IQR method
    fn detect_outliers_iqr(&self, values: &[f64], _col_idx: usize, n_rows: usize) -> Vec<usize> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let q1_idx = sorted.len() / 4;
        let q3_idx = (sorted.len() * 3) / 4;
        let q1 = sorted[q1_idx];
        let q3 = sorted[q3_idx];
        let iqr = q3 - q1;

        if iqr < 1e-10 {
            return vec![];
        }

        let lower_bound = q1 - self.config.outlier_threshold * iqr;
        let upper_bound = q3 + self.config.outlier_threshold * iqr;

        let mut outliers = Vec::new();
        let mut value_idx = 0;

        for i in 0..n_rows {
            if value_idx < values.len() {
                let val = values[value_idx];
                if val < lower_bound || val > upper_bound {
                    outliers.push(i);
                }
                value_idx += 1;
            }
        }

        outliers
    }

    /// Detect outliers using Modified Z-score method
    fn detect_outliers_modified_zscore(
        &self,
        values: &[f64],
        _col_idx: usize,
        n_rows: usize,
    ) -> Vec<usize> {
        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let median = sorted[sorted.len() / 2];
        let mad = {
            let deviations: Vec<f64> = sorted.iter().map(|v| (v - median).abs()).collect();
            let mut dev_sorted = deviations.clone();
            dev_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            dev_sorted[dev_sorted.len() / 2]
        };

        if mad < 1e-10 {
            return vec![];
        }

        let mut outliers = Vec::new();
        let mut value_idx = 0;

        for i in 0..n_rows {
            if value_idx < values.len() {
                let modified_z = 0.6745 * (values[value_idx] - median).abs() / mad;
                if modified_z > self.config.outlier_threshold {
                    outliers.push(i);
                }
                value_idx += 1;
            }
        }

        outliers
    }

    /// Compute distribution statistics
    fn compute_distribution_stats(&self, x: &Array2<f64>) -> Vec<DistributionStats> {
        (0..x.ncols())
            .map(|j| {
                let col = x.column(j);
                let values: Vec<f64> = col.iter().copied().filter(|v| !v.is_nan()).collect();

                if values.is_empty() {
                    return DistributionStats {
                        feature_idx: j,
                        mean: f64::NAN,
                        std: f64::NAN,
                        min: f64::NAN,
                        max: f64::NAN,
                        median: f64::NAN,
                        q25: f64::NAN,
                        q75: f64::NAN,
                        skewness: f64::NAN,
                        kurtosis: f64::NAN,
                        unique_count: 0,
                        constant: true,
                    };
                }

                let mean = values.iter().sum::<f64>() / values.len() as f64;
                let variance =
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                let std = variance.sqrt();

                let mut sorted = values.clone();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

                let min = sorted.first().copied().unwrap_or(f64::NAN);
                let max = sorted.last().copied().unwrap_or(f64::NAN);
                let median = sorted[sorted.len() / 2];
                let q25 = sorted[sorted.len() / 4];
                let q75 = sorted[(sorted.len() * 3) / 4];

                let skewness = if std > 1e-10 {
                    values
                        .iter()
                        .map(|v| ((v - mean) / std).powi(3))
                        .sum::<f64>()
                        / values.len() as f64
                } else {
                    0.0
                };

                let kurtosis = if std > 1e-10 {
                    values
                        .iter()
                        .map(|v| ((v - mean) / std).powi(4))
                        .sum::<f64>()
                        / values.len() as f64
                        - 3.0
                } else {
                    0.0
                };

                // Count unique values (with epsilon comparison for floats)
                let mut unique_values = Vec::new();
                for &v in &values {
                    if !unique_values.iter().any(|&uv: &f64| (uv - v).abs() < 1e-10) {
                        unique_values.push(v);
                    }
                }
                let unique_count = unique_values.len();

                let constant = std < 1e-10;

                DistributionStats {
                    feature_idx: j,
                    mean,
                    std,
                    min,
                    max,
                    median,
                    q25,
                    q75,
                    skewness,
                    kurtosis,
                    unique_count,
                    constant,
                }
            })
            .collect()
    }

    /// Compute correlation warnings
    fn compute_correlation_warnings(&self, x: &Array2<f64>) -> Vec<CorrelationWarning> {
        let mut warnings = Vec::new();

        for i in 0..x.ncols() {
            for j in (i + 1)..x.ncols() {
                let corr = self.compute_correlation(x, i, j);
                if corr.abs() > self.config.correlation_threshold {
                    warnings.push(CorrelationWarning {
                        feature_i: i,
                        feature_j: j,
                        correlation: corr,
                    });
                }
            }
        }

        warnings
    }

    /// Compute Pearson correlation between two features
    fn compute_correlation(&self, x: &Array2<f64>, i: usize, j: usize) -> f64 {
        let col_i = x.column(i);
        let col_j = x.column(j);

        let pairs: Vec<(f64, f64)> = col_i
            .iter()
            .zip(col_j.iter())
            .filter(|(a, b)| !a.is_nan() && !b.is_nan())
            .map(|(&a, &b)| (a, b))
            .collect();

        if pairs.len() < 2 {
            return 0.0;
        }

        let mean_i = pairs.iter().map(|(a, _)| a).sum::<f64>() / pairs.len() as f64;
        let mean_j = pairs.iter().map(|(_, b)| b).sum::<f64>() / pairs.len() as f64;

        let mut cov = 0.0;
        let mut var_i = 0.0;
        let mut var_j = 0.0;

        for (a, b) in &pairs {
            let di = a - mean_i;
            let dj = b - mean_j;
            cov += di * dj;
            var_i += di * di;
            var_j += dj * dj;
        }

        if var_i < 1e-10 || var_j < 1e-10 {
            return 0.0;
        }

        cov / (var_i * var_j).sqrt()
    }

    /// Count duplicate samples
    fn count_duplicate_samples(&self, x: &Array2<f64>) -> usize {
        let mut seen = HashMap::new();
        let mut duplicates = 0;

        for i in 0..x.nrows() {
            let row: Vec<_> = x.row(i).iter().copied().collect();
            *seen.entry(format!("{:?}", row)).or_insert(0) += 1;
        }

        for count in seen.values() {
            if *count > 1 {
                duplicates += count - 1;
            }
        }

        duplicates
    }

    /// Calculate overall quality score (0-100)
    fn calculate_quality_score(
        &self,
        issues: &[QualityIssue],
        _n_samples: usize,
        _n_features: usize,
    ) -> f64 {
        let mut score: f64 = 100.0;

        for issue in issues {
            let penalty: f64 = match issue.severity {
                IssueSeverity::Critical => 20.0,
                IssueSeverity::Warning => 10.0,
                IssueSeverity::Info => 2.0,
            };
            score -= penalty;
        }

        score.max(0.0)
    }
}

impl Default for DataQualityValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl DataQualityReport {
    /// Print a human-readable summary of the report
    pub fn print_summary(&self) {
        println!("Data Quality Report");
        println!("==================");
        println!("Samples: {}, Features: {}", self.n_samples, self.n_features);
        println!("Quality Score: {:.1}/100", self.quality_score);
        println!();

        if !self.issues.is_empty() {
            println!("Issues Found: {}", self.issues.len());
            println!();

            for issue in &self.issues {
                let severity_str = match issue.severity {
                    IssueSeverity::Critical => "CRITICAL",
                    IssueSeverity::Warning => "WARNING",
                    IssueSeverity::Info => "INFO",
                };
                println!("[{}] {}", severity_str, issue.description);
            }
        } else {
            println!("No issues detected!");
        }
    }

    /// Get issues by severity
    pub fn issues_by_severity(&self, severity: IssueSeverity) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.severity == severity)
            .collect()
    }

    /// Get issues by category
    pub fn issues_by_category(&self, category: IssueCategory) -> Vec<&QualityIssue> {
        self.issues
            .iter()
            .filter(|issue| issue.category == category)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::random::essentials::Normal;
    use scirs2_core::random::{seeded_rng, Distribution};

    fn generate_test_data(nrows: usize, ncols: usize, seed: u64) -> Array2<f64> {
        let mut rng = seeded_rng(seed);
        let normal = Normal::new(0.0, 1.0).unwrap();

        let data: Vec<f64> = (0..nrows * ncols)
            .map(|_| normal.sample(&mut rng))
            .collect();

        Array2::from_shape_vec((nrows, ncols), data).unwrap()
    }

    #[test]
    fn test_data_quality_validator_basic() {
        let x = generate_test_data(100, 5, 42);
        let validator = DataQualityValidator::new();
        let report = validator.validate(&x).unwrap();

        assert_eq!(report.n_samples, 100);
        assert_eq!(report.n_features, 5);
        assert!(report.quality_score > 0.0);
    }

    #[test]
    fn test_missing_value_detection() {
        let mut x = generate_test_data(100, 3, 123);

        // Add missing values
        for i in 0..20 {
            x[[i, 0]] = f64::NAN;
        }

        let validator = DataQualityValidator::new();
        let report = validator.validate(&x).unwrap();

        let missing_in_col0 = &report.missing_stats[0];
        assert_eq!(missing_in_col0.missing_count, 20);
        assert!((missing_in_col0.missing_percentage - 20.0).abs() < 0.1);
    }

    #[test]
    fn test_constant_feature_detection() {
        let mut x = generate_test_data(50, 3, 456);

        // Make one feature constant
        for i in 0..x.nrows() {
            x[[i, 1]] = 5.0;
        }

        let validator = DataQualityValidator::new();
        let report = validator.validate(&x).unwrap();

        let constant_issues: Vec<_> = report.issues_by_category(IssueCategory::ConstantFeatures);

        assert!(!constant_issues.is_empty());
    }

    #[test]
    fn test_outlier_detection() {
        let mut x = generate_test_data(100, 2, 789);

        // Add outliers
        x[[0, 0]] = 100.0;
        x[[1, 0]] = -100.0;

        let validator = DataQualityValidator::new();
        let report = validator.validate(&x).unwrap();

        let outliers_in_col0 = &report.outlier_stats[0];
        assert!(outliers_in_col0.outlier_count > 0);
    }

    #[test]
    fn test_quality_score_calculation() {
        let x = generate_test_data(100, 5, 321);
        let validator = DataQualityValidator::new();
        let report = validator.validate(&x).unwrap();

        // Clean data should have high quality score
        assert!(report.quality_score > 80.0);
    }
}
