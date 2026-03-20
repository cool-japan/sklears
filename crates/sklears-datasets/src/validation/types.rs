//! Core data structures for dataset validation framework
//!
//! This module defines the data structures used throughout the validation
//! system for statistical analysis, quality metrics, and validation reporting.

use std::collections::HashMap;
use std::fmt;

/// Comprehensive statistical summary for a dataset feature
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FeatureStatistics {
    pub name: String,
    pub count: usize,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub variance: f64,
    pub min: f64,
    pub max: f64,
    pub range: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub q1: f64,
    pub q3: f64,
    pub iqr: f64,
    pub outlier_count: usize,
    pub outlier_ratio: f64,
    pub missing_count: usize,
    pub missing_ratio: f64,
    pub unique_count: usize,
    pub mode: Option<f64>,
    pub percentiles: HashMap<u8, f64>,
}

/// Summary statistics for target/label data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct TargetStatistics {
    pub name: String,
    pub data_type: String, // "continuous", "categorical", "binary"
    pub count: usize,
    pub unique_count: usize,
    pub missing_count: usize,
    pub missing_ratio: f64,
    pub class_distribution: HashMap<String, usize>,
    pub class_balance_ratio: f64,
    pub entropy: f64,
    pub continuous_stats: Option<FeatureStatistics>,
}

/// Complete dataset statistical summary
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct StatisticalSummary {
    pub dataset_name: String,
    pub n_samples: usize,
    pub n_features: usize,
    pub feature_statistics: Vec<FeatureStatistics>,
    pub target_statistics: Option<TargetStatistics>,
    pub correlation_matrix: Vec<Vec<f64>>,
    pub feature_correlations: HashMap<String, f64>,
    pub data_quality_score: f64,
    pub missing_data_pattern: String,
    pub outlier_summary: HashMap<String, usize>,
    pub distribution_types: HashMap<String, String>,
    pub generation_timestamp: String,
    pub metadata: HashMap<String, String>,
}

impl StatisticalSummary {
    /// Create a new statistical summary
    pub fn new(dataset_name: String, n_samples: usize, n_features: usize) -> Self {
        Self {
            dataset_name,
            n_samples,
            n_features,
            feature_statistics: Vec::new(),
            target_statistics: None,
            correlation_matrix: Vec::new(),
            feature_correlations: HashMap::new(),
            data_quality_score: 0.0,
            missing_data_pattern: "None".to_string(),
            outlier_summary: HashMap::new(),
            distribution_types: HashMap::new(),
            generation_timestamp: chrono::Utc::now().to_rfc3339(),
            metadata: HashMap::new(),
        }
    }

    /// Add feature statistics
    pub fn add_feature_stats(&mut self, stats: FeatureStatistics) {
        self.feature_statistics.push(stats);
    }

    /// Set target statistics
    pub fn set_target_stats(&mut self, stats: TargetStatistics) {
        self.target_statistics = Some(stats);
    }

    /// Calculate data quality score (0-100)
    pub fn calculate_quality_score(&mut self) {
        let mut score = 100.0;

        // Penalize for missing data
        let total_missing_ratio: f64 = self
            .feature_statistics
            .iter()
            .map(|fs| fs.missing_ratio)
            .sum::<f64>()
            / self.feature_statistics.len() as f64;
        score -= total_missing_ratio * 50.0;

        // Penalize for excessive outliers
        let total_outlier_ratio: f64 = self
            .feature_statistics
            .iter()
            .map(|fs| fs.outlier_ratio)
            .sum::<f64>()
            / self.feature_statistics.len() as f64;
        if total_outlier_ratio > 0.1 {
            score -= (total_outlier_ratio - 0.1) * 100.0;
        }

        // Penalize for extreme class imbalance
        if let Some(ref target_stats) = self.target_statistics {
            if target_stats.class_balance_ratio < 0.1 {
                score -= (0.1 - target_stats.class_balance_ratio) * 200.0;
            }
        }

        self.data_quality_score = score.max(0.0);
    }

    /// Export summary as JSON string
    #[cfg(feature = "serde")]
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Export summary as CSV string
    pub fn to_csv(&self) -> String {
        let mut csv = String::new();
        csv.push_str("metric,value\n");
        csv.push_str(&format!("dataset_name,{}\n", self.dataset_name));
        csv.push_str(&format!("n_samples,{}\n", self.n_samples));
        csv.push_str(&format!("n_features,{}\n", self.n_features));
        csv.push_str(&format!(
            "data_quality_score,{:.2}\n",
            self.data_quality_score
        ));
        csv.push_str(&format!(
            "missing_data_pattern,{}\n",
            self.missing_data_pattern
        ));
        csv.push_str(&format!(
            "generation_timestamp,{}\n",
            self.generation_timestamp
        ));

        for (i, feature) in self.feature_statistics.iter().enumerate() {
            csv.push_str(&format!("feature_{}_mean,{:.4}\n", i, feature.mean));
            csv.push_str(&format!("feature_{}_std,{:.4}\n", i, feature.std_dev));
            csv.push_str(&format!("feature_{}_min,{:.4}\n", i, feature.min));
            csv.push_str(&format!("feature_{}_max,{:.4}\n", i, feature.max));
            csv.push_str(&format!(
                "feature_{}_outlier_ratio,{:.4}\n",
                i, feature.outlier_ratio
            ));
        }

        csv
    }
}

impl fmt::Display for StatisticalSummary {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Statistical Summary: {} ===", self.dataset_name)?;
        writeln!(
            f,
            "Dataset Shape: {} samples × {} features",
            self.n_samples, self.n_features
        )?;
        writeln!(f, "Data Quality Score: {:.2}/100", self.data_quality_score)?;
        writeln!(f, "Missing Data Pattern: {}", self.missing_data_pattern)?;
        writeln!(f, "Generation Time: {}", self.generation_timestamp)?;
        writeln!(f)?;

        writeln!(f, "Feature Statistics:")?;
        for (i, feature) in self.feature_statistics.iter().enumerate() {
            writeln!(f, "  Feature {}: {}", i, feature.name)?;
            writeln!(
                f,
                "    Mean: {:.4}, Std: {:.4}",
                feature.mean, feature.std_dev
            )?;
            writeln!(
                f,
                "    Min: {:.4}, Max: {:.4}, Range: {:.4}",
                feature.min, feature.max, feature.range
            )?;
            writeln!(
                f,
                "    Skewness: {:.4}, Kurtosis: {:.4}",
                feature.skewness, feature.kurtosis
            )?;
            writeln!(
                f,
                "    Outliers: {} ({:.2}%)",
                feature.outlier_count,
                feature.outlier_ratio * 100.0
            )?;
            writeln!(
                f,
                "    Missing: {} ({:.2}%)",
                feature.missing_count,
                feature.missing_ratio * 100.0
            )?;
        }

        if let Some(ref target) = self.target_statistics {
            writeln!(f)?;
            writeln!(f, "Target Statistics:")?;
            writeln!(f, "  Type: {}", target.data_type)?;
            writeln!(f, "  Unique Values: {}", target.unique_count)?;
            writeln!(
                f,
                "  Class Balance Ratio: {:.4}",
                target.class_balance_ratio
            )?;
            writeln!(f, "  Entropy: {:.4}", target.entropy)?;
            if !target.class_distribution.is_empty() {
                writeln!(f, "  Class Distribution:")?;
                for (class, count) in &target.class_distribution {
                    writeln!(f, "    {}: {}", class, count)?;
                }
            }
        }

        Ok(())
    }
}

/// Configuration for statistical summary generation
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SummaryConfig {
    pub include_percentiles: bool,
    pub percentile_values: Vec<u8>,
    pub include_correlation_matrix: bool,
    pub outlier_threshold: f64,
    pub missing_threshold: f64,
    pub include_distribution_analysis: bool,
    pub feature_names: Option<Vec<String>>,
    pub target_name: Option<String>,
}

impl Default for SummaryConfig {
    fn default() -> Self {
        Self {
            include_percentiles: true,
            percentile_values: vec![5, 10, 25, 50, 75, 90, 95],
            include_correlation_matrix: true,
            outlier_threshold: 1.5,
            missing_threshold: 0.05,
            include_distribution_analysis: true,
            feature_names: None,
            target_name: None,
        }
    }
}

/// Validation result for a single statistical property
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub property: String,
    pub passed: bool,
    pub expected: f64,
    pub actual: f64,
    pub tolerance: f64,
    pub message: String,
}

/// Comprehensive validation report for a dataset
#[derive(Debug, Clone)]
pub struct ValidationReport {
    pub total_tests: usize,
    pub passed_tests: usize,
    pub failed_tests: usize,
    pub results: Vec<ValidationResult>,
    pub overall_pass: bool,
}

impl Default for ValidationReport {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationReport {
    /// Create a new validation report
    pub fn new() -> Self {
        Self {
            total_tests: 0,
            passed_tests: 0,
            failed_tests: 0,
            results: Vec::new(),
            overall_pass: true,
        }
    }

    /// Add a validation result to the report
    pub fn add_result(&mut self, result: ValidationResult) {
        self.total_tests += 1;
        if result.passed {
            self.passed_tests += 1;
        } else {
            self.failed_tests += 1;
            self.overall_pass = false;
        }
        self.results.push(result);
    }

    /// Get the success rate as a percentage
    pub fn success_rate(&self) -> f64 {
        if self.total_tests == 0 {
            0.0
        } else {
            (self.passed_tests as f64 / self.total_tests as f64) * 100.0
        }
    }
}

impl fmt::Display for ValidationReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "=== Dataset Validation Report ===")?;
        writeln!(f, "Total tests: {}", self.total_tests)?;
        writeln!(f, "Passed: {}", self.passed_tests)?;
        writeln!(f, "Failed: {}", self.failed_tests)?;
        writeln!(f, "Success rate: {:.2}%", self.success_rate())?;
        writeln!(
            f,
            "Overall: {}",
            if self.overall_pass { "PASS" } else { "FAIL" }
        )?;
        writeln!(f)?;

        for result in &self.results {
            let status = if result.passed { "PASS" } else { "FAIL" };
            writeln!(f, "[{}] {}: {}", status, result.property, result.message)?;
        }

        Ok(())
    }
}

/// Dataset validation configuration
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    pub tolerance: f64,
    pub min_samples: usize,
    pub check_normality: bool,
    pub check_correlation: bool,
    pub check_distribution: bool,
    pub check_outliers: bool,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            tolerance: 0.1,
            min_samples: 10,
            check_normality: true,
            check_correlation: true,
            check_distribution: true,
            check_outliers: true,
        }
    }
}

/// Dataset quality metrics for comprehensive quality assessment
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DatasetQualityMetrics {
    pub overall_quality_score: f64,
    pub completeness_score: f64,
    pub consistency_score: f64,
    pub validity_score: f64,
    pub accuracy_score: f64,
    pub uniqueness_score: f64,
    pub timeliness_score: f64,
    pub missing_data_ratio: f64,
    pub outlier_ratio: f64,
    pub duplicate_ratio: f64,
    pub data_type_violations: usize,
    pub range_violations: usize,
    pub pattern_violations: usize,
    pub fingerprint: String,
    pub quality_issues: Vec<String>,
    pub recommendations: Vec<String>,
}

impl DatasetQualityMetrics {
    /// Calculate overall quality score from individual metrics
    pub fn calculate_overall_score(&mut self) {
        let weights = [0.25, 0.20, 0.15, 0.15, 0.10, 0.15]; // Completeness, Consistency, Validity, Accuracy, Uniqueness, Timeliness
        let scores = [
            self.completeness_score,
            self.consistency_score,
            self.validity_score,
            self.accuracy_score,
            self.uniqueness_score,
            self.timeliness_score,
        ];

        self.overall_quality_score = scores
            .iter()
            .zip(weights.iter())
            .map(|(score, weight)| score * weight)
            .sum::<f64>()
            .max(0.0)
            .min(100.0);
    }

    /// Add a quality issue with recommendation
    pub fn add_issue(&mut self, issue: String, recommendation: String) {
        self.quality_issues.push(issue);
        self.recommendations.push(recommendation);
    }

    /// Generate quality report summary
    pub fn generate_report(&self) -> String {
        let mut report = String::new();
        report.push_str(&"=== Dataset Quality Report ===\n".to_string());
        report.push_str(&format!(
            "Overall Quality Score: {:.2}/100\n",
            self.overall_quality_score
        ));
        report.push_str(&format!(
            "Completeness: {:.2}/100\n",
            self.completeness_score
        ));
        report.push_str(&format!("Consistency: {:.2}/100\n", self.consistency_score));
        report.push_str(&format!("Validity: {:.2}/100\n", self.validity_score));
        report.push_str(&format!("Accuracy: {:.2}/100\n", self.accuracy_score));
        report.push_str(&format!("Uniqueness: {:.2}/100\n", self.uniqueness_score));
        report.push_str(&format!("Timeliness: {:.2}/100\n", self.timeliness_score));
        report.push_str(&"\nData Issues:\n".to_string());
        report.push_str(&format!(
            "- Missing Data: {:.2}%\n",
            self.missing_data_ratio * 100.0
        ));
        report.push_str(&format!("- Outliers: {:.2}%\n", self.outlier_ratio * 100.0));
        report.push_str(&format!(
            "- Duplicates: {:.2}%\n",
            self.duplicate_ratio * 100.0
        ));
        report.push_str(&format!(
            "- Type Violations: {}\n",
            self.data_type_violations
        ));
        report.push_str(&format!("- Range Violations: {}\n", self.range_violations));
        report.push_str(&format!(
            "- Pattern Violations: {}\n",
            self.pattern_violations
        ));
        report.push_str(&format!("\nFingerprint: {}\n", self.fingerprint));

        if !self.quality_issues.is_empty() {
            report.push_str(&"\nQuality Issues:\n".to_string());
            for issue in &self.quality_issues {
                report.push_str(&format!("- {}\n", issue));
            }
        }

        if !self.recommendations.is_empty() {
            report.push_str(&"\nRecommendations:\n".to_string());
            for rec in &self.recommendations {
                report.push_str(&format!("- {}\n", rec));
            }
        }

        report
    }
}

/// Data drift detection result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct DataDriftReport {
    pub drift_detected: bool,
    pub drift_score: f64,
    pub drift_threshold: f64,
    pub affected_features: Vec<String>,
    pub drift_statistics: HashMap<String, f64>,
    pub drift_type: String, // "covariate", "concept", "prior_probability"
    pub detection_method: String,
    pub confidence_level: f64,
    pub timestamp: String,
}

impl Default for DataDriftReport {
    fn default() -> Self {
        Self::new()
    }
}

impl DataDriftReport {
    /// Create a new drift report
    pub fn new() -> Self {
        Self {
            drift_detected: false,
            drift_score: 0.0,
            drift_threshold: 0.05,
            affected_features: Vec::new(),
            drift_statistics: HashMap::new(),
            drift_type: "none".to_string(),
            detection_method: "kolmogorov_smirnov".to_string(),
            confidence_level: 0.95,
            timestamp: chrono::Utc::now().to_rfc3339(),
        }
    }

    /// Add drift detection for a feature
    pub fn add_feature_drift(&mut self, feature_name: String, drift_statistic: f64) {
        self.drift_statistics
            .insert(feature_name.clone(), drift_statistic);
        if drift_statistic > self.drift_threshold {
            self.affected_features.push(feature_name);
            self.drift_detected = true;
        }
    }

    /// Calculate overall drift score
    pub fn calculate_overall_drift(&mut self) {
        if self.drift_statistics.is_empty() {
            return;
        }

        self.drift_score =
            self.drift_statistics.values().sum::<f64>() / self.drift_statistics.len() as f64;

        if self.drift_score > self.drift_threshold {
            self.drift_detected = true;
        }
    }
}

/// Anomaly detection result in generated data
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct AnomalyDetectionResult {
    pub anomalies_detected: bool,
    pub anomaly_count: usize,
    pub anomaly_ratio: f64,
    pub anomaly_threshold: f64,
    pub anomaly_indices: Vec<usize>,
    pub anomaly_scores: Vec<f64>,
    pub detection_method: String,
    pub feature_anomalies: HashMap<String, Vec<usize>>,
}

impl Default for AnomalyDetectionResult {
    fn default() -> Self {
        Self::new()
    }
}

impl AnomalyDetectionResult {
    /// Create a new anomaly detection result
    pub fn new() -> Self {
        Self {
            anomalies_detected: false,
            anomaly_count: 0,
            anomaly_ratio: 0.0,
            anomaly_threshold: 0.05,
            anomaly_indices: Vec::new(),
            anomaly_scores: Vec::new(),
            detection_method: "isolation_forest".to_string(),
            feature_anomalies: HashMap::new(),
        }
    }

    /// Add an anomaly
    pub fn add_anomaly(&mut self, index: usize, score: f64) {
        self.anomaly_indices.push(index);
        self.anomaly_scores.push(score);
        self.anomalies_detected = true;
    }

    /// Calculate anomaly statistics
    pub fn calculate_statistics(&mut self, total_samples: usize) {
        self.anomaly_count = self.anomaly_indices.len();
        self.anomaly_ratio = self.anomaly_count as f64 / total_samples as f64;

        if self.anomaly_ratio > self.anomaly_threshold {
            self.anomalies_detected = true;
        }
    }
}
