//! Data Analysis Module for AutoML Feature Selection
//!
//! Analyzes dataset characteristics to inform feature selection method choice.
//! All implementations follow the SciRS2 policy using scirs2-core for numerical computations.

use scirs2_core::ndarray::{ArrayView1, ArrayView2};

use super::automl_core::{
    AutoMLError, ComputationalBudget, CorrelationStructure, DataCharacteristics, TargetType,
};
use sklears_core::error::Result as SklResult;

type Result<T> = SklResult<T>;

/// Data analyzer for understanding dataset characteristics
#[derive(Debug, Clone)]
pub struct DataAnalyzer;

impl DataAnalyzer {
    pub fn new() -> Self {
        Self
    }

    pub fn analyze_data(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<DataCharacteristics> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(AutoMLError::InsufficientData.into());
        }

        let feature_to_sample_ratio = n_features as f64 / n_samples as f64;

        // Determine target type
        let target_type = self.determine_target_type(y)?;

        // Check for missing values (NaN detection)
        let has_missing_values = X.iter().any(|&x| x.is_nan()) || y.iter().any(|&x| x.is_nan());

        // Check for categorical features (heuristic: integer values with low cardinality)
        let has_categorical_features = self.detect_categorical_features(X)?;

        // Compute feature variance distribution
        let feature_variance_distribution = self.compute_feature_variances(X)?;

        // Analyze correlation structure
        let correlation_structure = self.analyze_correlation_structure(X)?;

        // Set default computational budget
        let computational_budget = ComputationalBudget {
            max_time_seconds: 300.0, // 5 minutes default
            max_memory_mb: 1024.0,   // 1GB default
            prefer_speed: n_samples > 10000 || n_features > 1000,
            allow_complex_methods: n_samples >= 100 && feature_to_sample_ratio <= 1.0,
        };

        Ok(DataCharacteristics {
            n_samples,
            n_features,
            feature_to_sample_ratio,
            target_type,
            has_missing_values,
            has_categorical_features,
            feature_variance_distribution,
            correlation_structure,
            computational_budget,
        })
    }

    fn determine_target_type(&self, y: ArrayView1<f64>) -> Result<TargetType> {
        let unique_values: std::collections::HashSet<_> = y.iter().map(|&x| x as i32).collect();

        match unique_values.len() {
            2 => Ok(TargetType::BinaryClassification),
            3..=20 => {
                // Check if values are integers (classification) or continuous (regression)
                let is_integer = y.iter().all(|&x| x.fract() == 0.0);
                if is_integer {
                    Ok(TargetType::MultiClassification)
                } else {
                    Ok(TargetType::Regression)
                }
            }
            _ => Ok(TargetType::Regression),
        }
    }

    fn detect_categorical_features(&self, X: ArrayView2<f64>) -> Result<bool> {
        let mut categorical_count = 0;

        for col in 0..X.ncols() {
            let column = X.column(col);
            let unique_values: std::collections::HashSet<_> =
                column.iter().map(|&x| x as i32).collect();

            // Heuristic: if feature has integer values and low cardinality, consider categorical
            let is_integer = column.iter().all(|&x| x.fract() == 0.0);
            let low_cardinality = unique_values.len() <= 10 || unique_values.len() < X.nrows() / 10;

            if is_integer && low_cardinality {
                categorical_count += 1;
            }
        }

        Ok(categorical_count > X.ncols() / 4) // If >25% of features are categorical
    }

    fn compute_feature_variances(&self, X: ArrayView2<f64>) -> Result<Vec<f64>> {
        let mut variances = Vec::with_capacity(X.ncols());

        for col in 0..X.ncols() {
            let column = X.column(col);
            let variance = column.var(1.0);
            variances.push(variance);
        }

        Ok(variances)
    }

    fn analyze_correlation_structure(&self, X: ArrayView2<f64>) -> Result<CorrelationStructure> {
        let mut correlations = Vec::new();
        let mut high_correlation_pairs = 0;

        // Sample subset of feature pairs for efficiency
        let max_pairs = 1000;
        let n_features = X.ncols();
        let total_pairs = (n_features * (n_features - 1)) / 2;

        let step = if total_pairs > max_pairs {
            total_pairs / max_pairs
        } else {
            1
        };

        let mut pair_count = 0;
        for i in 0..n_features {
            for j in (i + 1)..n_features {
                if pair_count % step == 0 {
                    let corr = self.compute_correlation(X.column(i), X.column(j));
                    correlations.push(corr.abs());

                    if corr.abs() > 0.8 {
                        high_correlation_pairs += 1;
                    }
                }
                pair_count += 1;
            }
        }

        let average_correlation = if correlations.is_empty() {
            0.0
        } else {
            correlations.iter().sum::<f64>() / correlations.len() as f64
        };

        let max_correlation = correlations.iter().fold(0.0_f64, |acc, &x| acc.max(x));

        // Estimate correlation clusters (simplified)
        let correlation_clusters = if max_correlation > 0.7 {
            (high_correlation_pairs / 3).max(1) // Rough estimate
        } else {
            n_features / 10 // Assume features are mostly independent
        };

        Ok(CorrelationStructure {
            high_correlation_pairs,
            average_correlation,
            max_correlation,
            correlation_clusters,
        })
    }

    fn compute_correlation(&self, x: ArrayView1<f64>, y: ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        if n < 2.0 {
            return 0.0;
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let mut sum_xy = 0.0;
        let mut sum_x2 = 0.0;
        let mut sum_y2 = 0.0;

        for i in 0..x.len() {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_x2 += dx * dx;
            sum_y2 += dy * dy;
        }

        let denom = (sum_x2 * sum_y2).sqrt();
        if denom < 1e-10 {
            0.0
        } else {
            sum_xy / denom
        }
    }
}

impl Default for DataAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}
