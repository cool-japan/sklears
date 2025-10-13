//! Feature engineering and interaction detection
//!
//! Automated feature engineering, interaction detection, and feature selection.

use scirs2_core::ndarray::{concatenate, Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::Result as SklResult,
    prelude::SklearsError,
    types::{Float, FloatBounds},
};
use std::collections::HashSet;

/// Feature interaction detector
pub struct FeatureInteractionDetector {
    interaction_type: InteractionType,
    max_interactions: usize,
    min_correlation: f64,
    method: DetectionMethod,
    threshold: f64,
}

/// Types of feature interactions to detect
#[derive(Debug, Clone)]
pub enum InteractionType {
    /// Linear interactions (correlation-based)
    Linear,
    /// Polynomial interactions
    Polynomial { degree: usize },
    /// Multiplicative interactions
    Multiplicative,
    /// Statistical interactions (ANOVA-based)
    Statistical,
    /// Mutual information based
    MutualInformation,
}

/// Methods for detecting interactions
#[derive(Debug, Clone)]
pub enum DetectionMethod {
    /// Correlation analysis
    Correlation,
    /// Mutual information
    MutualInfo,
    /// Statistical tests
    StatisticalTest,
    /// Tree-based importance
    TreeBased,
}

impl FeatureInteractionDetector {
    /// Create a new feature interaction detector
    #[must_use]
    pub fn new() -> Self {
        Self {
            interaction_type: InteractionType::Linear,
            max_interactions: 100,
            min_correlation: 0.1,
            method: DetectionMethod::Correlation,
            threshold: 0.05,
        }
    }

    /// Set interaction type
    #[must_use]
    pub fn interaction_type(mut self, interaction_type: InteractionType) -> Self {
        self.interaction_type = interaction_type;
        self
    }

    /// Set maximum number of interactions to detect
    #[must_use]
    pub fn max_interactions(mut self, max: usize) -> Self {
        self.max_interactions = max;
        self
    }

    /// Set minimum correlation threshold
    #[must_use]
    pub fn min_correlation(mut self, min_corr: f64) -> Self {
        self.min_correlation = min_corr;
        self
    }

    /// Set detection method
    #[must_use]
    pub fn method(mut self, method: DetectionMethod) -> Self {
        self.method = method;
        self
    }

    /// Set threshold for detection
    #[must_use]
    pub fn threshold(mut self, threshold: f64) -> Self {
        self.threshold = threshold;
        self
    }

    /// Detect feature interactions
    pub fn detect_interactions(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<FeatureInteraction>> {
        match self.method {
            DetectionMethod::Correlation => self.detect_correlation_interactions(x),
            DetectionMethod::MutualInfo => self.detect_mutual_info_interactions(x, y),
            DetectionMethod::StatisticalTest => self.detect_statistical_interactions(x, y),
            DetectionMethod::TreeBased => self.detect_tree_based_interactions(x, y),
        }
    }

    fn detect_correlation_interactions(
        &self,
        x: &ArrayView2<'_, Float>,
    ) -> SklResult<Vec<FeatureInteraction>> {
        let mut interactions = Vec::new();
        let n_features = x.ncols();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let correlation = self.calculate_correlation(&x.column(i), &x.column(j))?;

                if correlation.abs() >= self.min_correlation {
                    interactions.push(FeatureInteraction {
                        feature_indices: vec![i, j],
                        interaction_type: self.interaction_type.clone(),
                        strength: correlation.abs(),
                        p_value: None,
                    });
                }
            }
        }

        // Sort by strength and take top interactions
        interactions.sort_by(|a, b| b.strength.partial_cmp(&a.strength).unwrap());
        interactions.truncate(self.max_interactions);

        Ok(interactions)
    }

    fn detect_mutual_info_interactions(
        &self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<FeatureInteraction>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn detect_statistical_interactions(
        &self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<FeatureInteraction>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn detect_tree_based_interactions(
        &self,
        _x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Vec<FeatureInteraction>> {
        // Placeholder implementation
        Ok(Vec::new())
    }

    fn calculate_correlation(
        &self,
        x1: &ArrayView1<'_, Float>,
        x2: &ArrayView1<'_, Float>,
    ) -> SklResult<f64> {
        let n = x1.len();
        if n != x2.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{n}"),
                actual: format!("{}", x2.len()),
            });
        }

        let mean1 = x1.iter().copied().sum::<f64>() / n as f64;
        let mean2 = x2.iter().copied().sum::<f64>() / n as f64;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..n {
            let diff1 = x1[i] - mean1;
            let diff2 = x2[i] - mean2;

            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
}

impl Default for FeatureInteractionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Represents a detected feature interaction
#[derive(Debug, Clone)]
pub struct FeatureInteraction {
    /// Indices of features involved in the interaction
    pub feature_indices: Vec<usize>,
    /// Type of interaction
    pub interaction_type: InteractionType,
    /// Strength of interaction
    pub strength: f64,
    /// Statistical significance (p-value)
    pub p_value: Option<f64>,
}

/// Automatic feature engineering pipeline
pub struct AutoFeatureEngineer {
    enable_polynomial: bool,
    polynomial_degree: usize,
    enable_interactions: bool,
    enable_binning: bool,
    n_bins: usize,
    enable_scaling: bool,
    enable_selection: bool,
    max_features: Option<usize>,
}

impl AutoFeatureEngineer {
    /// Create a new automatic feature engineer
    #[must_use]
    pub fn new() -> Self {
        Self {
            enable_polynomial: true,
            polynomial_degree: 2,
            enable_interactions: true,
            enable_binning: false,
            n_bins: 10,
            enable_scaling: true,
            enable_selection: true,
            max_features: None,
        }
    }

    /// Enable/disable polynomial features
    #[must_use]
    pub fn polynomial_features(mut self, enable: bool, degree: usize) -> Self {
        self.enable_polynomial = enable;
        self.polynomial_degree = degree;
        self
    }

    /// Enable/disable interaction features
    #[must_use]
    pub fn interaction_features(mut self, enable: bool) -> Self {
        self.enable_interactions = enable;
        self
    }

    /// Enable/disable binning features
    #[must_use]
    pub fn binning_features(mut self, enable: bool, n_bins: usize) -> Self {
        self.enable_binning = enable;
        self.n_bins = n_bins;
        self
    }

    /// Enable/disable scaling
    #[must_use]
    pub fn scaling(mut self, enable: bool) -> Self {
        self.enable_scaling = enable;
        self
    }

    /// Enable/disable feature selection
    #[must_use]
    pub fn feature_selection(mut self, enable: bool, max_features: Option<usize>) -> Self {
        self.enable_selection = enable;
        self.max_features = max_features;
        self
    }

    /// Generate engineered features
    pub fn generate_features(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array2<f64>> {
        let mut engineered = x.mapv(|v| v);

        if self.enable_polynomial {
            engineered = self.add_polynomial_features(&engineered)?;
        }

        if self.enable_interactions {
            engineered = self.add_interaction_features(&engineered)?;
        }

        if self.enable_binning {
            engineered = self.add_binning_features(&engineered)?;
        }

        if self.enable_scaling {
            engineered = self.apply_scaling(&engineered)?;
        }

        if self.enable_selection {
            engineered = self.select_features(&engineered, y)?;
        }

        Ok(engineered)
    }

    fn add_polynomial_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut features = x.clone();

        for degree in 2..=self.polynomial_degree {
            for i in 0..n_features {
                let mut poly_col = Array1::zeros(n_samples);
                for (j, &val) in x.column(i).iter().enumerate() {
                    poly_col[j] = val.powi(degree as i32);
                }

                // Add polynomial feature as new column
                let new_features = concatenate![Axis(1), features, poly_col.insert_axis(Axis(1))];
                features = new_features;
            }
        }

        Ok(features)
    }

    fn add_interaction_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut features = x.clone();

        for i in 0..n_features {
            for j in (i + 1)..n_features {
                let mut interaction_col = Array1::zeros(n_samples);
                for k in 0..n_samples {
                    interaction_col[k] = x[[k, i]] * x[[k, j]];
                }

                // Add interaction feature as new column
                let new_features =
                    concatenate![Axis(1), features, interaction_col.insert_axis(Axis(1))];
                features = new_features;
            }
        }

        Ok(features)
    }

    fn add_binning_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut features = x.clone();

        for i in 0..n_features {
            let column = x.column(i);
            let min_val = column.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_val = column.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let bin_width = (max_val - min_val) / self.n_bins as f64;

            let mut binned_col = Array1::zeros(n_samples);
            for (j, &val) in column.iter().enumerate() {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                binned_col[j] = bin.min(self.n_bins - 1) as f64;
            }

            // Add binned feature as new column
            let new_features = concatenate![Axis(1), features, binned_col.insert_axis(Axis(1))];
            features = new_features;
        }

        Ok(features)
    }

    fn apply_scaling(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = x.dim();
        let mut scaled = Array2::zeros((n_samples, n_features));

        for i in 0..n_features {
            let column = x.column(i);
            let mean = column.mean().unwrap_or(0.0);
            let std = column.var(0.0).sqrt();

            for j in 0..n_samples {
                scaled[[j, i]] = if std > 0.0 {
                    (x[[j, i]] - mean) / std
                } else {
                    0.0
                };
            }
        }

        Ok(scaled)
    }

    fn select_features(
        &self,
        x: &Array2<f64>,
        _y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<Array2<f64>> {
        // Simple feature selection based on variance
        let (n_samples, n_features) = x.dim();

        if let Some(max_features) = self.max_features {
            if max_features >= n_features {
                return Ok(x.clone());
            }

            let mut feature_scores = Vec::new();

            for i in 0..n_features {
                let column = x.column(i);
                let variance = column.var(0.0);
                feature_scores.push((i, variance));
            }

            // Sort by variance (descending)
            feature_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

            // Select top features
            let selected_indices: Vec<usize> = feature_scores
                .into_iter()
                .take(max_features)
                .map(|(idx, _)| idx)
                .collect();

            // Create new array with selected features
            let mut selected = Array2::zeros((n_samples, max_features));
            for (new_idx, &old_idx) in selected_indices.iter().enumerate() {
                for j in 0..n_samples {
                    selected[[j, new_idx]] = x[[j, old_idx]];
                }
            }

            Ok(selected)
        } else {
            Ok(x.clone())
        }
    }
}

impl Default for AutoFeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

/// Column type detector for automatic preprocessing
pub struct ColumnTypeDetector {
    categorical_threshold: f64,
    date_pattern_detection: bool,
    text_detection: bool,
}

/// Detected column types
#[derive(Debug, Clone, PartialEq)]
pub enum ColumnType {
    /// Numeric (continuous)
    Numeric,
    /// Categorical
    Categorical,
    /// Boolean
    Boolean,
    /// Date/Time
    DateTime,
    /// Text
    Text,
    /// Binary (0/1)
    Binary,
    /// Ordinal
    Ordinal,
}

impl ColumnTypeDetector {
    /// Create a new column type detector
    #[must_use]
    pub fn new() -> Self {
        Self {
            categorical_threshold: 0.1, // If unique values / total values < threshold, consider categorical
            date_pattern_detection: true,
            text_detection: true,
        }
    }

    /// Set categorical threshold
    #[must_use]
    pub fn categorical_threshold(mut self, threshold: f64) -> Self {
        self.categorical_threshold = threshold;
        self
    }

    /// Enable/disable date pattern detection
    #[must_use]
    pub fn date_pattern_detection(mut self, enable: bool) -> Self {
        self.date_pattern_detection = enable;
        self
    }

    /// Enable/disable text detection
    #[must_use]
    pub fn text_detection(mut self, enable: bool) -> Self {
        self.text_detection = enable;
        self
    }

    /// Detect column types
    #[must_use]
    pub fn detect_types(&self, x: &ArrayView2<'_, Float>) -> Vec<ColumnType> {
        let mut column_types = Vec::new();

        for i in 0..x.ncols() {
            let column = x.column(i);
            let column_type = self.detect_column_type(&column);
            column_types.push(column_type);
        }

        column_types
    }

    fn detect_column_type(&self, column: &ArrayView1<'_, Float>) -> ColumnType {
        let unique_values = self.count_unique_values(column);
        let total_values = column.len();
        let unique_ratio = unique_values as f64 / total_values as f64;

        // Check for binary
        if unique_values == 2 {
            return ColumnType::Binary;
        }

        // Check for boolean (assuming 0.0 and 1.0 represent false and true)
        if self.is_boolean_column(column) {
            return ColumnType::Boolean;
        }

        // Check for categorical
        if unique_ratio < self.categorical_threshold {
            return ColumnType::Categorical;
        }

        // Default to numeric
        ColumnType::Numeric
    }

    fn count_unique_values(&self, column: &ArrayView1<'_, Float>) -> usize {
        let mut unique_set = HashSet::new();
        for &value in column {
            // Use a small epsilon for floating point comparison
            let rounded = (value * 1000.0).round() / 1000.0;
            unique_set.insert(rounded.to_bits());
        }
        unique_set.len()
    }

    fn is_boolean_column(&self, column: &ArrayView1<'_, Float>) -> bool {
        for &value in column {
            if value != 0.0 && value != 1.0 {
                return false;
            }
        }
        true
    }
}

impl Default for ColumnTypeDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_feature_interaction_detector() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let detector = FeatureInteractionDetector::new().min_correlation(0.5);

        let interactions = detector.detect_interactions(&x.view(), None).unwrap();
        assert!(!interactions.is_empty());
    }

    #[test]
    fn test_auto_feature_engineer() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let engineer = AutoFeatureEngineer::new()
            .polynomial_features(true, 2)
            .interaction_features(true);

        let engineered = engineer.generate_features(&x.view(), None).unwrap();
        assert!(engineered.ncols() > x.ncols());
    }

    #[test]
    fn test_column_type_detector() {
        let x = array![[0.0, 1.0, 5.5], [1.0, 0.0, 6.2], [0.0, 1.0, 7.8]];

        let detector = ColumnTypeDetector::new();
        let types = detector.detect_types(&x.view());

        assert_eq!(types.len(), 3);
        assert_eq!(types[0], ColumnType::Binary); // Binary column
        assert_eq!(types[1], ColumnType::Binary); // Binary column
        assert_eq!(types[2], ColumnType::Numeric); // Numeric column
    }
}
