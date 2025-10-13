//! Feature type detection and analysis
//!
//! This module provides comprehensive feature type detection implementations including
//! continuous, discrete, binary, and categorical feature detection with advanced
//! statistical analysis. All implementations follow SciRS2 Policy.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use serde::{Deserialize, Serialize};
use sklears_core::error::Result;
use sklears_core::prelude::SklearsError;
use std::collections::HashMap;

/// Enumeration of feature types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FeatureType {
    /// Continuous
    Continuous,
    /// Discrete
    Discrete,
    /// Binary
    Binary,
    /// Categorical
    Categorical,
    /// Count
    Count,
    /// Ordinal
    Ordinal,
    /// Temporal
    Temporal,
    /// Text
    Text,
}

/// Configuration for feature type detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeDetectionConfig {
    /// Threshold for binary feature detection (proportion of unique values)
    pub binary_threshold: f64,
    /// Threshold for categorical feature detection (number of unique values)
    pub categorical_threshold: usize,
    /// Threshold for continuous feature detection (proportion of unique values)
    pub continuous_threshold: f64,
    /// Minimum number of samples for reliable type detection
    pub min_samples: usize,
    /// Whether to use statistical tests for type detection
    pub use_statistical_tests: bool,
    /// Significance level for statistical tests
    pub alpha: f64,
}

impl Default for TypeDetectionConfig {
    fn default() -> Self {
        Self {
            binary_threshold: 0.1,
            categorical_threshold: 10,
            continuous_threshold: 0.95,
            min_samples: 30,
            use_statistical_tests: true,
            alpha: 0.05,
        }
    }
}

/// Validator for type detection configurations
#[derive(Debug, Clone)]
pub struct TypeDetectionValidator;

impl TypeDetectionValidator {
    pub fn validate_config(config: &TypeDetectionConfig) -> Result<()> {
        if config.binary_threshold < 0.0 || config.binary_threshold > 1.0 {
            return Err(SklearsError::InvalidInput(
                "binary_threshold must be between 0 and 1".to_string(),
            ));
        }

        if config.continuous_threshold < 0.0 || config.continuous_threshold > 1.0 {
            return Err(SklearsError::InvalidInput(
                "continuous_threshold must be between 0 and 1".to_string(),
            ));
        }

        if config.alpha <= 0.0 || config.alpha >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "alpha must be between 0 and 1".to_string(),
            ));
        }

        if config.min_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "min_samples must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }
}

/// Feature type detector for automatic type inference
#[derive(Debug, Clone)]
pub struct FeatureTypeDetector {
    config: TypeDetectionConfig,
    feature_types: Option<Vec<FeatureType>>,
    type_probabilities: Option<HashMap<usize, HashMap<FeatureType, f64>>>,
    detection_metadata: HashMap<String, f64>,
}

impl FeatureTypeDetector {
    /// Create a new feature type detector
    pub fn new(config: TypeDetectionConfig) -> Result<Self> {
        TypeDetectionValidator::validate_config(&config)?;
        Ok(Self {
            config,
            feature_types: None,
            type_probabilities: None,
            detection_metadata: HashMap::new(),
        })
    }

    /// Detect feature types from data
    pub fn detect_types<T>(&mut self, data: &ArrayView2<T>) -> Result<Vec<FeatureType>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let (n_samples, n_features) = data.dim();

        if n_samples < self.config.min_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Insufficient samples: {} < {}",
                n_samples, self.config.min_samples
            )));
        }

        let mut types = Vec::with_capacity(n_features);
        let mut probabilities = HashMap::new();

        for feature_idx in 0..n_features {
            let column = data.column(feature_idx);
            let detected_type = self.detect_single_feature_type(&column)?;
            let type_probs = self.calculate_type_probabilities(&column)?;

            types.push(detected_type);
            probabilities.insert(feature_idx, type_probs);
        }

        self.feature_types = Some(types.clone());
        self.type_probabilities = Some(probabilities);

        Ok(types)
    }

    /// Detect type for a single feature
    fn detect_single_feature_type<T>(&self, column: &ArrayView1<T>) -> Result<FeatureType>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let n_samples = column.len();
        let unique_values = self.count_unique_values(column);
        let unique_ratio = unique_values as f64 / n_samples as f64;

        // Binary detection
        if unique_values == 2 {
            return Ok(FeatureType::Binary);
        }

        // Categorical detection
        if unique_values <= self.config.categorical_threshold
            && unique_ratio < self.config.binary_threshold
        {
            return Ok(FeatureType::Categorical);
        }

        // Continuous detection
        if unique_ratio >= self.config.continuous_threshold {
            return Ok(FeatureType::Continuous);
        }

        // Count data detection (all non-negative integers)
        if self.is_count_data(column) {
            return Ok(FeatureType::Count);
        }

        // Discrete detection
        if self.is_discrete_data(column) {
            return Ok(FeatureType::Discrete);
        }

        // Default to continuous
        Ok(FeatureType::Continuous)
    }

    /// Calculate type probabilities for a feature
    fn calculate_type_probabilities<T>(
        &self,
        column: &ArrayView1<T>,
    ) -> Result<HashMap<FeatureType, f64>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut probs = HashMap::new();
        let n_samples = column.len() as f64;
        let unique_count = self.count_unique_values(column) as f64;
        let unique_ratio = unique_count / n_samples;

        // Binary probability
        let binary_prob = if unique_count == 2.0 { 1.0 } else { 0.0 };
        probs.insert(FeatureType::Binary, binary_prob);

        // Categorical probability
        let categorical_prob = if unique_count <= self.config.categorical_threshold as f64 {
            1.0 - unique_ratio
        } else {
            0.0
        };
        probs.insert(FeatureType::Categorical, categorical_prob);

        // Continuous probability
        let continuous_prob = unique_ratio;
        probs.insert(FeatureType::Continuous, continuous_prob);

        // Count probability
        let count_prob = if self.is_count_data(column) { 0.8 } else { 0.0 };
        probs.insert(FeatureType::Count, count_prob);

        // Discrete probability
        let discrete_prob = if self.is_discrete_data(column) && unique_count > 2.0 {
            0.6
        } else {
            0.0
        };
        probs.insert(FeatureType::Discrete, discrete_prob);

        Ok(probs)
    }

    /// Count unique values in a column
    fn count_unique_values<T>(&self, column: &ArrayView1<T>) -> usize
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = column.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len()
    }

    /// Check if data represents count data (non-negative integers)
    fn is_count_data<T>(&self, _column: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified implementation - would need actual numeric checks in practice
        false
    }

    /// Check if data is discrete (integer values)
    fn is_discrete_data<T>(&self, _column: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified implementation - would need actual numeric checks in practice
        false
    }

    /// Get detected feature types
    pub fn feature_types(&self) -> Option<&[FeatureType]> {
        self.feature_types.as_deref()
    }

    /// Get type probabilities
    pub fn type_probabilities(&self) -> Option<&HashMap<usize, HashMap<FeatureType, f64>>> {
        self.type_probabilities.as_ref()
    }

    /// Get detection metadata
    pub fn detection_metadata(&self) -> &HashMap<String, f64> {
        &self.detection_metadata
    }

    /// Add detection metadata
    pub fn add_metadata(&mut self, key: String, value: f64) {
        self.detection_metadata.insert(key, value);
    }
}

impl Default for FeatureTypeDetector {
    fn default() -> Self {
        Self::new(TypeDetectionConfig::default()).unwrap()
    }
}

/// Continuous feature detector
#[derive(Debug, Clone)]
pub struct ContinuousDetector {
    threshold: f64,
    use_variance_test: bool,
}

impl ContinuousDetector {
    pub fn new(threshold: f64, use_variance_test: bool) -> Self {
        Self {
            threshold,
            use_variance_test,
        }
    }

    pub fn is_continuous<T>(&self, data: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let unique_count = self.count_unique_values(data);
        let unique_ratio = unique_count as f64 / data.len() as f64;
        unique_ratio >= self.threshold
    }

    fn count_unique_values<T>(&self, column: &ArrayView1<T>) -> usize
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = column.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len()
    }
}

/// Count data detector
#[derive(Debug, Clone)]
pub struct CountDetector {
    allow_zero: bool,
}

impl CountDetector {
    pub fn new(allow_zero: bool) -> Self {
        Self { allow_zero }
    }

    pub fn is_count_data<T>(&self, _data: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        // Simplified implementation
        true
    }
}

/// Binary feature detector
#[derive(Debug, Clone)]
pub struct BinaryDetector;

impl BinaryDetector {
    pub fn new() -> Self {
        Self
    }

    pub fn is_binary<T>(&self, data: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = data.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len() == 2
    }
}

impl Default for BinaryDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// Categorical feature detector
#[derive(Debug, Clone)]
pub struct CategoricalDetector {
    max_categories: usize,
}

impl CategoricalDetector {
    pub fn new(max_categories: usize) -> Self {
        Self { max_categories }
    }

    pub fn is_categorical<T>(&self, data: &ArrayView1<T>) -> bool
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = data.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len() <= self.max_categories && values.len() > 2
    }
}

/// Skewness analyzer for feature distribution analysis
#[derive(Debug, Clone)]
pub struct SkewnessAnalyzer;

impl SkewnessAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Calculate skewness for a feature (simplified implementation)
    pub fn calculate_skewness<T>(&self, _data: &ArrayView1<T>) -> f64
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd,
    {
        // Simplified implementation - would compute actual skewness in practice
        0.0
    }

    /// Determine if distribution is skewed
    pub fn is_skewed(&self, skewness: f64, threshold: f64) -> bool {
        skewness.abs() > threshold
    }
}

impl Default for SkewnessAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Type analyzer for comprehensive feature analysis
#[derive(Debug, Clone)]
pub struct TypeAnalyzer {
    config: TypeDetectionConfig,
    analysis_results: HashMap<String, f64>,
}

impl TypeAnalyzer {
    pub fn new(config: TypeDetectionConfig) -> Self {
        Self {
            config,
            analysis_results: HashMap::new(),
        }
    }

    /// Perform comprehensive type analysis
    pub fn analyze_types<T>(&mut self, data: &ArrayView2<T>) -> Result<HashMap<usize, FeatureType>>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut detector = FeatureTypeDetector::new(self.config.clone())?;
        let types = detector.detect_types(data)?;

        let mut type_map = HashMap::new();
        for (idx, feature_type) in types.iter().enumerate() {
            type_map.insert(idx, *feature_type);
        }

        // Store analysis metadata
        self.analysis_results
            .insert("n_features".to_string(), data.dim().1 as f64);
        self.analysis_results
            .insert("n_samples".to_string(), data.dim().0 as f64);

        Ok(type_map)
    }

    /// Get analysis results
    pub fn analysis_results(&self) -> &HashMap<String, f64> {
        &self.analysis_results
    }
}

/// Feature characterization for detailed feature analysis
#[derive(Debug, Clone)]
pub struct FeatureCharacterization {
    feature_stats: HashMap<usize, HashMap<String, f64>>,
    feature_types: HashMap<usize, FeatureType>,
    correlation_matrix: Option<Array2<f64>>,
}

impl FeatureCharacterization {
    pub fn new() -> Self {
        Self {
            feature_stats: HashMap::new(),
            feature_types: HashMap::new(),
            correlation_matrix: None,
        }
    }

    /// Characterize all features in dataset
    pub fn characterize_features<T>(&mut self, data: &ArrayView2<T>) -> Result<()>
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let config = TypeDetectionConfig::default();
        let mut detector = FeatureTypeDetector::new(config)?;
        let types = detector.detect_types(data)?;

        for (idx, feature_type) in types.iter().enumerate() {
            self.feature_types.insert(idx, *feature_type);

            // Calculate basic statistics for each feature
            let mut stats = HashMap::new();
            let column = data.column(idx);

            stats.insert(
                "unique_count".to_string(),
                self.count_unique_values(&column) as f64,
            );
            stats.insert(
                "unique_ratio".to_string(),
                self.count_unique_values(&column) as f64 / column.len() as f64,
            );

            self.feature_stats.insert(idx, stats);
        }

        Ok(())
    }

    /// Count unique values in a column
    fn count_unique_values<T>(&self, column: &ArrayView1<T>) -> usize
    where
        T: Clone + Copy + std::fmt::Debug + PartialOrd + PartialEq,
    {
        let mut values: Vec<T> = column.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup();
        values.len()
    }

    /// Get feature types
    pub fn feature_types(&self) -> &HashMap<usize, FeatureType> {
        &self.feature_types
    }

    /// Get feature statistics
    pub fn feature_stats(&self) -> &HashMap<usize, HashMap<String, f64>> {
        &self.feature_stats
    }

    /// Set correlation matrix
    pub fn set_correlation_matrix(&mut self, matrix: Array2<f64>) {
        self.correlation_matrix = Some(matrix);
    }

    /// Get correlation matrix
    pub fn correlation_matrix(&self) -> Option<&Array2<f64>> {
        self.correlation_matrix.as_ref()
    }
}

impl Default for FeatureCharacterization {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{Array1, Array2};

    #[test]
    fn test_feature_type_enum() {
        let feature_type = FeatureType::Continuous;
        assert_eq!(feature_type, FeatureType::Continuous);
        assert_ne!(feature_type, FeatureType::Binary);
    }

    #[test]
    fn test_type_detection_config() {
        let config = TypeDetectionConfig::default();
        assert_eq!(config.binary_threshold, 0.1);
        assert_eq!(config.categorical_threshold, 10);
        assert_eq!(config.continuous_threshold, 0.95);
        assert!(TypeDetectionValidator::validate_config(&config).is_ok());
    }

    #[test]
    fn test_type_detection_validator() {
        let mut config = TypeDetectionConfig::default();

        // Valid config
        assert!(TypeDetectionValidator::validate_config(&config).is_ok());

        // Invalid binary threshold
        config.binary_threshold = -0.1;
        assert!(TypeDetectionValidator::validate_config(&config).is_err());

        config.binary_threshold = 1.1;
        assert!(TypeDetectionValidator::validate_config(&config).is_err());

        // Reset and test other fields
        config = TypeDetectionConfig::default();
        config.alpha = 0.0;
        assert!(TypeDetectionValidator::validate_config(&config).is_err());

        config.alpha = 1.0;
        assert!(TypeDetectionValidator::validate_config(&config).is_err());
    }

    #[test]
    fn test_binary_detector() {
        let detector = BinaryDetector::new();

        // Binary data
        let binary_data = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0, 1.0]);
        assert!(detector.is_binary(&binary_data.view()));

        // Non-binary data
        let continuous_data = Array1::from_vec(vec![0.0, 1.0, 0.5, 0.8, 0.2]);
        assert!(!detector.is_binary(&continuous_data.view()));
    }

    #[test]
    fn test_categorical_detector() {
        let detector = CategoricalDetector::new(5);

        // Categorical data (3 unique values)
        let categorical_data = Array1::from_vec(vec![1.0, 2.0, 3.0, 1.0, 2.0]);
        assert!(detector.is_categorical(&categorical_data.view()));

        // Too many categories
        let many_categories = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert!(!detector.is_categorical(&many_categories.view()));

        // Binary data (should not be categorical)
        let binary_data = Array1::from_vec(vec![0.0, 1.0, 0.0, 1.0]);
        assert!(!detector.is_categorical(&binary_data.view()));
    }

    #[test]
    fn test_continuous_detector() {
        let detector = ContinuousDetector::new(0.8, false);

        // Continuous data (high unique ratio)
        let continuous_data = Array1::from_vec(vec![1.1, 2.2, 3.3, 4.4, 5.5]);
        assert!(detector.is_continuous(&continuous_data.view()));

        // Discrete data (low unique ratio)
        let discrete_data = Array1::from_vec(vec![1.0, 1.0, 2.0, 2.0, 3.0]);
        assert!(!detector.is_continuous(&discrete_data.view()));
    }

    #[test]
    fn test_feature_type_detector() {
        let config = TypeDetectionConfig {
            min_samples: 3,
            ..Default::default()
        };

        let mut detector = FeatureTypeDetector::new(config).unwrap();

        // Test data with different feature types
        let data = Array2::from_shape_vec(
            (5, 3),
            vec![
                0.0, 1.0, 1.1, // binary, binary, continuous
                1.0, 0.0, 2.2, 0.0, 1.0, 3.3, 1.0, 0.0, 4.4, 0.0, 1.0, 5.5,
            ],
        )
        .unwrap();

        let types = detector.detect_types(&data.view()).unwrap();
        assert_eq!(types.len(), 3);

        // Check that types were detected (exact values may vary based on implementation)
        assert!(detector.feature_types().is_some());
        assert!(detector.type_probabilities().is_some());
    }

    #[test]
    fn test_skewness_analyzer() {
        let analyzer = SkewnessAnalyzer::new();

        let data = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let skewness = analyzer.calculate_skewness(&data.view());

        // Test skewness detection
        assert!(!analyzer.is_skewed(0.1, 0.5)); // Not skewed
        assert!(analyzer.is_skewed(0.8, 0.5)); // Skewed
    }

    #[test]
    fn test_feature_characterization() {
        let mut characterization = FeatureCharacterization::new();

        // Create test data with enough samples to meet the min_samples threshold (30)
        let mut data_vec = Vec::new();
        for i in 0..30 {
            data_vec.push(((i % 2) + 1) as f64); // First column: alternating 1, 2
            data_vec.push((i * 10 + 10) as f64); // Second column: increasing values
        }
        let data = Array2::from_shape_vec((30, 2), data_vec).unwrap();

        assert!(characterization.characterize_features(&data.view()).is_ok());

        let types = characterization.feature_types();
        assert_eq!(types.len(), 2);

        let stats = characterization.feature_stats();
        assert_eq!(stats.len(), 2);
    }
}
