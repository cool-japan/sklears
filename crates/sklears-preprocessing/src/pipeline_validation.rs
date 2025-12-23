//! Pipeline Validation Utilities
//!
//! Comprehensive validation framework for preprocessing pipelines,
//! ensuring correctness, compatibility, and optimal configuration.

use scirs2_core::ndarray::Array2;
use sklears_core::prelude::SklearsError;

/// Pipeline validation result
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether the pipeline is valid
    pub is_valid: bool,
    /// Validation warnings
    pub warnings: Vec<ValidationWarning>,
    /// Validation errors
    pub errors: Vec<ValidationError>,
    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,
    /// Estimated memory usage (bytes)
    pub estimated_memory: Option<usize>,
    /// Estimated computation time (milliseconds)
    pub estimated_time: Option<f64>,
}

/// Validation warning
#[derive(Debug, Clone)]
pub struct ValidationWarning {
    pub step: String,
    pub message: String,
    pub severity: WarningSeverity,
}

/// Warning severity levels
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WarningSeverity {
    Low,
    Medium,
    High,
}

/// Validation error
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub step: String,
    pub message: String,
    pub error_type: ValidationErrorType,
}

/// Validation error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationErrorType {
    IncompatibleDimensions,
    MissingRequirement,
    InvalidConfiguration,
    DataTypeMismatch,
    ResourceExceeded,
}

/// Performance recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    pub category: RecommendationCategory,
    pub message: String,
    pub expected_improvement: Option<f64>,
}

/// Recommendation categories
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RecommendationCategory {
    OrderOptimization,
    ParallelProcessing,
    MemoryEfficiency,
    ComputationEfficiency,
    DataQuality,
}

/// Pipeline validator configuration
#[derive(Debug, Clone)]
pub struct PipelineValidatorConfig {
    /// Maximum allowed memory usage (bytes)
    pub max_memory: Option<usize>,
    /// Maximum allowed computation time (milliseconds)
    pub max_time: Option<f64>,
    /// Check for redundant transformations
    pub check_redundancy: bool,
    /// Check for optimal ordering
    pub check_ordering: bool,
    /// Validate data compatibility
    pub validate_data: bool,
}

impl Default for PipelineValidatorConfig {
    fn default() -> Self {
        Self {
            max_memory: Some(8 * 1024 * 1024 * 1024), // 8GB
            max_time: Some(60000.0),                  // 60 seconds
            check_redundancy: true,
            check_ordering: true,
            validate_data: true,
        }
    }
}

/// Pipeline validator
pub struct PipelineValidator {
    config: PipelineValidatorConfig,
}

impl PipelineValidator {
    /// Create a new pipeline validator
    pub fn new() -> Self {
        Self {
            config: PipelineValidatorConfig::default(),
        }
    }

    /// Create a validator with custom configuration
    pub fn with_config(config: PipelineValidatorConfig) -> Self {
        Self { config }
    }

    /// Validate a preprocessing pipeline
    pub fn validate(
        &self,
        steps: &[String],
        sample_data: Option<&Array2<f64>>,
    ) -> Result<ValidationResult, SklearsError> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut recommendations = Vec::new();

        // Check for empty pipeline
        if steps.is_empty() {
            errors.push(ValidationError {
                step: "pipeline".to_string(),
                message: "Pipeline has no steps".to_string(),
                error_type: ValidationErrorType::InvalidConfiguration,
            });
        }

        // Check for redundant transformations
        if self.config.check_redundancy {
            let redundant = self.check_redundancy(steps);
            for (step1, step2) in redundant {
                warnings.push(ValidationWarning {
                    step: step1.clone(),
                    message: format!("Redundant with step: {}", step2),
                    severity: WarningSeverity::Medium,
                });
            }
        }

        // Check for optimal ordering
        if self.config.check_ordering {
            let ordering_issues = self.check_ordering(steps);
            for (step, suggestion) in ordering_issues {
                recommendations.push(PerformanceRecommendation {
                    category: RecommendationCategory::OrderOptimization,
                    message: format!("Step '{}': {}", step, suggestion),
                    expected_improvement: Some(1.5), // Estimated 50% improvement
                });
            }
        }

        // Validate data if provided
        if self.config.validate_data {
            if let Some(data) = sample_data {
                let data_issues = self.validate_data(data, steps);
                errors.extend(data_issues);
            }
        }

        // Estimate resource usage
        let estimated_memory = self.estimate_memory(steps, sample_data);
        let estimated_time = self.estimate_time(steps, sample_data);

        // Check resource limits
        if let (Some(est_mem), Some(max_mem)) = (estimated_memory, self.config.max_memory) {
            if est_mem > max_mem {
                errors.push(ValidationError {
                    step: "pipeline".to_string(),
                    message: format!(
                        "Estimated memory usage ({} bytes) exceeds limit ({} bytes)",
                        est_mem, max_mem
                    ),
                    error_type: ValidationErrorType::ResourceExceeded,
                });
            }
        }

        if let (Some(est_time), Some(max_time)) = (estimated_time, self.config.max_time) {
            if est_time > max_time {
                warnings.push(ValidationWarning {
                    step: "pipeline".to_string(),
                    message: format!(
                        "Estimated computation time ({:.2}ms) may exceed limit ({:.2}ms)",
                        est_time, max_time
                    ),
                    severity: WarningSeverity::High,
                });
            }
        }

        // Add general recommendations
        if steps.len() > 10 {
            recommendations.push(PerformanceRecommendation {
                category: RecommendationCategory::ComputationEfficiency,
                message: "Consider using FeatureUnion or ColumnTransformer for parallel processing"
                    .to_string(),
                expected_improvement: Some(2.0), // 2x improvement
            });
        }

        let is_valid = errors.is_empty();

        Ok(ValidationResult {
            is_valid,
            warnings,
            errors,
            recommendations,
            estimated_memory,
            estimated_time,
        })
    }

    /// Check for redundant transformations
    fn check_redundancy(&self, steps: &[String]) -> Vec<(String, String)> {
        let mut redundant = Vec::new();

        // Check for duplicate scaling steps
        let scaling_steps: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| {
                s.contains("Scaler")
                    || s.contains("Normalizer")
                    || s.contains("StandardScaler")
                    || s.contains("MinMaxScaler")
            })
            .collect();

        if scaling_steps.len() > 1 {
            for i in 1..scaling_steps.len() {
                redundant.push((scaling_steps[i].1.clone(), scaling_steps[0].1.clone()));
            }
        }

        // Check for duplicate imputation steps
        let imputation_steps: Vec<_> = steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.contains("Imputer"))
            .collect();

        if imputation_steps.len() > 1 {
            for i in 1..imputation_steps.len() {
                redundant.push((imputation_steps[i].1.clone(), imputation_steps[0].1.clone()));
            }
        }

        redundant
    }

    /// Check for suboptimal ordering
    fn check_ordering(&self, steps: &[String]) -> Vec<(String, String)> {
        let mut issues = Vec::new();

        for (i, step) in steps.iter().enumerate() {
            // Imputation should come before scaling
            if step.contains("Scaler") || step.contains("Normalizer") {
                if steps[..i].iter().any(|s| s.contains("Imputer")) {
                    // Good order
                } else if steps[i..].iter().any(|s| s.contains("Imputer")) {
                    issues.push((
                        step.clone(),
                        "Consider moving imputation before scaling".to_string(),
                    ));
                }
            }

            // Feature selection should come after feature generation
            if (step.contains("FeatureSelector") || step.contains("SelectK"))
                && !steps[..i]
                    .iter()
                    .any(|s| s.contains("PolynomialFeatures") || s.contains("FeatureUnion"))
                && steps[i..]
                    .iter()
                    .any(|s| s.contains("PolynomialFeatures") || s.contains("FeatureUnion"))
            {
                issues.push((
                    step.clone(),
                    "Consider moving feature selection after feature generation".to_string(),
                ));
            }

            // Encoding should come before numerical transformations
            if step.contains("Encoder")
                && steps[..i].iter().any(|s| {
                    s.contains("Scaler") || s.contains("Normalizer") || s.contains("Transformer")
                })
            {
                issues.push((
                    step.clone(),
                    "Consider moving encoding before numerical transformations".to_string(),
                ));
            }
        }

        issues
    }

    /// Validate data compatibility
    fn validate_data(&self, data: &Array2<f64>, steps: &[String]) -> Vec<ValidationError> {
        let mut errors = Vec::new();

        let (n_samples, n_features) = (data.nrows(), data.ncols());

        // Check for insufficient samples
        if n_samples < 2 {
            errors.push(ValidationError {
                step: "data".to_string(),
                message: "Insufficient samples (need at least 2)".to_string(),
                error_type: ValidationErrorType::InvalidConfiguration,
            });
        }

        // Check for steps that require minimum samples
        for step in steps {
            if step.contains("KNN") && n_samples < 5 {
                errors.push(ValidationError {
                    step: step.clone(),
                    message: format!(
                        "KNN-based methods require at least 5 samples, found {}",
                        n_samples
                    ),
                    error_type: ValidationErrorType::MissingRequirement,
                });
            }

            if step.contains("PCA") && n_samples < n_features {
                errors.push(ValidationError {
                    step: step.clone(),
                    message: format!(
                        "PCA requires n_samples >= n_features ({} < {})",
                        n_samples, n_features
                    ),
                    error_type: ValidationErrorType::InvalidConfiguration,
                });
            }
        }

        // Check for NaN values
        let has_nan = data.iter().any(|v| v.is_nan());
        if has_nan {
            let has_imputer = steps.iter().any(|s| s.contains("Imputer"));
            if !has_imputer {
                errors.push(ValidationError {
                    step: "data".to_string(),
                    message: "Data contains NaN values but no imputation step".to_string(),
                    error_type: ValidationErrorType::MissingRequirement,
                });
            }
        }

        errors
    }

    /// Estimate memory usage
    fn estimate_memory(
        &self,
        steps: &[String],
        sample_data: Option<&Array2<f64>>,
    ) -> Option<usize> {
        let base_size = if let Some(data) = sample_data {
            data.nrows() * data.ncols() * std::mem::size_of::<f64>()
        } else {
            0
        };

        let mut total = base_size;

        for step in steps {
            if step.contains("PolynomialFeatures") {
                // Polynomial features can significantly increase memory
                total = total.saturating_mul(3); // Rough estimate
            } else if step.contains("OneHotEncoder") {
                // One-hot encoding increases dimensions
                total = total.saturating_mul(2);
            } else {
                // Most transformations keep similar size
                total = total.saturating_add(base_size / 2);
            }
        }

        Some(total)
    }

    /// Estimate computation time
    fn estimate_time(&self, steps: &[String], sample_data: Option<&Array2<f64>>) -> Option<f64> {
        let n_operations = if let Some(data) = sample_data {
            (data.nrows() * data.ncols()) as f64
        } else {
            10000.0 // Default estimate
        };

        let mut total_time = 0.0;

        for step in steps {
            let step_time = if step.contains("KNN") {
                n_operations * 0.001 // KNN is O(n²) but we approximate
            } else if step.contains("PCA") {
                n_operations * 0.0005 // Matrix operations
            } else if step.contains("PolynomialFeatures") {
                n_operations * 0.0002
            } else {
                n_operations * 0.00001 // Simple operations
            };

            total_time += step_time;
        }

        Some(total_time)
    }
}

impl Default for PipelineValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationResult {
    /// Print a summary of the validation result
    pub fn print_summary(&self) {
        println!("Pipeline Validation Result");
        println!("==========================");
        println!(
            "Status: {}",
            if self.is_valid { "VALID" } else { "INVALID" }
        );
        println!();

        if !self.errors.is_empty() {
            println!("Errors: {}", self.errors.len());
            for error in &self.errors {
                println!("  [ERROR] {}: {}", error.step, error.message);
            }
            println!();
        }

        if !self.warnings.is_empty() {
            println!("Warnings: {}", self.warnings.len());
            for warning in &self.warnings {
                let severity = match warning.severity {
                    WarningSeverity::Low => "LOW",
                    WarningSeverity::Medium => "MEDIUM",
                    WarningSeverity::High => "HIGH",
                };
                println!("  [{}] {}: {}", severity, warning.step, warning.message);
            }
            println!();
        }

        if !self.recommendations.is_empty() {
            println!("Recommendations: {}", self.recommendations.len());
            for rec in &self.recommendations {
                let improvement = if let Some(imp) = rec.expected_improvement {
                    format!(" (expected {:.1}x improvement)", imp)
                } else {
                    String::new()
                };
                println!("  [RECOMMEND] {}{}", rec.message, improvement);
            }
            println!();
        }

        if let Some(mem) = self.estimated_memory {
            println!("Estimated Memory: {:.2} MB", mem as f64 / 1024.0 / 1024.0);
        }

        if let Some(time) = self.estimated_time {
            println!("Estimated Time: {:.2} ms", time);
        }
    }

    /// Get high severity warnings
    pub fn high_severity_warnings(&self) -> Vec<&ValidationWarning> {
        self.warnings
            .iter()
            .filter(|w| w.severity == WarningSeverity::High)
            .collect()
    }

    /// Get errors of specific type
    pub fn errors_of_type(&self, error_type: ValidationErrorType) -> Vec<&ValidationError> {
        self.errors
            .iter()
            .filter(|e| e.error_type == error_type)
            .collect()
    }

    /// Get recommendations by category
    pub fn recommendations_by_category(
        &self,
        category: RecommendationCategory,
    ) -> Vec<&PerformanceRecommendation> {
        self.recommendations
            .iter()
            .filter(|r| r.category == category)
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
    fn test_pipeline_validator_empty() {
        let validator = PipelineValidator::new();
        let result = validator.validate(&[], None).unwrap();

        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_validator_redundancy() {
        let validator = PipelineValidator::new();
        let steps = vec!["StandardScaler".to_string(), "MinMaxScaler".to_string()];

        let result = validator.validate(&steps, None).unwrap();

        // Should warn about redundant scaling
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_pipeline_validator_ordering() {
        let validator = PipelineValidator::new();
        let steps = vec![
            "StandardScaler".to_string(),
            "SimpleImputer".to_string(), // Imputation should come first
        ];

        let result = validator.validate(&steps, None).unwrap();

        // Should recommend reordering
        assert!(!result.recommendations.is_empty());
    }

    #[test]
    fn test_pipeline_validator_data() {
        let data = generate_test_data(100, 10, 42);
        let validator = PipelineValidator::new();
        let steps = vec!["StandardScaler".to_string()];

        let result = validator.validate(&steps, Some(&data)).unwrap();

        assert!(result.is_valid);
    }

    #[test]
    fn test_pipeline_validator_insufficient_samples() {
        let data = Array2::from_elem((1, 5), 1.0);
        let validator = PipelineValidator::new();
        let steps = vec!["KNNImputer".to_string()];

        let result = validator.validate(&steps, Some(&data)).unwrap();

        // Should error on insufficient samples
        assert!(!result.is_valid);
        assert!(!result.errors.is_empty());
    }

    #[test]
    fn test_pipeline_validator_nan_without_imputer() {
        let mut data = generate_test_data(50, 5, 123);
        data[[0, 0]] = f64::NAN;

        let validator = PipelineValidator::new();
        let steps = vec!["StandardScaler".to_string()];

        let result = validator.validate(&steps, Some(&data)).unwrap();

        // Should error on NaN without imputation
        assert!(!result.is_valid);
    }

    #[test]
    fn test_memory_estimation() {
        let data = generate_test_data(1000, 100, 456);
        let validator = PipelineValidator::new();
        let steps = vec![
            "StandardScaler".to_string(),
            "PolynomialFeatures".to_string(),
        ];

        let result = validator.validate(&steps, Some(&data)).unwrap();

        assert!(result.estimated_memory.is_some());
        assert!(result.estimated_memory.unwrap() > 0);
    }

    #[test]
    fn test_time_estimation() {
        let data = generate_test_data(1000, 50, 789);
        let validator = PipelineValidator::new();
        let steps = vec!["StandardScaler".to_string(), "PCA".to_string()];

        let result = validator.validate(&steps, Some(&data)).unwrap();

        assert!(result.estimated_time.is_some());
        assert!(result.estimated_time.unwrap() > 0.0);
    }

    #[test]
    fn test_validation_result_filtering() {
        let validator = PipelineValidator::new();
        let steps = vec![
            "StandardScaler".to_string(),
            "MinMaxScaler".to_string(),
            "SimpleImputer".to_string(),
        ];

        let result = validator.validate(&steps, None).unwrap();

        let high_warnings = result.high_severity_warnings();
        assert!(high_warnings.len() <= result.warnings.len());
    }
}
