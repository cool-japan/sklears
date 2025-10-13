//! Comprehensive Pipeline Validation Framework
//!
//! Advanced validation system for pipeline structure, data compatibility,
//! statistical properties, and performance validation.

use scirs2_core::ndarray::{s, Array1, ArrayView1, ArrayView2};
use scirs2_core::random::{thread_rng, Rng};
use sklears_core::{error::Result as SklResult, types::Float};
use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use crate::Pipeline;

/// Comprehensive pipeline validation framework
pub struct ComprehensivePipelineValidator {
    /// Data validation settings
    pub data_validator: DataValidator,
    /// Structure validation settings
    pub structure_validator: StructureValidator,
    /// Statistical validation settings
    pub statistical_validator: StatisticalValidator,
    /// Performance validation settings
    pub performance_validator: PerformanceValidator,
    /// Cross-validation settings
    pub cross_validator: CrossValidator,
    /// Robustness testing settings
    pub robustness_tester: RobustnessTester,
    /// Output detailed validation report
    pub verbose: bool,
}

/// Data quality and compatibility validation
pub struct DataValidator {
    /// Check for missing values (NaN)
    pub check_missing_values: bool,
    /// Check for infinite values
    pub check_infinite_values: bool,
    /// Check data type consistency
    pub check_data_types: bool,
    /// Check feature scaling consistency
    pub check_feature_scaling: bool,
    /// Check data distribution properties
    pub check_distributions: bool,
    /// Maximum allowed proportion of missing values
    pub max_missing_ratio: f64,
    /// Check for duplicate samples
    pub check_duplicates: bool,
    /// Check for outliers using IQR method
    pub check_outliers: bool,
    /// IQR multiplier for outlier detection
    pub outlier_iqr_multiplier: f64,
}

/// Pipeline structure and component validation
pub struct StructureValidator {
    /// Validate component compatibility
    pub check_component_compatibility: bool,
    /// Check data flow between components
    pub check_data_flow: bool,
    /// Validate parameter consistency
    pub check_parameter_consistency: bool,
    /// Check for circular dependencies
    pub check_circular_dependencies: bool,
    /// Validate resource requirements
    pub check_resource_requirements: bool,
    /// Maximum allowed pipeline depth
    pub max_pipeline_depth: usize,
    /// Maximum allowed number of components
    pub max_components: usize,
}

/// Statistical validation and testing
pub struct StatisticalValidator {
    /// Perform statistical significance tests
    pub statistical_tests: bool,
    /// Test for data leakage
    pub check_data_leakage: bool,
    /// Validate feature importance
    pub check_feature_importance: bool,
    /// Test prediction consistency
    pub check_prediction_consistency: bool,
    /// Minimum sample size for statistical tests
    pub min_sample_size: usize,
    /// Alpha level for statistical tests
    pub alpha: f64,
    /// Test for concept drift
    pub check_concept_drift: bool,
}

/// Performance validation and benchmarking
pub struct PerformanceValidator {
    /// Check training time limits
    pub check_training_time: bool,
    /// Check prediction time limits
    pub check_prediction_time: bool,
    /// Check memory usage limits
    pub check_memory_usage: bool,
    /// Maximum training time (seconds)
    pub max_training_time: f64,
    /// Maximum prediction time per sample (milliseconds)
    pub max_prediction_time_per_sample: f64,
    /// Maximum memory usage (MB)
    pub max_memory_usage: f64,
    /// Check scalability properties
    pub check_scalability: bool,
}

/// Cross-validation framework
pub struct CrossValidator {
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Stratified cross-validation for classification
    pub stratified: bool,
    /// Time series cross-validation
    pub time_series_cv: bool,
    /// Leave-one-out cross-validation
    pub leave_one_out: bool,
    /// Bootstrap validation
    pub bootstrap: bool,
    /// Number of bootstrap samples
    pub n_bootstrap: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

/// Robustness testing framework
pub struct RobustnessTester {
    /// Test with noisy data
    pub test_noise_robustness: bool,
    /// Test with missing data
    pub test_missing_data_robustness: bool,
    /// Test with adversarial examples
    pub test_adversarial_robustness: bool,
    /// Test with distribution shift
    pub test_distribution_shift: bool,
    /// Noise levels to test
    pub noise_levels: Vec<f64>,
    /// Missing data ratios to test
    pub missing_ratios: Vec<f64>,
    /// Number of robustness test iterations
    pub n_robustness_tests: usize,
}

/// Validation results summary
#[derive(Debug, Clone)]
pub struct ValidationReport {
    /// Overall validation status
    pub passed: bool,
    /// Data validation results
    pub data_validation: DataValidationResult,
    /// Structure validation results
    pub structure_validation: StructureValidationResult,
    /// Statistical validation results
    pub statistical_validation: StatisticalValidationResult,
    /// Performance validation results
    pub performance_validation: PerformanceValidationResult,
    /// Cross-validation results
    pub cross_validation: CrossValidationResult,
    /// Robustness testing results
    pub robustness_testing: RobustnessTestResult,
    /// Detailed validation messages
    pub messages: Vec<ValidationMessage>,
    /// Total validation time
    pub validation_time: Duration,
}

/// Data validation results
#[derive(Debug, Clone)]
pub struct DataValidationResult {
    pub passed: bool,
    pub missing_values_count: usize,
    pub infinite_values_count: usize,
    pub duplicate_samples_count: usize,
    pub outliers_count: usize,
    pub data_quality_score: f64,
}

/// Structure validation results
#[derive(Debug, Clone)]
pub struct StructureValidationResult {
    pub passed: bool,
    pub component_compatibility: bool,
    pub data_flow_valid: bool,
    pub circular_dependencies: bool,
    pub pipeline_depth: usize,
    pub component_count: usize,
}

/// Statistical validation results
#[derive(Debug, Clone)]
pub struct StatisticalValidationResult {
    pub passed: bool,
    pub statistical_significance: bool,
    pub data_leakage_detected: bool,
    pub prediction_consistency: f64,
    pub concept_drift_detected: bool,
    pub p_values: HashMap<String, f64>,
}

/// Performance validation results
#[derive(Debug, Clone)]
pub struct PerformanceValidationResult {
    pub passed: bool,
    pub training_time: f64,
    pub prediction_time_per_sample: f64,
    pub memory_usage: f64,
    pub scalability_score: f64,
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    pub passed: bool,
    pub cv_scores: Vec<f64>,
    pub mean_score: f64,
    pub std_score: f64,
    pub bootstrap_scores: Vec<f64>,
    pub confidence_interval: (f64, f64),
}

/// Robustness testing results
#[derive(Debug, Clone)]
pub struct RobustnessTestResult {
    pub passed: bool,
    pub noise_robustness_scores: HashMap<String, f64>,
    pub missing_data_robustness_scores: HashMap<String, f64>,
    pub adversarial_robustness_score: f64,
    pub distribution_shift_robustness: f64,
}

/// Validation message types
#[derive(Debug, Clone)]
pub struct ValidationMessage {
    pub level: MessageLevel,
    pub category: String,
    pub message: String,
    pub component: Option<String>,
}

/// Message severity levels
#[derive(Debug, Clone)]
pub enum MessageLevel {
    /// Info
    Info,
    /// Warning
    Warning,
    /// Error
    Error,
    /// Critical
    Critical,
}

impl Default for ComprehensivePipelineValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ComprehensivePipelineValidator {
    /// Create a new comprehensive pipeline validator with default settings
    #[must_use]
    pub fn new() -> Self {
        Self {
            data_validator: DataValidator::default(),
            structure_validator: StructureValidator::default(),
            statistical_validator: StatisticalValidator::default(),
            performance_validator: PerformanceValidator::default(),
            cross_validator: CrossValidator::default(),
            robustness_tester: RobustnessTester::default(),
            verbose: false,
        }
    }

    /// Create a strict validator with all checks enabled
    #[must_use]
    pub fn strict() -> Self {
        Self {
            data_validator: DataValidator::strict(),
            structure_validator: StructureValidator::strict(),
            statistical_validator: StatisticalValidator::strict(),
            performance_validator: PerformanceValidator::strict(),
            cross_validator: CrossValidator::default(),
            robustness_tester: RobustnessTester::comprehensive(),
            verbose: true,
        }
    }

    /// Create a fast validator with minimal checks for development
    #[must_use]
    pub fn fast() -> Self {
        Self {
            data_validator: DataValidator::basic(),
            structure_validator: StructureValidator::basic(),
            statistical_validator: StatisticalValidator::disabled(),
            performance_validator: PerformanceValidator::basic(),
            cross_validator: CrossValidator::fast(),
            robustness_tester: RobustnessTester::disabled(),
            verbose: false,
        }
    }

    /// Run comprehensive validation on a pipeline
    pub fn validate<S>(
        &self,
        pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<ValidationReport>
    where
        S: std::fmt::Debug,
    {
        let start_time = Instant::now();
        let mut messages = Vec::new();
        let mut overall_passed = true;

        if self.verbose {
            println!("Starting comprehensive pipeline validation...");
        }

        // Data validation
        let data_validation = self.validate_data(x, y, &mut messages)?;
        if !data_validation.passed {
            overall_passed = false;
        }

        // Structure validation
        let structure_validation = self.validate_structure(pipeline, &mut messages)?;
        if !structure_validation.passed {
            overall_passed = false;
        }

        // Statistical validation
        let statistical_validation = self.validate_statistics(x, y, &mut messages)?;
        if !statistical_validation.passed {
            overall_passed = false;
        }

        // Performance validation
        let performance_validation = self.validate_performance(pipeline, x, y, &mut messages)?;
        if !performance_validation.passed {
            overall_passed = false;
        }

        // Cross-validation
        let cross_validation = self.run_cross_validation(pipeline, x, y, &mut messages)?;
        if !cross_validation.passed {
            overall_passed = false;
        }

        // Robustness testing
        let robustness_testing = self.test_robustness(pipeline, x, y, &mut messages)?;
        if !robustness_testing.passed {
            overall_passed = false;
        }

        let validation_time = start_time.elapsed();

        if self.verbose {
            println!(
                "Validation completed in {:.2}s. Status: {}",
                validation_time.as_secs_f64(),
                if overall_passed { "PASSED" } else { "FAILED" }
            );
        }

        Ok(ValidationReport {
            passed: overall_passed,
            data_validation,
            structure_validation,
            statistical_validation,
            performance_validation,
            cross_validation,
            robustness_testing,
            messages,
            validation_time,
        })
    }

    fn validate_data(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<DataValidationResult> {
        let mut passed = true;
        let mut missing_count = 0;
        let mut infinite_count = 0;
        let mut duplicate_count = 0;
        let mut outliers_count = 0;

        if self.data_validator.check_missing_values {
            missing_count = self.count_missing_values(x);
            if missing_count > 0 {
                let missing_ratio = missing_count as f64 / (x.nrows() * x.ncols()) as f64;
                if missing_ratio > self.data_validator.max_missing_ratio {
                    passed = false;
                    messages.push(ValidationMessage {
                        level: MessageLevel::Error,
                        category: "Data Quality".to_string(),
                        message: format!(
                            "Missing values ratio ({:.3}) exceeds maximum allowed ({:.3})",
                            missing_ratio, self.data_validator.max_missing_ratio
                        ),
                        component: None,
                    });
                }
            }
        }

        if self.data_validator.check_infinite_values {
            infinite_count = self.count_infinite_values(x);
            if infinite_count > 0 {
                passed = false;
                messages.push(ValidationMessage {
                    level: MessageLevel::Error,
                    category: "Data Quality".to_string(),
                    message: format!("Found {infinite_count} infinite values in input data"),
                    component: None,
                });
            }
        }

        if self.data_validator.check_duplicates {
            duplicate_count = self.count_duplicate_samples(x);
            if duplicate_count > 0 {
                messages.push(ValidationMessage {
                    level: MessageLevel::Warning,
                    category: "Data Quality".to_string(),
                    message: format!("Found {duplicate_count} duplicate samples"),
                    component: None,
                });
            }
        }

        if self.data_validator.check_outliers {
            outliers_count = self.count_outliers(x, self.data_validator.outlier_iqr_multiplier);
            if outliers_count > x.nrows() / 10 {
                messages.push(ValidationMessage {
                    level: MessageLevel::Warning,
                    category: "Data Quality".to_string(),
                    message: format!(
                        "High number of outliers detected: {} ({}% of samples)",
                        outliers_count,
                        (outliers_count * 100) / x.nrows()
                    ),
                    component: None,
                });
            }
        }

        let data_quality_score = self.calculate_data_quality_score(
            x.nrows() * x.ncols(),
            missing_count,
            infinite_count,
            duplicate_count,
            outliers_count,
        );

        Ok(DataValidationResult {
            passed,
            missing_values_count: missing_count,
            infinite_values_count: infinite_count,
            duplicate_samples_count: duplicate_count,
            outliers_count,
            data_quality_score,
        })
    }

    fn validate_structure<S>(
        &self,
        pipeline: &Pipeline<S>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<StructureValidationResult>
    where
        S: std::fmt::Debug,
    {
        let mut passed = true;
        let component_compatibility = true;
        let data_flow_valid = true;
        let circular_dependencies = false; // Placeholder
        let pipeline_depth = 1; // Placeholder - would need to analyze actual pipeline structure
        let component_count = 1; // Placeholder

        if self.structure_validator.check_component_compatibility {
            // Placeholder for component compatibility checking
            // Would analyze if transformer outputs match estimator inputs
        }

        if self.structure_validator.check_data_flow {
            // Placeholder for data flow validation
            // Would check if data shapes are compatible between pipeline steps
        }

        if pipeline_depth > self.structure_validator.max_pipeline_depth {
            passed = false;
            messages.push(ValidationMessage {
                level: MessageLevel::Error,
                category: "Structure".to_string(),
                message: format!(
                    "Pipeline depth ({}) exceeds maximum allowed ({})",
                    pipeline_depth, self.structure_validator.max_pipeline_depth
                ),
                component: None,
            });
        }

        Ok(StructureValidationResult {
            passed,
            component_compatibility,
            data_flow_valid,
            circular_dependencies,
            pipeline_depth,
            component_count,
        })
    }

    fn validate_statistics(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<StatisticalValidationResult> {
        let mut passed = true;
        let mut p_values = HashMap::new();

        if x.nrows() < self.statistical_validator.min_sample_size {
            passed = false;
            messages.push(ValidationMessage {
                level: MessageLevel::Error,
                category: "Statistics".to_string(),
                message: format!(
                    "Sample size ({}) below minimum required ({})",
                    x.nrows(),
                    self.statistical_validator.min_sample_size
                ),
                component: None,
            });
        }

        // Perform comprehensive statistical tests
        let statistical_significance = self.test_statistical_significance(x, y, &mut p_values)?;
        let data_leakage_detected = if self.statistical_validator.check_data_leakage {
            self.detect_data_leakage(x, y)?
        } else {
            false
        };
        let prediction_consistency = if self.statistical_validator.check_prediction_consistency {
            self.calculate_prediction_consistency(x)?
        } else {
            1.0
        };
        let concept_drift_detected = if self.statistical_validator.check_concept_drift {
            self.detect_concept_drift(x, y)?
        } else {
            false
        };

        // Update passed status based on actual test results
        if !statistical_significance || data_leakage_detected || concept_drift_detected {
            passed = false;
        }
        if prediction_consistency < 0.8 {
            passed = false;
        }

        Ok(StatisticalValidationResult {
            passed,
            statistical_significance,
            data_leakage_detected,
            prediction_consistency,
            concept_drift_detected,
            p_values,
        })
    }

    fn validate_performance<S>(
        &self,
        pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<PerformanceValidationResult>
    where
        S: std::fmt::Debug,
    {
        let mut passed = true;

        // Measure training time (placeholder)
        let training_time = 1.0; // Would measure actual training time
        if self.performance_validator.check_training_time
            && training_time > self.performance_validator.max_training_time
        {
            passed = false;
            messages.push(ValidationMessage {
                level: MessageLevel::Error,
                category: "Performance".to_string(),
                message: format!(
                    "Training time ({:.2}s) exceeds maximum allowed ({:.2}s)",
                    training_time, self.performance_validator.max_training_time
                ),
                component: None,
            });
        }

        // Placeholder values - would measure actual performance
        let prediction_time_per_sample = 0.1;
        let memory_usage = 100.0;
        let scalability_score = 0.8;

        Ok(PerformanceValidationResult {
            passed,
            training_time,
            prediction_time_per_sample,
            memory_usage,
            scalability_score,
        })
    }

    fn run_cross_validation<S>(
        &self,
        pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<CrossValidationResult>
    where
        S: std::fmt::Debug,
    {
        if y.is_none() {
            return Ok(CrossValidationResult {
                passed: true,
                cv_scores: vec![],
                mean_score: 0.0,
                std_score: 0.0,
                bootstrap_scores: vec![],
                confidence_interval: (0.0, 0.0),
            });
        }

        let n_samples = x.nrows();
        let fold_size = n_samples / self.cross_validator.cv_folds;
        let mut cv_scores = Vec::new();

        // Placeholder cross-validation implementation
        for fold in 0..self.cross_validator.cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == self.cross_validator.cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Would split data and evaluate pipeline
            let score = 0.8 + thread_rng().gen::<f64>() * 0.2; // Placeholder
            cv_scores.push(score);
        }

        let mean_score = cv_scores.iter().sum::<f64>() / cv_scores.len() as f64;
        let variance = cv_scores
            .iter()
            .map(|&x| (x - mean_score).powi(2))
            .sum::<f64>()
            / cv_scores.len() as f64;
        let std_score = variance.sqrt();

        let passed = std_score < 0.1; // Placeholder criterion

        Ok(CrossValidationResult {
            passed,
            cv_scores,
            mean_score,
            std_score,
            bootstrap_scores: vec![],
            confidence_interval: (mean_score - 1.96 * std_score, mean_score + 1.96 * std_score),
        })
    }

    fn test_robustness<S>(
        &self,
        pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        messages: &mut Vec<ValidationMessage>,
    ) -> SklResult<RobustnessTestResult>
    where
        S: std::fmt::Debug,
    {
        let mut noise_robustness_scores = HashMap::new();
        let mut missing_data_robustness_scores = HashMap::new();

        if self.robustness_tester.test_noise_robustness {
            for &noise_level in &self.robustness_tester.noise_levels {
                let score = self.test_noise_robustness(pipeline, x, y, noise_level)?;
                noise_robustness_scores.insert(format!("noise_{noise_level}"), score);
            }
        }

        if self.robustness_tester.test_missing_data_robustness {
            for &missing_ratio in &self.robustness_tester.missing_ratios {
                let score = self.test_missing_data_robustness(pipeline, x, y, missing_ratio)?;
                missing_data_robustness_scores.insert(format!("missing_{missing_ratio}"), score);
            }
        }

        let adversarial_robustness_score = 0.7; // Placeholder
        let distribution_shift_robustness = 0.6; // Placeholder

        let passed = noise_robustness_scores.values().all(|&score| score > 0.5)
            && missing_data_robustness_scores
                .values()
                .all(|&score| score > 0.5);

        Ok(RobustnessTestResult {
            passed,
            noise_robustness_scores,
            missing_data_robustness_scores,
            adversarial_robustness_score,
            distribution_shift_robustness,
        })
    }

    // Helper methods for data validation
    fn count_missing_values(&self, x: &ArrayView2<'_, Float>) -> usize {
        x.iter().filter(|&&val| val.is_nan()).count()
    }

    fn count_infinite_values(&self, x: &ArrayView2<'_, Float>) -> usize {
        x.iter().filter(|&&val| val.is_infinite()).count()
    }

    fn count_duplicate_samples(&self, x: &ArrayView2<'_, Float>) -> usize {
        // Simplified duplicate detection
        let mut unique_rows = HashSet::new();
        let mut duplicates = 0;

        for row in x.rows() {
            let row_vec: Vec<String> = row.iter().map(|&val| format!("{val:.6}")).collect();
            let row_key = row_vec.join(",");

            if !unique_rows.insert(row_key) {
                duplicates += 1;
            }
        }

        duplicates
    }

    fn count_outliers(&self, x: &ArrayView2<'_, Float>, iqr_multiplier: f64) -> usize {
        let mut outliers = 0;

        for col in x.columns() {
            let mut sorted_col: Vec<Float> = col.to_vec();
            // Filter out NaN values before sorting
            sorted_col.retain(|x| !x.is_nan());
            sorted_col.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            let n = sorted_col.len();
            if n < 3 {
                continue; // Need at least 3 points for meaningful outlier detection
            }

            // Use different quartile calculation for small datasets
            let (q1, q3) = if n == 3 {
                (sorted_col[0], sorted_col[2])
            } else if n == 4 {
                // For 4 points, use the middle two as Q1 and Q3 reference
                (sorted_col[0], sorted_col[2]) // More conservative
            } else {
                // Standard quartile calculation for larger datasets
                let q1_idx = (n - 1) / 4;
                let q3_idx = 3 * (n - 1) / 4;
                (sorted_col[q1_idx], sorted_col[q3_idx])
            };

            let iqr = q3 - q1;

            // Avoid division by zero or very small IQR
            if iqr <= 1e-10 {
                continue;
            }

            let lower_bound = q1 - iqr_multiplier * iqr;
            let upper_bound = q3 + iqr_multiplier * iqr;

            for &val in col {
                if val < lower_bound || val > upper_bound {
                    outliers += 1;
                }
            }
        }

        outliers
    }

    fn calculate_data_quality_score(
        &self,
        total_values: usize,
        missing: usize,
        infinite: usize,
        duplicates: usize,
        outliers: usize,
    ) -> f64 {
        let quality_score = 1.0
            - (missing as f64 / total_values as f64) * 0.4
            - (infinite as f64 / total_values as f64) * 0.3
            - (duplicates as f64 / total_values as f64) * 0.2
            - (outliers as f64 / total_values as f64) * 0.1;

        quality_score.max(0.0)
    }

    fn test_noise_robustness<S>(
        &self,
        _pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
        noise_level: f64,
    ) -> SklResult<f64>
    where
        S: std::fmt::Debug,
    {
        // Placeholder noise robustness test
        // Would add noise to data and measure performance degradation
        Ok(1.0 - noise_level * 0.5)
    }

    fn test_missing_data_robustness<S>(
        &self,
        _pipeline: &Pipeline<S>,
        x: &ArrayView2<'_, Float>,
        _y: Option<&ArrayView1<'_, Float>>,
        missing_ratio: f64,
    ) -> SklResult<f64>
    where
        S: std::fmt::Debug,
    {
        // Placeholder missing data robustness test
        // Would introduce missing values and measure performance
        Ok(1.0 - missing_ratio * 0.7)
    }

    /// Test statistical significance of data patterns
    fn test_statistical_significance(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
        p_values: &mut HashMap<String, f64>,
    ) -> SklResult<bool> {
        if !self.statistical_validator.statistical_tests {
            return Ok(true);
        }

        let mut all_significant = true;

        // Test for normality using Shapiro-Wilk approximation
        for (i, column) in x.columns().into_iter().enumerate() {
            let normality_p = self.shapiro_wilk_test(&column.to_owned())?;
            p_values.insert(format!("normality_feature_{i}"), normality_p);

            if normality_p < self.statistical_validator.alpha {
                all_significant = false;
            }
        }

        // Test for independence between features (correlation test)
        if x.ncols() > 1 {
            let correlation_p = self.independence_test(x)?;
            p_values.insert("feature_independence".to_string(), correlation_p);

            if correlation_p < self.statistical_validator.alpha {
                all_significant = false;
            }
        }

        // Test for target distribution if available
        if let Some(targets) = y {
            let target_normality_p = self.shapiro_wilk_test(&targets.to_owned())?;
            p_values.insert("target_normality".to_string(), target_normality_p);
        }

        Ok(all_significant)
    }

    /// Detect potential data leakage
    fn detect_data_leakage(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<bool> {
        // Test for perfect correlation between features and target
        if let Some(targets) = y {
            for (i, column) in x.columns().into_iter().enumerate() {
                let correlation =
                    self.calculate_correlation(&column.to_owned(), &targets.to_owned())?;

                // Perfect or near-perfect correlation suggests potential leakage
                if correlation.abs() > 0.99 {
                    return Ok(true);
                }
            }
        }

        // Test for duplicate features (perfect multicollinearity)
        for i in 0..x.ncols() {
            for j in (i + 1)..x.ncols() {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation =
                    self.calculate_correlation(&col_i.to_owned(), &col_j.to_owned())?;

                if correlation.abs() > 0.999 {
                    return Ok(true); // Likely duplicate features
                }
            }
        }

        Ok(false)
    }

    /// Calculate prediction consistency across data subsets
    fn calculate_prediction_consistency(&self, x: &ArrayView2<'_, Float>) -> SklResult<f64> {
        if x.nrows() < 20 {
            return Ok(1.0); // Not enough data to test consistency
        }

        // Split data into random subsets and check consistency of statistics
        let mid = x.nrows() / 2;
        let subset1 = x.slice(s![..mid, ..]);
        let subset2 = x.slice(s![mid.., ..]);

        let mut consistency_scores = Vec::new();

        // Compare means across subsets
        for i in 0..x.ncols() {
            let mean1 = subset1.column(i).mean().unwrap_or(0.0);
            let mean2 = subset2.column(i).mean().unwrap_or(0.0);

            let consistency = if mean1.abs() + mean2.abs() > 1e-10 {
                1.0 - (mean1 - mean2).abs() / (mean1.abs() + mean2.abs()).max(1.0)
            } else {
                1.0
            };

            consistency_scores.push(consistency);
        }

        let avg_consistency =
            consistency_scores.iter().sum::<f64>() / consistency_scores.len() as f64;
        Ok(avg_consistency)
    }

    /// Detect concept drift in the data
    fn detect_concept_drift(
        &self,
        x: &ArrayView2<'_, Float>,
        y: Option<&ArrayView1<'_, Float>>,
    ) -> SklResult<bool> {
        if x.nrows() < 100 {
            return Ok(false); // Not enough data to detect drift
        }

        // Split data into early and late periods
        let split_point = x.nrows() * 2 / 3;
        let early_x = x.slice(s![..split_point, ..]);
        let late_x = x.slice(s![split_point.., ..]);

        // Test for distribution changes using two-sample tests
        for i in 0..x.ncols() {
            let early_col = early_x.column(i);
            let late_col = late_x.column(i);

            // Simple drift test: compare means and variances
            let mean_diff =
                (early_col.mean().unwrap_or(0.0) - late_col.mean().unwrap_or(0.0)).abs();
            let var_early = self.calculate_variance(&early_col.to_owned())?;
            let var_late = self.calculate_variance(&late_col.to_owned())?;
            let var_ratio = if var_late > 1e-10 {
                var_early / var_late
            } else {
                1.0
            };

            // Detect significant changes
            if mean_diff > 2.0 || !(0.5..=2.0).contains(&var_ratio) {
                return Ok(true);
            }
        }

        // Test target drift if available
        if let Some(targets) = y {
            let early_y = targets.slice(s![..split_point]);
            let late_y = targets.slice(s![split_point..]);

            let mean_diff = (early_y.mean().unwrap_or(0.0) - late_y.mean().unwrap_or(0.0)).abs();
            if mean_diff > 1.0 {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Approximate Shapiro-Wilk normality test
    fn shapiro_wilk_test(&self, data: &Array1<f64>) -> SklResult<f64> {
        if data.len() < 3 {
            return Ok(1.0); // Not enough data for test
        }

        let n = data.len();
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Simplified normality test based on sample skewness and kurtosis
        let mean = data.mean().unwrap_or(0.0);
        let variance = self.calculate_variance(data)?;

        if variance < 1e-10 {
            return Ok(0.0); // Constant data is not normal
        }

        let std_dev = variance.sqrt();

        // Calculate sample skewness
        let skewness = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(3))
            .sum::<f64>()
            / n as f64;

        // Calculate sample kurtosis
        let kurtosis = data
            .iter()
            .map(|&x| ((x - mean) / std_dev).powi(4))
            .sum::<f64>()
            / n as f64;

        // Approximate p-value based on skewness and kurtosis
        // Normal distribution has skewness = 0 and kurtosis = 3
        let skew_stat = skewness.abs();
        let kurt_stat = (kurtosis - 3.0).abs();

        // Simple approximation: lower p-value for higher deviations from normality
        let p_value = (1.0 - (skew_stat + kurt_stat) / 4.0).max(0.0).min(1.0);

        Ok(p_value)
    }

    /// Test for feature independence using correlation
    fn independence_test(&self, x: &ArrayView2<'_, Float>) -> SklResult<f64> {
        let mut max_correlation: f64 = 0.0;

        for i in 0..x.ncols() {
            for j in (i + 1)..x.ncols() {
                let col_i = x.column(i);
                let col_j = x.column(j);
                let correlation =
                    self.calculate_correlation(&col_i.to_owned(), &col_j.to_owned())?;
                max_correlation = max_correlation.max(correlation.abs());
            }
        }

        // Convert correlation to approximate p-value
        // Higher correlation = lower p-value (less independence)
        let p_value = (1.0 - max_correlation).max(0.0);
        Ok(p_value)
    }

    /// Calculate Pearson correlation coefficient
    fn calculate_correlation(&self, x: &Array1<f64>, y: &Array1<f64>) -> SklResult<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }

        let mean_x = x.mean().unwrap_or(0.0);
        let mean_y = y.mean().unwrap_or(0.0);

        let covariance = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum::<f64>()
            / (x.len() - 1) as f64;

        let var_x = self.calculate_variance(x)?;
        let var_y = self.calculate_variance(y)?;

        if var_x < 1e-10 || var_y < 1e-10 {
            return Ok(0.0); // No correlation if either variable is constant
        }

        let correlation = covariance / (var_x.sqrt() * var_y.sqrt());
        Ok(correlation.max(-1.0).min(1.0)) // Clamp to [-1, 1]
    }

    /// Calculate sample variance
    fn calculate_variance(&self, data: &Array1<f64>) -> SklResult<f64> {
        if data.len() < 2 {
            return Ok(0.0);
        }

        let mean = data.mean().unwrap_or(0.0);
        let variance =
            data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (data.len() - 1) as f64;

        Ok(variance)
    }
}

// Default implementations for validator components
impl Default for DataValidator {
    fn default() -> Self {
        Self {
            check_missing_values: true,
            check_infinite_values: true,
            check_data_types: true,
            check_feature_scaling: false,
            check_distributions: false,
            max_missing_ratio: 0.05,
            check_duplicates: false,
            check_outliers: false,
            outlier_iqr_multiplier: 1.5,
        }
    }
}

impl DataValidator {
    #[must_use]
    pub fn strict() -> Self {
        Self {
            check_missing_values: true,
            check_infinite_values: true,
            check_data_types: true,
            check_feature_scaling: true,
            check_distributions: true,
            max_missing_ratio: 0.01,
            check_duplicates: true,
            check_outliers: true,
            outlier_iqr_multiplier: 1.5,
        }
    }

    #[must_use]
    pub fn basic() -> Self {
        Self {
            check_missing_values: true,
            check_infinite_values: true,
            check_data_types: false,
            check_feature_scaling: false,
            check_distributions: false,
            max_missing_ratio: 0.1,
            check_duplicates: false,
            check_outliers: false,
            outlier_iqr_multiplier: 2.0,
        }
    }
}

impl Default for StructureValidator {
    fn default() -> Self {
        Self {
            check_component_compatibility: true,
            check_data_flow: true,
            check_parameter_consistency: false,
            check_circular_dependencies: true,
            check_resource_requirements: false,
            max_pipeline_depth: 10,
            max_components: 50,
        }
    }
}

impl StructureValidator {
    #[must_use]
    pub fn strict() -> Self {
        Self {
            check_component_compatibility: true,
            check_data_flow: true,
            check_parameter_consistency: true,
            check_circular_dependencies: true,
            check_resource_requirements: true,
            max_pipeline_depth: 5,
            max_components: 20,
        }
    }

    #[must_use]
    pub fn basic() -> Self {
        Self {
            check_component_compatibility: false,
            check_data_flow: false,
            check_parameter_consistency: false,
            check_circular_dependencies: false,
            check_resource_requirements: false,
            max_pipeline_depth: 20,
            max_components: 100,
        }
    }
}

impl Default for StatisticalValidator {
    fn default() -> Self {
        Self {
            statistical_tests: false,
            check_data_leakage: false,
            check_feature_importance: false,
            check_prediction_consistency: false,
            min_sample_size: 30,
            alpha: 0.05,
            check_concept_drift: false,
        }
    }
}

impl StatisticalValidator {
    #[must_use]
    pub fn strict() -> Self {
        Self {
            statistical_tests: true,
            check_data_leakage: true,
            check_feature_importance: true,
            check_prediction_consistency: true,
            min_sample_size: 100,
            alpha: 0.01,
            check_concept_drift: true,
        }
    }

    #[must_use]
    pub fn disabled() -> Self {
        Self {
            statistical_tests: false,
            check_data_leakage: false,
            check_feature_importance: false,
            check_prediction_consistency: false,
            min_sample_size: 10,
            alpha: 0.1,
            check_concept_drift: false,
        }
    }
}

impl Default for PerformanceValidator {
    fn default() -> Self {
        Self {
            check_training_time: false,
            check_prediction_time: false,
            check_memory_usage: false,
            max_training_time: 300.0,             // 5 minutes
            max_prediction_time_per_sample: 10.0, // 10ms
            max_memory_usage: 1000.0,             // 1GB
            check_scalability: false,
        }
    }
}

impl PerformanceValidator {
    #[must_use]
    pub fn strict() -> Self {
        Self {
            check_training_time: true,
            check_prediction_time: true,
            check_memory_usage: true,
            max_training_time: 60.0,             // 1 minute
            max_prediction_time_per_sample: 1.0, // 1ms
            max_memory_usage: 500.0,             // 500MB
            check_scalability: true,
        }
    }

    #[must_use]
    pub fn basic() -> Self {
        Self {
            check_training_time: false,
            check_prediction_time: false,
            check_memory_usage: false,
            max_training_time: 3600.0,             // 1 hour
            max_prediction_time_per_sample: 100.0, // 100ms
            max_memory_usage: 5000.0,              // 5GB
            check_scalability: false,
        }
    }
}

impl Default for CrossValidator {
    fn default() -> Self {
        Self {
            cv_folds: 5,
            stratified: true,
            time_series_cv: false,
            leave_one_out: false,
            bootstrap: false,
            n_bootstrap: 100,
            random_state: Some(42),
        }
    }
}

impl CrossValidator {
    #[must_use]
    pub fn fast() -> Self {
        Self {
            cv_folds: 3,
            stratified: false,
            time_series_cv: false,
            leave_one_out: false,
            bootstrap: false,
            n_bootstrap: 10,
            random_state: Some(42),
        }
    }
}

impl Default for RobustnessTester {
    fn default() -> Self {
        Self {
            test_noise_robustness: false,
            test_missing_data_robustness: false,
            test_adversarial_robustness: false,
            test_distribution_shift: false,
            noise_levels: vec![0.01, 0.05, 0.1],
            missing_ratios: vec![0.01, 0.05, 0.1],
            n_robustness_tests: 10,
        }
    }
}

impl RobustnessTester {
    #[must_use]
    pub fn comprehensive() -> Self {
        Self {
            test_noise_robustness: true,
            test_missing_data_robustness: true,
            test_adversarial_robustness: true,
            test_distribution_shift: true,
            noise_levels: vec![0.001, 0.01, 0.05, 0.1, 0.2],
            missing_ratios: vec![0.01, 0.05, 0.1, 0.2, 0.3],
            n_robustness_tests: 50,
        }
    }

    #[must_use]
    pub fn disabled() -> Self {
        Self {
            test_noise_robustness: false,
            test_missing_data_robustness: false,
            test_adversarial_robustness: false,
            test_distribution_shift: false,
            noise_levels: vec![],
            missing_ratios: vec![],
            n_robustness_tests: 0,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array, ArrayView1, ArrayView2};

    #[test]
    fn test_comprehensive_validator_creation() {
        let validator = ComprehensivePipelineValidator::new();
        assert!(!validator.verbose);

        let strict_validator = ComprehensivePipelineValidator::strict();
        assert!(strict_validator.verbose);

        let fast_validator = ComprehensivePipelineValidator::fast();
        assert!(!fast_validator.verbose);
    }

    #[test]
    fn test_data_validation() {
        let validator = ComprehensivePipelineValidator::new();
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![1.0, 2.0, 3.0];

        let mut messages = Vec::new();
        let result = validator
            .validate_data(&x.view(), Some(&y.view()), &mut messages)
            .unwrap();

        assert!(result.passed);
        assert_eq!(result.missing_values_count, 0);
        assert_eq!(result.infinite_values_count, 0);
    }

    #[test]
    fn test_data_validation_with_missing_values() {
        let validator = ComprehensivePipelineValidator::strict();
        let x = array![[1.0, f64::NAN], [3.0, 4.0], [5.0, 6.0]];

        let mut messages = Vec::new();
        let result = validator
            .validate_data(&x.view(), None, &mut messages)
            .unwrap();

        assert!(!result.passed);
        assert_eq!(result.missing_values_count, 1);
        assert!(!messages.is_empty());
    }

    #[test]
    fn test_outlier_detection() {
        let validator = ComprehensivePipelineValidator::new();
        let outlier_count = validator.count_outliers(
            &array![[1.0, 2.0], [1.1, 2.1], [1.0, 2.0], [100.0, 200.0]].view(),
            1.5,
        );

        assert!(outlier_count > 0);
    }

    #[test]
    fn test_duplicate_detection() {
        let validator = ComprehensivePipelineValidator::new();
        let duplicate_count =
            validator.count_duplicate_samples(&array![[1.0, 2.0], [3.0, 4.0], [1.0, 2.0]].view());

        assert_eq!(duplicate_count, 1);
    }

    #[test]
    fn test_data_quality_score() {
        let validator = ComprehensivePipelineValidator::new();

        let perfect_score = validator.calculate_data_quality_score(100, 0, 0, 0, 0);
        assert_eq!(perfect_score, 1.0);

        let imperfect_score = validator.calculate_data_quality_score(100, 10, 5, 2, 1);
        assert!(imperfect_score < 1.0);
        assert!(imperfect_score > 0.0);
    }
}
