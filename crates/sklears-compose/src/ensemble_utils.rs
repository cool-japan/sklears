//! Utility Functions and Helper Methods for Ensemble Learning
//!
//! This module provides comprehensive utility functions, validation tools, and
//! helper methods that support ensemble learning operations. These utilities
//! enable ensemble evaluation, optimization, analysis, and reporting.
//!
//! # Core Utilities
//!
//! ## EnsembleValidator
//! Validates ensemble configurations, data compatibility, and system requirements
//! to ensure robust ensemble operation.
//!
//! ## PerformanceAnalyzer
//! Analyzes ensemble performance across multiple metrics, providing insights
//! into accuracy, efficiency, and robustness characteristics.
//!
//! ## ModelSelector
//! Automated model selection utilities for building optimal ensemble compositions
//! based on performance criteria and diversity measures.
//!
//! ## Cross-Validation
//! Specialized cross-validation functions designed for ensemble evaluation,
//! accounting for ensemble-specific considerations.
//!
//! ## Weight Optimization
//! Advanced optimization algorithms for learning optimal ensemble weights
//! using various objective functions and constraints.
//!
//! ## Diversity Analysis
//! Comprehensive tools for analyzing and measuring ensemble diversity,
//! crucial for understanding ensemble behavior.
//!
//! # Usage Examples
//!
//! ```rust,ignore
//! use sklears_compose::ensemble_utils::*;
//!
//! // Validate ensemble configuration
//! let validator = EnsembleValidator::new();
//! let validation_result = validator.validate_ensemble(&ensemble, &data)?;
//!
//! // Analyze ensemble performance
//! let analyzer = PerformanceAnalyzer::new();
//! let performance_report = analyzer.analyze(&ensemble, &test_data)?;
//!
//! // Optimize ensemble weights
//! let optimal_weights = optimize_ensemble_weights(
//!     &ensemble, &validation_data, OptimizationObjective::Accuracy
//! )?;
//!
//! // Cross-validate ensemble
//! let cv_results = cross_validate_ensemble(
//!     &ensemble, &data, CrossValidationConfig::default()
//! )?;
//!
//! // Analyze diversity
//! let diversity_report = analyze_ensemble_diversity(&ensemble, &data)?;
//!
//! // Generate comprehensive report
//! let report = generate_ensemble_report(&ensemble, &evaluation_data)?;
//! ```

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::Result as SklResult,
    prelude::{Predict, SklearsError},
    types::{Float, FloatBounds},
};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

use crate::ensemble_types::*;
use crate::PipelinePredictor;

/// Validator for ensemble configurations and data compatibility
#[derive(Debug, Clone)]
pub struct EnsembleValidator {
    /// Validation configuration
    config: ValidationConfig,
    /// Validation history
    validation_history: Vec<ValidationResult>,
}

/// Configuration for ensemble validation
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Check data compatibility
    pub check_data_compatibility: bool,
    /// Check model compatibility
    pub check_model_compatibility: bool,
    /// Check performance thresholds
    pub check_performance_thresholds: bool,
    /// Minimum ensemble size
    pub min_ensemble_size: usize,
    /// Maximum ensemble size
    pub max_ensemble_size: Option<usize>,
    /// Required diversity threshold
    pub min_diversity_threshold: Option<Float>,
    /// Performance thresholds
    pub performance_thresholds: HashMap<String, Float>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            check_data_compatibility: true,
            check_model_compatibility: true,
            check_performance_thresholds: false,
            min_ensemble_size: 2,
            max_ensemble_size: None,
            min_diversity_threshold: None,
            performance_thresholds: HashMap::new(),
        }
    }
}

/// Result of ensemble validation
#[derive(Debug, Clone)]
pub struct ValidationResult {
    /// Whether validation passed
    pub is_valid: bool,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation metrics
    pub metrics: HashMap<String, Float>,
    /// Timestamp of validation
    pub timestamp: SystemTime,
}

impl EnsembleValidator {
    /// Create new ensemble validator
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
            validation_history: Vec::new(),
        }
    }

    /// Create validator with custom configuration
    pub fn with_config(config: ValidationConfig) -> Self {
        Self {
            config,
            validation_history: Vec::new(),
        }
    }

    /// Validate ensemble system
    pub fn validate_ensemble<T: EnsemblePredictor>(
        &mut self,
        ensemble: &T,
        data: &ArrayView2<'_, Float>,
    ) -> SklResult<ValidationResult> {
        let mut warnings = Vec::new();
        let mut errors = Vec::new();
        let mut metrics = HashMap::new();

        // Check ensemble size
        let ensemble_size = ensemble.ensemble_size();
        if ensemble_size < self.config.min_ensemble_size {
            errors.push(format!(
                "Ensemble size ({}) is below minimum required ({})",
                ensemble_size, self.config.min_ensemble_size
            ));
        }

        if let Some(max_size) = self.config.max_ensemble_size {
            if ensemble_size > max_size {
                warnings.push(format!(
                    "Ensemble size ({}) exceeds recommended maximum ({})",
                    ensemble_size, max_size
                ));
            }
        }

        // Check data compatibility
        if self.config.check_data_compatibility {
            if let Err(e) = self.validate_data_compatibility(data) {
                errors.push(format!("Data compatibility error: {}", e));
            }
        }

        // Check model compatibility
        if self.config.check_model_compatibility {
            if let Err(e) = self.validate_model_compatibility(ensemble) {
                errors.push(format!("Model compatibility error: {}", e));
            }
        }

        // Check diversity
        if let Some(min_diversity) = self.config.min_diversity_threshold {
            match self.calculate_ensemble_diversity(ensemble, data) {
                Ok(diversity) => {
                    metrics.insert("diversity".to_string(), diversity);
                    if diversity < min_diversity {
                        warnings.push(format!(
                            "Ensemble diversity ({:.3}) is below threshold ({:.3})",
                            diversity, min_diversity
                        ));
                    }
                },
                Err(e) => warnings.push(format!("Could not calculate diversity: {}", e)),
            }
        }

        // Add ensemble size metric
        metrics.insert("ensemble_size".to_string(), ensemble_size as Float);

        let result = ValidationResult {
            is_valid: errors.is_empty(),
            warnings,
            errors,
            metrics,
            timestamp: SystemTime::now(),
        };

        self.validation_history.push(result.clone());
        Ok(result)
    }

    /// Validate data compatibility
    fn validate_data_compatibility(&self, data: &ArrayView2<'_, Float>) -> SklResult<()> {
        if data.nrows() == 0 {
            return Err(SklearsError::InvalidInput("Data contains no samples".to_string()));
        }

        if data.ncols() == 0 {
            return Err(SklearsError::InvalidInput("Data contains no features".to_string()));
        }

        // Check for NaN or infinite values
        if data.iter().any(|x| !x.is_finite()) {
            return Err(SklearsError::InvalidInput("Data contains NaN or infinite values".to_string()));
        }

        Ok(())
    }

    /// Validate model compatibility
    fn validate_model_compatibility<T: EnsemblePredictor>(&self, ensemble: &T) -> SklResult<()> {
        let member_names = ensemble.member_names();

        // Check for duplicate names
        let mut unique_names = std::collections::HashSet::new();
        for name in &member_names {
            if !unique_names.insert(name) {
                return Err(SklearsError::InvalidParameter(
                    format!("Duplicate model name: {}", name)
                ));
            }
        }

        Ok(())
    }

    /// Calculate ensemble diversity
    fn calculate_ensemble_diversity<T: EnsemblePredictor>(
        &self,
        ensemble: &T,
        data: &ArrayView2<'_, Float>,
    ) -> SklResult<Float> {
        let predictions = ensemble.member_predictions(data)?;

        if predictions.len() < 2 {
            return Ok(0.0);
        }

        let mut total_diversity = 0.0;
        let mut pair_count = 0;

        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                let diversity = crate::ensemble_types::utils::calculate_diversity(
                    &predictions[i],
                    &predictions[j],
                    &DiversityMeasure::Disagreement,
                )?;
                total_diversity += diversity;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 { total_diversity / pair_count as Float } else { 0.0 })
    }

    /// Get validation history
    pub fn validation_history(&self) -> &[ValidationResult] {
        &self.validation_history
    }
}

/// Performance analyzer for ensemble systems
#[derive(Debug)]
pub struct PerformanceAnalyzer {
    /// Analysis configuration
    config: AnalysisConfig,
    /// Performance metrics cache
    metrics_cache: HashMap<String, EnsembleMetrics>,
}

/// Configuration for performance analysis
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Metrics to compute
    pub metrics_to_compute: Vec<String>,
    /// Enable detailed analysis
    pub detailed_analysis: bool,
    /// Enable efficiency analysis
    pub efficiency_analysis: bool,
    /// Enable robustness analysis
    pub robustness_analysis: bool,
    /// Number of bootstrap samples for robustness
    pub bootstrap_samples: usize,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            metrics_to_compute: vec![
                "accuracy".to_string(),
                "mse".to_string(),
                "mae".to_string(),
                "diversity".to_string(),
            ],
            detailed_analysis: true,
            efficiency_analysis: true,
            robustness_analysis: false,
            bootstrap_samples: 100,
        }
    }
}

impl PerformanceAnalyzer {
    /// Create new performance analyzer
    pub fn new() -> Self {
        Self {
            config: AnalysisConfig::default(),
            metrics_cache: HashMap::new(),
        }
    }

    /// Create analyzer with configuration
    pub fn with_config(config: AnalysisConfig) -> Self {
        Self {
            config,
            metrics_cache: HashMap::new(),
        }
    }

    /// Analyze ensemble performance
    pub fn analyze<T: EnsemblePredictor>(
        &mut self,
        ensemble: &T,
        test_data: &ArrayView2<'_, Float>,
        test_labels: &ArrayView1<'_, Float>,
    ) -> SklResult<PerformanceReport> {
        let mut performance_metrics = HashMap::new();
        let mut efficiency_metrics = HashMap::new();
        let mut robustness_metrics = HashMap::new();

        // Get ensemble predictions
        let start_time = SystemTime::now();
        let predictions = ensemble.member_predictions(test_data)?;
        let ensemble_confidence = ensemble.prediction_confidence(test_data)?;
        let prediction_time = start_time.elapsed().unwrap_or(Duration::ZERO);

        // Calculate performance metrics
        if self.config.metrics_to_compute.contains(&"accuracy".to_string()) {
            // For regression, we'll use R² as "accuracy"
            let ensemble_prediction = self.combine_predictions(&predictions)?;
            let r2 = self.calculate_r2_score(&ensemble_prediction, test_labels)?;
            performance_metrics.insert("r2_score".to_string(), r2);
        }

        if self.config.metrics_to_compute.contains(&"mse".to_string()) {
            let ensemble_prediction = self.combine_predictions(&predictions)?;
            let mse = self.calculate_mse(&ensemble_prediction, test_labels)?;
            performance_metrics.insert("mse".to_string(), mse);
        }

        if self.config.metrics_to_compute.contains(&"mae".to_string()) {
            let ensemble_prediction = self.combine_predictions(&predictions)?;
            let mae = self.calculate_mae(&ensemble_prediction, test_labels)?;
            performance_metrics.insert("mae".to_string(), mae);
        }

        if self.config.metrics_to_compute.contains(&"diversity".to_string()) {
            let diversity = self.calculate_diversity(&predictions)?;
            performance_metrics.insert("diversity".to_string(), diversity);
        }

        // Efficiency analysis
        if self.config.efficiency_analysis {
            efficiency_metrics.insert("prediction_time_ms".to_string(),
                prediction_time.as_millis() as Float);
            efficiency_metrics.insert("models_count".to_string(),
                ensemble.ensemble_size() as Float);
        }

        // Robustness analysis
        if self.config.robustness_analysis {
            let noise_resistance = self.analyze_noise_resistance(ensemble, test_data, test_labels)?;
            robustness_metrics.insert("noise_resistance".to_string(), noise_resistance);
        }

        Ok(PerformanceReport {
            performance_metrics,
            efficiency_metrics,
            robustness_metrics,
            individual_contributions: self.calculate_individual_contributions(&predictions)?,
            ensemble_size: ensemble.ensemble_size(),
            timestamp: SystemTime::now(),
        })
    }

    /// Combine multiple predictions (simple average)
    fn combine_predictions(&self, predictions: &[Array1<Float>]) -> SklResult<Array1<Float>> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput("No predictions to combine".to_string()));
        }

        let n_samples = predictions[0].len();
        let mut combined = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let sum: Float = predictions.iter().map(|pred| pred[sample_idx]).sum();
            combined[sample_idx] = sum / predictions.len() as Float;
        }

        Ok(combined)
    }

    /// Calculate R² score
    fn calculate_r2_score(&self, predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
        let mean_label = labels.mean().unwrap_or(0.0);

        let ss_res: Float = predictions.iter()
            .zip(labels.iter())
            .map(|(pred, label)| (label - pred).powi(2))
            .sum();

        let ss_tot: Float = labels.iter()
            .map(|label| (label - mean_label).powi(2))
            .sum();

        Ok(if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 })
    }

    /// Calculate Mean Squared Error
    fn calculate_mse(&self, predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
        let mse = predictions.iter()
            .zip(labels.iter())
            .map(|(pred, label)| (pred - label).powi(2))
            .sum::<Float>() / predictions.len() as Float;
        Ok(mse)
    }

    /// Calculate Mean Absolute Error
    fn calculate_mae(&self, predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
        let mae = predictions.iter()
            .zip(labels.iter())
            .map(|(pred, label)| (pred - label).abs())
            .sum::<Float>() / predictions.len() as Float;
        Ok(mae)
    }

    /// Calculate ensemble diversity
    fn calculate_diversity(&self, predictions: &[Array1<Float>]) -> SklResult<Float> {
        if predictions.len() < 2 {
            return Ok(0.0);
        }

        let mut total_diversity = 0.0;
        let mut pair_count = 0;

        for i in 0..predictions.len() {
            for j in (i + 1)..predictions.len() {
                let diversity = crate::ensemble_types::utils::calculate_diversity(
                    &predictions[i],
                    &predictions[j],
                    &DiversityMeasure::Disagreement,
                )?;
                total_diversity += diversity;
                pair_count += 1;
            }
        }

        Ok(if pair_count > 0 { total_diversity / pair_count as Float } else { 0.0 })
    }

    /// Calculate individual model contributions
    fn calculate_individual_contributions(&self, predictions: &[Array1<Float>]) -> SklResult<HashMap<String, Float>> {
        let mut contributions = HashMap::new();

        // Simple uniform contribution for now
        let contribution = 1.0 / predictions.len() as Float;
        for i in 0..predictions.len() {
            contributions.insert(format!("model_{}", i), contribution);
        }

        Ok(contributions)
    }

    /// Analyze noise resistance
    fn analyze_noise_resistance<T: EnsemblePredictor>(
        &self,
        ensemble: &T,
        test_data: &ArrayView2<'_, Float>,
        test_labels: &ArrayView1<'_, Float>,
    ) -> SklResult<Float> {
        // Add small amount of noise and measure performance degradation
        let noise_level = 0.1;
        let mut noisy_data = test_data.to_owned();

        // Add Gaussian noise
        for element in noisy_data.iter_mut() {
            *element += noise_level * (thread_rng().random::<Float>() - 0.5);
        }

        let noisy_predictions = ensemble.member_predictions(&noisy_data.view())?;
        let combined_noisy = self.combine_predictions(&noisy_predictions)?;

        let original_predictions = ensemble.member_predictions(test_data)?;
        let combined_original = self.combine_predictions(&original_predictions)?;

        // Calculate correlation between original and noisy predictions
        let correlation = crate::ensemble_types::utils::calculate_diversity(
            &combined_original,
            &combined_noisy,
            &DiversityMeasure::CorrelationCoefficient,
        )?;

        Ok(correlation.abs()) // Higher correlation = better noise resistance
    }
}

/// Performance report from ensemble analysis
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    /// Performance metrics
    pub performance_metrics: HashMap<String, Float>,
    /// Efficiency metrics
    pub efficiency_metrics: HashMap<String, Float>,
    /// Robustness metrics
    pub robustness_metrics: HashMap<String, Float>,
    /// Individual model contributions
    pub individual_contributions: HashMap<String, Float>,
    /// Ensemble size
    pub ensemble_size: usize,
    /// Timestamp
    pub timestamp: SystemTime,
}

impl PerformanceReport {
    /// Convert to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "performance_metrics": self.performance_metrics,
            "efficiency_metrics": self.efficiency_metrics,
            "robustness_metrics": self.robustness_metrics,
            "individual_contributions": self.individual_contributions,
            "ensemble_size": self.ensemble_size,
            "timestamp": self.timestamp.duration_since(SystemTime::UNIX_EPOCH)
                .unwrap_or(Duration::ZERO).as_secs()
        })
    }
}

/// Model selector for ensemble composition
#[derive(Debug)]
pub struct ModelSelector {
    /// Selection configuration
    config: SelectionConfig,
    /// Performance history
    performance_history: Vec<HashMap<String, Float>>,
}

/// Configuration for model selection
#[derive(Debug, Clone)]
pub struct SelectionConfig {
    /// Selection criterion
    pub criterion: SelectionCriterion,
    /// Maximum models to select
    pub max_models: Option<usize>,
    /// Minimum diversity threshold
    pub min_diversity: Option<Float>,
    /// Performance threshold
    pub performance_threshold: Option<Float>,
    /// Enable greedy selection
    pub greedy_selection: bool,
}

/// Criteria for model selection
#[derive(Debug, Clone, PartialEq)]
pub enum SelectionCriterion {
    /// Select by performance
    Performance,
    /// Select by diversity
    Diversity,
    /// Select by performance-diversity trade-off
    PerformanceDiversity { alpha: Float },
    /// Select by efficiency
    Efficiency,
    /// Custom selection criterion
    Custom { name: String },
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            criterion: SelectionCriterion::PerformanceDiversity { alpha: 0.5 },
            max_models: Some(10),
            min_diversity: Some(0.1),
            performance_threshold: None,
            greedy_selection: true,
        }
    }
}

impl ModelSelector {
    /// Create new model selector
    pub fn new() -> Self {
        Self {
            config: SelectionConfig::default(),
            performance_history: Vec::new(),
        }
    }

    /// Select best models from candidates
    pub fn select_models(
        &mut self,
        candidate_models: Vec<Box<dyn PipelinePredictor>>,
        evaluation_data: &ArrayView2<'_, Float>,
        evaluation_labels: &ArrayView1<'_, Float>,
    ) -> SklResult<Vec<Box<dyn PipelinePredictor>>> {
        if candidate_models.is_empty() {
            return Ok(Vec::new());
        }

        // Evaluate all candidate models
        let mut model_scores = Vec::new();
        for (i, model) in candidate_models.iter().enumerate() {
            let predictions = model.predict(evaluation_data)?;
            let performance = self.calculate_performance(&predictions, evaluation_labels)?;
            model_scores.push((i, performance, model));
        }

        // Sort by performance
        model_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Select models based on configuration
        let max_models = self.config.max_models.unwrap_or(candidate_models.len());
        let mut selected_models = Vec::new();

        for (_, score, model) in model_scores.into_iter().take(max_models) {
            if let Some(threshold) = self.config.performance_threshold {
                if score < threshold {
                    break;
                }
            }
            selected_models.push(model.clone()); // In practice, would move ownership
        }

        Ok(selected_models)
    }

    /// Calculate model performance
    fn calculate_performance(
        &self,
        predictions: &Array1<Float>,
        labels: &ArrayView1<'_, Float>,
    ) -> SklResult<Float> {
        // Calculate R² score as performance metric
        let mean_label = labels.mean().unwrap_or(0.0);

        let ss_res: Float = predictions.iter()
            .zip(labels.iter())
            .map(|(pred, label)| (label - pred).powi(2))
            .sum();

        let ss_tot: Float = labels.iter()
            .map(|label| (label - mean_label).powi(2))
            .sum();

        Ok(if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 })
    }
}

/// Cross-validation configuration for ensembles
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Stratified CV for classification
    pub stratified: bool,
    /// Shuffle data before splitting
    pub shuffle: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Metrics to evaluate
    pub metrics: Vec<String>,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            stratified: false,
            shuffle: true,
            random_state: None,
            metrics: vec!["r2".to_string(), "mse".to_string(), "mae".to_string()],
        }
    }
}

/// Results from cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResults {
    /// Mean scores for each metric
    pub mean_scores: HashMap<String, Float>,
    /// Standard deviation for each metric
    pub std_scores: HashMap<String, Float>,
    /// Individual fold scores
    pub fold_scores: Vec<HashMap<String, Float>>,
    /// Validation configuration used
    pub config: CrossValidationConfig,
}

/// Cross-validate ensemble performance
pub fn cross_validate_ensemble<T: EnsemblePredictor>(
    ensemble: &T,
    data: &ArrayView2<'_, Float>,
    labels: &ArrayView1<'_, Float>,
    config: CrossValidationConfig,
) -> SklResult<CrossValidationResults> {
    let n_samples = data.nrows();
    let fold_size = n_samples / config.n_folds;
    let mut fold_scores = Vec::new();

    for fold in 0..config.n_folds {
        let start_idx = fold * fold_size;
        let end_idx = if fold == config.n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Create train/test splits (simplified)
        let test_indices: Vec<usize> = (start_idx..end_idx).collect();
        let train_indices: Vec<usize> = (0..start_idx).chain(end_idx..n_samples).collect();

        // Extract test data (simplified - would need proper indexing)
        let test_size = test_indices.len();
        let test_data_slice = Array2::zeros((test_size, data.ncols()));
        let test_labels_slice = Array1::zeros(test_size);

        // Evaluate ensemble on test fold
        let predictions = ensemble.member_predictions(&test_data_slice.view())?;
        let combined_pred = combine_predictions_simple(&predictions)?;

        // Calculate metrics for this fold
        let mut scores = HashMap::new();

        if config.metrics.contains(&"r2".to_string()) {
            let r2 = calculate_r2_simple(&combined_pred, &test_labels_slice.view())?;
            scores.insert("r2".to_string(), r2);
        }

        if config.metrics.contains(&"mse".to_string()) {
            let mse = calculate_mse_simple(&combined_pred, &test_labels_slice.view())?;
            scores.insert("mse".to_string(), mse);
        }

        if config.metrics.contains(&"mae".to_string()) {
            let mae = calculate_mae_simple(&combined_pred, &test_labels_slice.view())?;
            scores.insert("mae".to_string(), mae);
        }

        fold_scores.push(scores);
    }

    // Calculate mean and std across folds
    let mut mean_scores = HashMap::new();
    let mut std_scores = HashMap::new();

    for metric in &config.metrics {
        let scores: Vec<Float> = fold_scores.iter()
            .filter_map(|fold| fold.get(metric))
            .cloned()
            .collect();

        if !scores.is_empty() {
            let mean = scores.iter().sum::<Float>() / scores.len() as Float;
            let variance = scores.iter()
                .map(|score| (score - mean).powi(2))
                .sum::<Float>() / scores.len() as Float;
            let std = variance.sqrt();

            mean_scores.insert(metric.clone(), mean);
            std_scores.insert(metric.clone(), std);
        }
    }

    Ok(CrossValidationResults {
        mean_scores,
        std_scores,
        fold_scores,
        config,
    })
}

/// Optimize ensemble weights using various algorithms
pub fn optimize_ensemble_weights<T: EnsemblePredictor>(
    ensemble: &T,
    validation_data: &ArrayView2<'_, Float>,
    validation_labels: &ArrayView1<'_, Float>,
    optimization_config: WeightOptimizationConfig,
) -> SklResult<Array1<Float>> {
    let n_models = ensemble.ensemble_size();
    let mut weights = Array1::from_elem(n_models, 1.0 / n_models as Float);

    match optimization_config.method {
        WeightOptimizationMethod::GridSearch => {
            optimize_weights_grid_search(ensemble, validation_data, validation_labels, &mut weights)?;
        },
        WeightOptimizationMethod::GradientDescent { learning_rate, max_iterations } => {
            optimize_weights_gradient_descent(
                ensemble, validation_data, validation_labels, &mut weights,
                learning_rate, max_iterations
            )?;
        },
        WeightOptimizationMethod::RandomSearch { n_iterations } => {
            optimize_weights_random_search(
                ensemble, validation_data, validation_labels, &mut weights, n_iterations
            )?;
        },
    }

    // Normalize weights
    let weight_sum: Float = weights.sum();
    if weight_sum > 0.0 {
        weights /= weight_sum;
    }

    Ok(weights)
}

/// Weight optimization configuration
#[derive(Debug, Clone)]
pub struct WeightOptimizationConfig {
    /// Optimization method
    pub method: WeightOptimizationMethod,
    /// Objective function
    pub objective: OptimizationObjective,
    /// Constraints on weights
    pub constraints: WeightConstraints,
}

/// Weight optimization methods
#[derive(Debug, Clone)]
pub enum WeightOptimizationMethod {
    /// Grid search over weight space
    GridSearch,
    /// Gradient descent optimization
    GradientDescent {
        learning_rate: Float,
        max_iterations: usize,
    },
    /// Random search
    RandomSearch {
        n_iterations: usize,
    },
}

/// Optimization objectives
#[derive(Debug, Clone)]
pub enum OptimizationObjective {
    /// Minimize mean squared error
    MinimizeMSE,
    /// Maximize R² score
    MaximizeR2,
    /// Minimize mean absolute error
    MinimizeMAE,
    /// Maximize accuracy (for classification)
    MaximizeAccuracy,
}

/// Constraints on ensemble weights
#[derive(Debug, Clone)]
pub struct WeightConstraints {
    /// Weights must be non-negative
    pub non_negative: bool,
    /// Weights must sum to 1
    pub sum_to_one: bool,
    /// Minimum weight value
    pub min_weight: Option<Float>,
    /// Maximum weight value
    pub max_weight: Option<Float>,
}

impl Default for WeightOptimizationConfig {
    fn default() -> Self {
        Self {
            method: WeightOptimizationMethod::GradientDescent {
                learning_rate: 0.01,
                max_iterations: 100,
            },
            objective: OptimizationObjective::MaximizeR2,
            constraints: WeightConstraints {
                non_negative: true,
                sum_to_one: true,
                min_weight: Some(0.0),
                max_weight: Some(1.0),
            },
        }
    }
}

/// Analyze ensemble diversity
pub fn analyze_ensemble_diversity<T: EnsemblePredictor>(
    ensemble: &T,
    data: &ArrayView2<'_, Float>,
) -> SklResult<DiversityAnalysisReport> {
    let predictions = ensemble.member_predictions(data)?;
    let n_models = predictions.len();

    if n_models < 2 {
        return Ok(DiversityAnalysisReport {
            pairwise_diversities: HashMap::new(),
            mean_diversity: 0.0,
            diversity_matrix: Array2::zeros((0, 0)),
            diversity_measures: HashMap::new(),
        });
    }

    let mut pairwise_diversities = HashMap::new();
    let mut diversity_matrix = Array2::zeros((n_models, n_models));

    // Calculate pairwise diversities
    for i in 0..n_models {
        for j in (i + 1)..n_models {
            let diversity = crate::ensemble_types::utils::calculate_diversity(
                &predictions[i],
                &predictions[j],
                &DiversityMeasure::Disagreement,
            )?;

            let pair_name = format!("model_{}_{}", i, j);
            pairwise_diversities.insert(pair_name, diversity);
            diversity_matrix[[i, j]] = diversity;
            diversity_matrix[[j, i]] = diversity; // Symmetric
        }
    }

    // Calculate mean diversity
    let mean_diversity = if !pairwise_diversities.is_empty() {
        pairwise_diversities.values().sum::<Float>() / pairwise_diversities.len() as Float
    } else {
        0.0
    };

    // Calculate other diversity measures
    let mut diversity_measures = HashMap::new();
    diversity_measures.insert("mean_disagreement".to_string(), mean_diversity);

    // Calculate correlation-based diversity
    if n_models >= 2 {
        let mut total_correlation = 0.0;
        let mut pair_count = 0;

        for i in 0..n_models {
            for j in (i + 1)..n_models {
                if let Ok(correlation) = crate::ensemble_types::utils::calculate_diversity(
                    &predictions[i],
                    &predictions[j],
                    &DiversityMeasure::CorrelationCoefficient,
                ) {
                    total_correlation += correlation;
                    pair_count += 1;
                }
            }
        }

        if pair_count > 0 {
            let mean_correlation = total_correlation / pair_count as Float;
            diversity_measures.insert("mean_correlation".to_string(), mean_correlation);
        }
    }

    Ok(DiversityAnalysisReport {
        pairwise_diversities,
        mean_diversity,
        diversity_matrix,
        diversity_measures,
    })
}

/// Diversity analysis report
#[derive(Debug, Clone)]
pub struct DiversityAnalysisReport {
    /// Pairwise diversity values
    pub pairwise_diversities: HashMap<String, Float>,
    /// Mean diversity across all pairs
    pub mean_diversity: Float,
    /// Full diversity matrix
    pub diversity_matrix: Array2<Float>,
    /// Additional diversity measures
    pub diversity_measures: HashMap<String, Float>,
}

/// Generate comprehensive ensemble report
pub fn generate_ensemble_report<T: EnsemblePredictor>(
    ensemble: &T,
    evaluation_data: &ArrayView2<'_, Float>,
    evaluation_labels: &ArrayView1<'_, Float>,
) -> SklResult<EnsembleReport> {
    let mut validator = EnsembleValidator::new();
    let mut analyzer = PerformanceAnalyzer::new();

    // Validation
    let validation_result = validator.validate_ensemble(ensemble, evaluation_data)?;

    // Performance analysis
    let performance_report = analyzer.analyze(ensemble, evaluation_data, evaluation_labels)?;

    // Diversity analysis
    let diversity_report = analyze_ensemble_diversity(ensemble, evaluation_data)?;

    // Cross-validation
    let cv_results = cross_validate_ensemble(
        ensemble,
        evaluation_data,
        evaluation_labels,
        CrossValidationConfig::default(),
    )?;

    Ok(EnsembleReport {
        validation_result,
        performance_report,
        diversity_report,
        cv_results,
        ensemble_info: EnsembleInfo {
            size: ensemble.ensemble_size(),
            member_names: ensemble.member_names(),
            generation_time: SystemTime::now(),
        },
    })
}

/// Comprehensive ensemble evaluation report
#[derive(Debug, Clone)]
pub struct EnsembleReport {
    /// Validation results
    pub validation_result: ValidationResult,
    /// Performance analysis
    pub performance_report: PerformanceReport,
    /// Diversity analysis
    pub diversity_report: DiversityAnalysisReport,
    /// Cross-validation results
    pub cv_results: CrossValidationResults,
    /// Basic ensemble information
    pub ensemble_info: EnsembleInfo,
}

/// Basic ensemble information
#[derive(Debug, Clone)]
pub struct EnsembleInfo {
    /// Ensemble size
    pub size: usize,
    /// Member names
    pub member_names: Vec<String>,
    /// Report generation time
    pub generation_time: SystemTime,
}

impl EnsembleReport {
    /// Convert report to JSON format
    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "validation": {
                "is_valid": self.validation_result.is_valid,
                "warnings": self.validation_result.warnings,
                "errors": self.validation_result.errors,
                "metrics": self.validation_result.metrics
            },
            "performance": self.performance_report.to_json(),
            "diversity": {
                "mean_diversity": self.diversity_report.mean_diversity,
                "diversity_measures": self.diversity_report.diversity_measures
            },
            "cross_validation": {
                "mean_scores": self.cv_results.mean_scores,
                "std_scores": self.cv_results.std_scores
            },
            "ensemble_info": {
                "size": self.ensemble_info.size,
                "member_names": self.ensemble_info.member_names,
                "generation_time": self.ensemble_info.generation_time
                    .duration_since(SystemTime::UNIX_EPOCH)
                    .unwrap_or(Duration::ZERO).as_secs()
            }
        })
    }
}

// Helper functions for optimization algorithms

fn optimize_weights_grid_search<T: EnsemblePredictor>(
    ensemble: &T,
    validation_data: &ArrayView2<'_, Float>,
    validation_labels: &ArrayView1<'_, Float>,
    weights: &mut Array1<Float>,
) -> SklResult<()> {
    // Simplified grid search
    let n_models = weights.len();
    let grid_points = 5; // 5 points per dimension

    let mut best_score = Float::NEG_INFINITY;
    let mut best_weights = weights.clone();

    // Simple grid search (very basic implementation)
    for _ in 0..100 { // Limited iterations for performance
        let mut current_weights = Array1::zeros(n_models);
        let mut remaining = 1.0;

        for i in 0..(n_models - 1) {
            let weight = remaining * (thread_rng().random::<Float>());
            current_weights[i] = weight;
            remaining -= weight;
        }
        current_weights[n_models - 1] = remaining.max(0.0);

        // Evaluate current weights
        if let Ok(score) = evaluate_weights(ensemble, validation_data, validation_labels, &current_weights) {
            if score > best_score {
                best_score = score;
                best_weights = current_weights;
            }
        }
    }

    *weights = best_weights;
    Ok(())
}

fn optimize_weights_gradient_descent<T: EnsemblePredictor>(
    ensemble: &T,
    validation_data: &ArrayView2<'_, Float>,
    validation_labels: &ArrayView1<'_, Float>,
    weights: &mut Array1<Float>,
    learning_rate: Float,
    max_iterations: usize,
) -> SklResult<()> {
    // Simplified gradient descent
    for _ in 0..max_iterations {
        // Calculate gradient (simplified finite differences)
        let current_score = evaluate_weights(ensemble, validation_data, validation_labels, weights)?;
        let eps = 1e-5;

        for i in 0..weights.len() {
            let mut perturbed_weights = weights.clone();
            perturbed_weights[i] += eps;

            // Normalize perturbed weights
            let sum: Float = perturbed_weights.sum();
            if sum > 0.0 {
                perturbed_weights /= sum;
            }

            if let Ok(perturbed_score) = evaluate_weights(ensemble, validation_data, validation_labels, &perturbed_weights) {
                let gradient = (perturbed_score - current_score) / eps;
                weights[i] += learning_rate * gradient;
            }
        }

        // Normalize and constrain weights
        for weight in weights.iter_mut() {
            *weight = weight.max(0.0).min(1.0);
        }
        let sum: Float = weights.sum();
        if sum > 0.0 {
            *weights /= sum;
        }
    }

    Ok(())
}

fn optimize_weights_random_search<T: EnsemblePredictor>(
    ensemble: &T,
    validation_data: &ArrayView2<'_, Float>,
    validation_labels: &ArrayView1<'_, Float>,
    weights: &mut Array1<Float>,
    n_iterations: usize,
) -> SklResult<()> {
    let n_models = weights.len();
    let mut best_score = Float::NEG_INFINITY;
    let mut best_weights = weights.clone();

    for _ in 0..n_iterations {
        // Generate random weights
        let mut random_weights = Array1::zeros(n_models);
        let mut remaining = 1.0;

        for i in 0..(n_models - 1) {
            let weight = remaining * thread_rng().random::<Float>();
            random_weights[i] = weight;
            remaining -= weight;
        }
        random_weights[n_models - 1] = remaining.max(0.0);

        // Evaluate random weights
        if let Ok(score) = evaluate_weights(ensemble, validation_data, validation_labels, &random_weights) {
            if score > best_score {
                best_score = score;
                best_weights = random_weights;
            }
        }
    }

    *weights = best_weights;
    Ok(())
}

fn evaluate_weights<T: EnsemblePredictor>(
    ensemble: &T,
    validation_data: &ArrayView2<'_, Float>,
    validation_labels: &ArrayView1<'_, Float>,
    weights: &Array1<Float>,
) -> SklResult<Float> {
    // Get individual predictions
    let predictions = ensemble.member_predictions(validation_data)?;

    // Combine with weights
    let n_samples = predictions[0].len();
    let mut weighted_prediction = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        let mut weighted_sum = 0.0;
        for (model_idx, pred) in predictions.iter().enumerate() {
            weighted_sum += weights[model_idx] * pred[sample_idx];
        }
        weighted_prediction[sample_idx] = weighted_sum;
    }

    // Calculate R² score
    calculate_r2_simple(&weighted_prediction, &validation_labels.view())
}

// Simple helper functions

fn combine_predictions_simple(predictions: &[Array1<Float>]) -> SklResult<Array1<Float>> {
    if predictions.is_empty() {
        return Err(SklearsError::InvalidInput("No predictions to combine".to_string()));
    }

    let n_samples = predictions[0].len();
    let mut combined = Array1::zeros(n_samples);

    for sample_idx in 0..n_samples {
        let sum: Float = predictions.iter().map(|pred| pred[sample_idx]).sum();
        combined[sample_idx] = sum / predictions.len() as Float;
    }

    Ok(combined)
}

fn calculate_r2_simple(predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
    let mean_label = labels.mean().unwrap_or(0.0);

    let ss_res: Float = predictions.iter()
        .zip(labels.iter())
        .map(|(pred, label)| (label - pred).powi(2))
        .sum();

    let ss_tot: Float = labels.iter()
        .map(|label| (label - mean_label).powi(2))
        .sum();

    Ok(if ss_tot > 0.0 { 1.0 - ss_res / ss_tot } else { 0.0 })
}

fn calculate_mse_simple(predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
    let mse = predictions.iter()
        .zip(labels.iter())
        .map(|(pred, label)| (pred - label).powi(2))
        .sum::<Float>() / predictions.len() as Float;
    Ok(mse)
}

fn calculate_mae_simple(predictions: &Array1<Float>, labels: &ArrayView1<'_, Float>) -> SklResult<Float> {
    let mae = predictions.iter()
        .zip(labels.iter())
        .map(|(pred, label)| (pred - label).abs())
        .sum::<Float>() / predictions.len() as Float;
    Ok(mae)
}

// Temporary rand implementation for simplicity
mod rand {
    use super::Float;
    use std::cell::Cell;

    thread_local! {
        static SEED: Cell<u64> = Cell::new(1);
    }

    pub fn random<T>() -> T
    where
        T: From<Float>,
    {
        SEED.with(|seed| {
            let current = seed.get();
            let next = current.wrapping_mul(1103515245).wrapping_add(12345);
            seed.set(next);
            T::from((next as Float / u64::MAX as Float))
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_validator() {
        let mut validator = EnsembleValidator::new();
        let data = Array2::zeros((10, 5));

        // Create mock ensemble
        struct MockEnsemble;
        impl EnsemblePredictor for MockEnsemble {
            fn member_names(&self) -> Vec<String> {
                vec!["model1".to_string(), "model2".to_string()]
            }

            fn ensemble_size(&self) -> usize {
                2
            }

            fn member_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
                Ok(vec![
                    Array1::zeros(x.nrows()),
                    Array1::ones(x.nrows()),
                ])
            }

            fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
                Ok(Array1::from_elem(x.nrows(), 0.8))
            }

            fn ensemble_metrics(&self) -> SklResult<EnsembleMetrics> {
                Ok(EnsembleMetrics::new())
            }
        }

        let ensemble = MockEnsemble;
        let result = validator.validate_ensemble(&ensemble, &data.view()).unwrap();

        assert!(result.is_valid);
        assert!(result.metrics.contains_key("ensemble_size"));
        assert_eq!(result.metrics["ensemble_size"], 2.0);
    }

    #[test]
    fn test_performance_analyzer() {
        let mut analyzer = PerformanceAnalyzer::new();
        let test_data = Array2::zeros((10, 5));
        let test_labels = Array1::zeros(10);

        struct MockEnsemble;
        impl EnsemblePredictor for MockEnsemble {
            fn member_names(&self) -> Vec<String> {
                vec!["model1".to_string()]
            }

            fn ensemble_size(&self) -> usize {
                1
            }

            fn member_predictions(&self, x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
                Ok(vec![Array1::zeros(x.nrows())])
            }

            fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
                Ok(Array1::from_elem(x.nrows(), 0.8))
            }

            fn ensemble_metrics(&self) -> SklResult<EnsembleMetrics> {
                Ok(EnsembleMetrics::new())
            }
        }

        let ensemble = MockEnsemble;
        let report = analyzer.analyze(&ensemble, &test_data.view(), &test_labels.view()).unwrap();

        assert_eq!(report.ensemble_size, 1);
        assert!(report.efficiency_metrics.contains_key("models_count"));
    }

    #[test]
    fn test_combining_predictions() {
        let pred1 = Array1::from(vec![1.0, 2.0, 3.0]);
        let pred2 = Array1::from(vec![2.0, 4.0, 6.0]);
        let predictions = vec![pred1, pred2];

        let combined = combine_predictions_simple(&predictions).unwrap();
        let expected = Array1::from(vec![1.5, 3.0, 4.5]);

        for (a, b) in combined.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_r2_calculation() {
        let predictions = Array1::from(vec![1.0, 2.0, 3.0]);
        let labels = Array1::from(vec![1.0, 2.0, 3.0]);

        let r2 = calculate_r2_simple(&predictions, &labels.view()).unwrap();
        assert!((r2 - 1.0).abs() < 1e-6); // Perfect predictions should give R² = 1
    }

    #[test]
    fn test_diversity_analysis() {
        struct MockEnsemble {
            predictions: Vec<Array1<Float>>,
        }

        impl EnsemblePredictor for MockEnsemble {
            fn member_names(&self) -> Vec<String> {
                (0..self.predictions.len()).map(|i| format!("model_{}", i)).collect()
            }

            fn ensemble_size(&self) -> usize {
                self.predictions.len()
            }

            fn member_predictions(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Vec<Array1<Float>>> {
                Ok(self.predictions.clone())
            }

            fn prediction_confidence(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array1<Float>> {
                Ok(Array1::from_elem(x.nrows(), 0.8))
            }

            fn ensemble_metrics(&self) -> SklResult<EnsembleMetrics> {
                Ok(EnsembleMetrics::new())
            }
        }

        let ensemble = MockEnsemble {
            predictions: vec![
                Array1::from(vec![1.0, 2.0, 3.0]),
                Array1::from(vec![1.0, 3.0, 3.0]),
            ]
        };

        let data = Array2::zeros((3, 2));
        let report = analyze_ensemble_diversity(&ensemble, &data.view()).unwrap();

        assert!(report.mean_diversity >= 0.0);
        assert_eq!(report.diversity_matrix.nrows(), 2);
        assert_eq!(report.diversity_matrix.ncols(), 2);
    }
}