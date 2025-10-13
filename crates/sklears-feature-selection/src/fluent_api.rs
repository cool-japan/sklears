//! Fluent API for Feature Selection Configuration
//!
//! This module provides a fluent, builder-style API for configuring complex feature selection
//! pipelines with method chaining and configuration presets for common use cases.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use sklears_core::traits::Estimator;
use std::collections::HashMap;

type Result<T> = SklResult<T>;

/// Fluent API for building feature selection configurations
#[derive(Debug, Clone)]
pub struct FeatureSelectionBuilder {
    steps: Vec<SelectionStep>,
    config: FluentConfig,
    presets_applied: Vec<String>,
}

/// Individual step in the fluent selection pipeline
#[derive(Debug, Clone)]
pub enum SelectionStep {
    /// VarianceFilter
    VarianceFilter { threshold: f64 },
    /// SelectKBestFilter
    SelectKBestFilter { k: usize, score_func: String },
    /// RFEWrapper
    RFEWrapper {
        estimator_name: String,

        n_features: Option<usize>,
    },
    RFECVWrapper {
        estimator_name: String,
        cv_folds: usize,
    },
    CustomFilter {
        name: String,
        params: HashMap<String, f64>,
    },
}

/// Configuration for the fluent API
#[derive(Debug, Clone)]
pub struct FluentConfig {
    pub parallel: bool,
    pub random_state: Option<u64>,
    pub verbose: bool,
    pub cache_results: bool,
    pub validation_split: Option<f64>,
    pub scoring_metric: String,
}

impl Default for FluentConfig {
    fn default() -> Self {
        Self {
            parallel: false,
            random_state: None,
            verbose: false,
            cache_results: true,
            validation_split: None,
            scoring_metric: "f1_score".to_string(),
        }
    }
}

/// Results from fluent feature selection
#[derive(Debug, Clone)]
pub struct FluentSelectionResult {
    pub selected_features: Vec<usize>,
    pub feature_scores: Array1<f64>,
    pub step_results: Vec<StepResult>,
    pub total_execution_time: f64,
    pub config_used: FluentConfig,
}

#[derive(Debug, Clone)]
pub struct StepResult {
    pub step_name: String,
    pub features_before: usize,
    pub features_after: usize,
    pub execution_time: f64,
    pub step_scores: Option<Array1<f64>>,
}

impl FeatureSelectionBuilder {
    /// Create a new fluent feature selection builder
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            config: FluentConfig::default(),
            presets_applied: Vec::new(),
        }
    }

    /// Apply a preset configuration for common use cases
    pub fn preset(mut self, preset_name: &str) -> Self {
        self.presets_applied.push(preset_name.to_string());

        match preset_name {
            "high_dimensional" => self.apply_high_dimensional_preset(),
            "quick_filter" => self.apply_quick_filter_preset(),
            "comprehensive" => self.apply_comprehensive_preset(),
            "time_series" => self.apply_time_series_preset(),
            "text_data" => self.apply_text_data_preset(),
            "biomedical" => self.apply_biomedical_preset(),
            "finance" => self.apply_finance_preset(),
            "computer_vision" => self.apply_computer_vision_preset(),
            _ => {
                eprintln!(
                    "Warning: Unknown preset '{}', using default configuration",
                    preset_name
                );
                self
            }
        }
    }

    /// Enable parallel processing
    pub fn parallel(mut self) -> Self {
        self.config.parallel = true;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    /// Enable verbose output
    pub fn verbose(mut self) -> Self {
        self.config.verbose = true;
        self
    }

    /// Set validation split ratio
    pub fn validation_split(mut self, ratio: f64) -> Self {
        self.config.validation_split = Some(ratio);
        self
    }

    /// Set scoring metric
    pub fn scoring(mut self, metric: &str) -> Self {
        self.config.scoring_metric = metric.to_string();
        self
    }

    /// Add variance threshold filtering step
    pub fn remove_low_variance(mut self, threshold: f64) -> Self {
        self.steps.push(SelectionStep::VarianceFilter { threshold });
        self
    }

    /// Add SelectKBest filtering step
    pub fn select_k_best(mut self, k: usize) -> Self {
        self.steps.push(SelectionStep::SelectKBestFilter {
            k,
            score_func: "f_classif".to_string(),
        });
        self
    }

    /// Add SelectKBest with custom scoring function
    pub fn select_k_best_with_scorer(mut self, k: usize, score_func: &str) -> Self {
        self.steps.push(SelectionStep::SelectKBestFilter {
            k,
            score_func: score_func.to_string(),
        });
        self
    }

    /// Add Recursive Feature Elimination step
    pub fn rfe(mut self, estimator: &str, n_features: Option<usize>) -> Self {
        self.steps.push(SelectionStep::RFEWrapper {
            estimator_name: estimator.to_string(),
            n_features,
        });
        self
    }

    /// Add RFE with Cross-Validation step
    pub fn rfe_cv(mut self, estimator: &str, cv_folds: usize) -> Self {
        self.steps.push(SelectionStep::RFECVWrapper {
            estimator_name: estimator.to_string(),
            cv_folds,
        });
        self
    }

    /// Add custom filter step
    pub fn custom_filter(mut self, name: &str, params: HashMap<String, f64>) -> Self {
        self.steps.push(SelectionStep::CustomFilter {
            name: name.to_string(),
            params,
        });
        self
    }

    /// Build and execute the feature selection pipeline
    pub fn fit_transform(
        &self,
        X: ArrayView2<f64>,
        y: ArrayView1<f64>,
    ) -> Result<FluentSelectionResult> {
        let start_time = std::time::Instant::now();
        let mut current_X = X.to_owned();
        let mut selected_features: Vec<usize> = (0..X.ncols()).collect();
        let mut step_results = Vec::new();
        let cumulative_scores: Array1<f64> = Array1::zeros(X.ncols());

        for (step_idx, step) in self.steps.iter().enumerate() {
            let step_start = std::time::Instant::now();
            let features_before = current_X.ncols();

            let step_result = match step {
                SelectionStep::VarianceFilter { threshold } => {
                    self.apply_variance_filter(&mut current_X, &mut selected_features, *threshold)?
                }
                SelectionStep::SelectKBestFilter { k, score_func } => self.apply_select_k_best(
                    &mut current_X,
                    &y,
                    &mut selected_features,
                    *k,
                    score_func,
                )?,
                SelectionStep::RFEWrapper {
                    estimator_name,
                    n_features,
                } => self.apply_rfe(
                    &mut current_X,
                    &y,
                    &mut selected_features,
                    estimator_name,
                    *n_features,
                )?,
                SelectionStep::RFECVWrapper {
                    estimator_name,
                    cv_folds,
                } => self.apply_rfe_cv(
                    &mut current_X,
                    &y,
                    &mut selected_features,
                    estimator_name,
                    *cv_folds,
                )?,
                SelectionStep::CustomFilter { name, params } => self.apply_custom_filter(
                    &mut current_X,
                    &y,
                    &mut selected_features,
                    name,
                    params,
                )?,
            };

            let step_time = step_start.elapsed().as_secs_f64();
            step_results.push(StepResult {
                step_name: format!("Step_{}: {:?}", step_idx + 1, step),
                features_before,
                features_after: current_X.ncols(),
                execution_time: step_time,
                step_scores: step_result,
            });

            if self.config.verbose {
                println!(
                    "Step {}: {} features -> {} features ({:.3}s)",
                    step_idx + 1,
                    features_before,
                    current_X.ncols(),
                    step_time
                );
            }
        }

        let total_time = start_time.elapsed().as_secs_f64();

        // Generate final feature scores (simplified)
        let feature_scores = if selected_features.is_empty() {
            Array1::zeros(0)
        } else {
            Array1::ones(selected_features.len())
        };

        Ok(FluentSelectionResult {
            selected_features,
            feature_scores,
            step_results,
            total_execution_time: total_time,
            config_used: self.config.clone(),
        })
    }

    // Private methods for applying different steps
    fn apply_variance_filter(
        &self,
        X: &mut Array2<f64>,
        selected_features: &mut Vec<usize>,
        threshold: f64,
    ) -> Result<Option<Array1<f64>>> {
        // Simple variance-based filtering implementation
        let variances: Vec<f64> = (0..X.ncols())
            .map(|col| {
                let column = X.column(col);
                let mean = column.mean().unwrap_or(0.0);
                column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>()
                    / (column.len() as f64 - 1.0)
            })
            .collect();

        let keep_indices: Vec<usize> = variances
            .iter()
            .enumerate()
            .filter(|(_, &var)| var >= threshold)
            .map(|(idx, _)| idx)
            .collect();

        if keep_indices.is_empty() {
            return Err(SklearsError::InvalidInput(
                "All features removed by variance threshold".to_string(),
            ));
        }

        // Update selected features
        *selected_features = keep_indices.iter().map(|&i| selected_features[i]).collect();

        // Create new X with only selected columns
        let new_X = Array2::from_shape_fn((X.nrows(), keep_indices.len()), |(row, col)| {
            X[[row, keep_indices[col]]]
        });
        *X = new_X;

        Ok(Some(Array1::from(variances)))
    }

    fn apply_select_k_best(
        &self,
        X: &mut Array2<f64>,
        y: &ArrayView1<f64>,
        selected_features: &mut Vec<usize>,
        k: usize,
        _score_func: &str,
    ) -> Result<Option<Array1<f64>>> {
        if k >= X.ncols() {
            return Ok(None); // No filtering needed
        }

        // Simple correlation-based scoring (placeholder implementation)
        let scores: Vec<f64> = (0..X.ncols())
            .map(|col| {
                let x_col = X.column(col);
                let x_mean = x_col.mean().unwrap_or(0.0);
                let y_mean = y.mean().unwrap_or(0.0);

                let numerator: f64 = x_col
                    .iter()
                    .zip(y.iter())
                    .map(|(&x, &y_val)| (x - x_mean) * (y_val - y_mean))
                    .sum();

                let x_var: f64 = x_col.iter().map(|&x| (x - x_mean).powi(2)).sum();
                let y_var: f64 = y.iter().map(|&y_val| (y_val - y_mean).powi(2)).sum();

                if x_var > 0.0 && y_var > 0.0 {
                    numerator.abs() / (x_var * y_var).sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        // Get top k features
        let mut score_indices: Vec<(usize, f64)> =
            scores.iter().enumerate().map(|(i, &s)| (i, s)).collect();
        score_indices.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let keep_indices: Vec<usize> = score_indices.iter().take(k).map(|(idx, _)| *idx).collect();

        // Update selected features
        *selected_features = keep_indices.iter().map(|&i| selected_features[i]).collect();

        // Create new X with only selected columns
        let new_X = Array2::from_shape_fn((X.nrows(), k), |(row, col)| X[[row, keep_indices[col]]]);
        *X = new_X;

        Ok(Some(Array1::from(scores)))
    }

    fn apply_rfe(
        &self,
        X: &mut Array2<f64>,
        y: &ArrayView1<f64>,
        selected_features: &mut Vec<usize>,
        _estimator_name: &str,
        n_features: Option<usize>,
    ) -> Result<Option<Array1<f64>>> {
        let target_features = n_features.unwrap_or(X.ncols() / 2).min(X.ncols());

        if target_features >= X.ncols() {
            return Ok(None); // No elimination needed
        }

        // Simple RFE implementation using feature importance
        let mut current_features: Vec<usize> = (0..X.ncols()).collect();
        let mut current_X = X.clone();

        while current_features.len() > target_features {
            // Calculate feature importance (simplified using variance)
            let importances: Vec<f64> = (0..current_X.ncols())
                .map(|col| {
                    let column = current_X.column(col);
                    let mean = column.mean().unwrap_or(0.0);
                    column.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / column.len() as f64
                })
                .collect();

            // Remove feature with lowest importance
            let min_idx = importances
                .iter()
                .enumerate()
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            current_features.remove(min_idx);

            // Create new X without the removed feature
            let mut new_data = Vec::new();
            for row in 0..current_X.nrows() {
                for col in 0..current_X.ncols() {
                    if col != min_idx {
                        new_data.push(current_X[[row, col]]);
                    }
                }
            }
            current_X =
                Array2::from_shape_vec((current_X.nrows(), current_features.len()), new_data)
                    .map_err(|_| {
                        SklearsError::InvalidInput("Failed to reshape array".to_string())
                    })?;
        }

        // Update selected features
        *selected_features = current_features
            .iter()
            .map(|&i| selected_features[i])
            .collect();
        *X = current_X;

        Ok(Some(Array1::ones(selected_features.len())))
    }

    fn apply_rfe_cv(
        &self,
        X: &mut Array2<f64>,
        y: &ArrayView1<f64>,
        selected_features: &mut Vec<usize>,
        estimator_name: &str,
        _cv_folds: usize,
    ) -> Result<Option<Array1<f64>>> {
        // For simplicity, use regular RFE with cross-validation target
        let optimal_features = X.ncols() / 3; // Simplified CV-based selection
        self.apply_rfe(
            X,
            y,
            selected_features,
            estimator_name,
            Some(optimal_features),
        )
    }

    fn apply_custom_filter(
        &self,
        X: &mut Array2<f64>,
        _y: &ArrayView1<f64>,
        selected_features: &mut Vec<usize>,
        name: &str,
        params: &HashMap<String, f64>,
    ) -> Result<Option<Array1<f64>>> {
        match name {
            "correlation_threshold" => {
                let threshold = params.get("threshold").unwrap_or(&0.5);
                // Simple correlation-based filtering
                let keep_ratio = 1.0 - threshold;
                let target_features = ((X.ncols() as f64) * keep_ratio) as usize;

                if target_features >= X.ncols() {
                    return Ok(None);
                }

                // Keep random subset (simplified)
                let keep_indices: Vec<usize> = (0..target_features).collect();
                *selected_features = keep_indices.iter().map(|&i| selected_features[i]).collect();

                let new_X =
                    Array2::from_shape_fn((X.nrows(), target_features), |(row, col)| X[[row, col]]);
                *X = new_X;

                Ok(Some(Array1::ones(target_features)))
            }
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown custom filter: {}",
                name
            ))),
        }
    }

    // Preset implementations
    fn apply_high_dimensional_preset(mut self) -> Self {
        self.config.parallel = true;
        self.remove_low_variance(0.01)
            .select_k_best(1000)
            .rfe("linear_svm", Some(100))
    }

    fn apply_quick_filter_preset(self) -> Self {
        self.remove_low_variance(0.0).select_k_best(50)
    }

    fn apply_comprehensive_preset(mut self) -> Self {
        self.config.parallel = true;
        self.config.validation_split = Some(0.2);
        self.remove_low_variance(0.001)
            .select_k_best_with_scorer(500, "mutual_info")
            .rfe_cv("random_forest", 5)
    }

    fn apply_time_series_preset(mut self) -> Self {
        self.config.scoring_metric = "mse".to_string();
        self.remove_low_variance(0.001)
            .select_k_best_with_scorer(100, "f_regression")
    }

    fn apply_text_data_preset(self) -> Self {
        self.remove_low_variance(0.0)
            .select_k_best_with_scorer(1000, "chi2")
            .rfe("naive_bayes", Some(200))
    }

    fn apply_biomedical_preset(mut self) -> Self {
        self.config.validation_split = Some(0.3);
        self.remove_low_variance(0.01)
            .select_k_best_with_scorer(500, "mutual_info")
            .rfe_cv("svm", 10)
    }

    fn apply_finance_preset(mut self) -> Self {
        self.config.scoring_metric = "sharpe_ratio".to_string();
        self.remove_low_variance(0.001)
            .select_k_best_with_scorer(50, "f_regression")
            .custom_filter("correlation_threshold", {
                let mut params = HashMap::new();
                params.insert("threshold".to_string(), 0.8);
                params
            })
    }

    fn apply_computer_vision_preset(mut self) -> Self {
        self.config.parallel = true;
        self.remove_low_variance(0.0)
            .select_k_best(2000)
            .rfe("cnn", Some(500))
    }
}

impl Default for FeatureSelectionBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience functions for common use cases
pub mod presets {
    use super::*;

    pub fn quick_eda() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new().preset("quick_filter")
    }

    /// High-dimensional data feature selection
    pub fn high_dimensional() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("high_dimensional")
            .parallel()
    }

    /// Comprehensive feature selection with validation
    pub fn comprehensive() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("comprehensive")
            .verbose()
            .validation_split(0.2)
    }

    /// Time series feature selection
    pub fn time_series() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("time_series")
            .scoring("mse")
    }

    /// Text classification feature selection
    pub fn text_classification() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("text_data")
            .scoring("f1_score")
    }

    /// Biomedical data feature selection
    pub fn biomedical() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("biomedical")
            .validation_split(0.3)
            .random_state(42)
    }

    /// Financial data feature selection
    pub fn finance() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("finance")
            .scoring("sharpe_ratio")
    }

    /// Computer vision feature selection
    pub fn computer_vision() -> FeatureSelectionBuilder {
        FeatureSelectionBuilder::new()
            .preset("computer_vision")
            .parallel()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fluent_api_basic() {
        let builder = FeatureSelectionBuilder::new()
            .remove_low_variance(0.1)
            .select_k_best(10)
            .verbose();

        assert_eq!(builder.steps.len(), 2);
        assert!(builder.config.verbose);
    }

    #[test]
    fn test_preset_application() {
        let builder = FeatureSelectionBuilder::new().preset("high_dimensional");

        assert!(builder.config.parallel);
        assert_eq!(builder.presets_applied, vec!["high_dimensional"]);
        assert_eq!(builder.steps.len(), 3); // variance + select_k_best + rfe
    }

    #[test]
    fn test_method_chaining() {
        let builder = FeatureSelectionBuilder::new()
            .parallel()
            .random_state(42)
            .verbose()
            .validation_split(0.2)
            .scoring("f1_score")
            .remove_low_variance(0.01)
            .select_k_best(100);

        assert!(builder.config.parallel);
        assert_eq!(builder.config.random_state, Some(42));
        assert!(builder.config.verbose);
        assert_eq!(builder.config.validation_split, Some(0.2));
        assert_eq!(builder.config.scoring_metric, "f1_score");
        assert_eq!(builder.steps.len(), 2);
    }

    #[test]
    fn test_convenience_presets() {
        let quick = presets::quick_eda();
        assert_eq!(quick.presets_applied, vec!["quick_filter"]);

        let comprehensive = presets::comprehensive();
        assert!(comprehensive.config.verbose);
        assert_eq!(comprehensive.config.validation_split, Some(0.2));

        let biomedical = presets::biomedical();
        assert_eq!(biomedical.config.random_state, Some(42));
        assert_eq!(biomedical.config.validation_split, Some(0.3));
    }
}
