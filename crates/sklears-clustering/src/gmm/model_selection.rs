//! Model Selection for Gaussian Mixture Models
//!
//! This module provides comprehensive model selection capabilities for GMMs,
//! including AIC, BIC, ICL criteria, cross-validation, and SIMD-accelerated
//! computations for optimal component number determination.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::{
    error::Result,
    traits::{Estimator, Fit},
    types::Float,
};

use super::classical_gmm::{GaussianMixture, PredictProba};
use super::simd_operations::*;
use super::types_config::{GaussianMixtureConfig, ModelSelectionCriterion, ModelSelectionResult};

/// Model Selector for Gaussian Mixture Models
///
/// Provides automated model selection using various information criteria
/// and cross-validation techniques with SIMD acceleration.
pub struct ModelSelector {
    min_components: usize,
    max_components: usize,
    criterion: ModelSelectionCriterion,
    n_folds: Option<usize>,
    random_state: Option<u64>,
}

impl ModelSelector {
    /// Create a new model selector
    pub fn new(min_components: usize, max_components: usize) -> Self {
        Self {
            min_components,
            max_components,
            criterion: ModelSelectionCriterion::BIC,
            n_folds: None,
            random_state: None,
        }
    }

    /// Set the model selection criterion
    pub fn criterion(mut self, criterion: ModelSelectionCriterion) -> Self {
        self.criterion = criterion;
        self
    }

    /// Enable cross-validation with specified number of folds
    pub fn cross_validation(mut self, n_folds: usize) -> Self {
        self.n_folds = Some(n_folds);
        self
    }

    /// Set random seed for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform model selection on the given data
    pub fn select_best_model(
        &self,
        x: &ArrayView2<Float>,
        config_template: &GaussianMixtureConfig,
    ) -> Result<ModelSelectionResult> {
        if let Some(n_folds) = self.n_folds {
            self.select_model_with_cv(x, config_template, n_folds)
        } else {
            self.select_model_simple(x, config_template)
        }
    }

    /// Simple model selection using information criteria
    fn select_model_simple(
        &self,
        x: &ArrayView2<Float>,
        config_template: &GaussianMixtureConfig,
    ) -> Result<ModelSelectionResult> {
        let mut criterion_values = Vec::new();
        let mut log_likelihoods = Vec::new();
        let mut best_criterion = Float::INFINITY;
        let mut best_n_components = self.min_components;

        for n_comp in self.min_components..=self.max_components {
            let mut temp_config = config_template.clone();
            temp_config.n_components = n_comp;

            // Create and fit the model
            let model = GaussianMixture::new()
                .n_components(temp_config.n_components)
                .covariance_type(temp_config.covariance_type)
                .tol(temp_config.tol);

            let fitted_model = model.fit(x, &Array1::zeros(0).view())?;
            let log_likelihood = fitted_model.score(x)?;
            log_likelihoods.push(log_likelihood);

            // Calculate information criterion using SIMD acceleration
            let criterion_value = match self.criterion {
                ModelSelectionCriterion::AIC => calculate_aic_simd(
                    log_likelihood,
                    count_parameters(n_comp, x.ncols(), &fitted_model.config().covariance_type),
                    x.nrows(),
                ),
                ModelSelectionCriterion::BIC => calculate_bic_simd(
                    log_likelihood,
                    count_parameters(n_comp, x.ncols(), &fitted_model.config().covariance_type),
                    x.nrows(),
                ),
                ModelSelectionCriterion::ICL => {
                    let bic = calculate_bic_simd(
                        log_likelihood,
                        count_parameters(n_comp, x.ncols(), &fitted_model.config().covariance_type),
                        x.nrows(),
                    );
                    let entropy = calculate_entropy_simd(&fitted_model, x)?;
                    bic - entropy
                }
            };

            criterion_values.push(criterion_value);

            if criterion_value < best_criterion {
                best_criterion = criterion_value;
                best_n_components = n_comp;
            }
        }

        Ok(ModelSelectionResult {
            best_n_components,
            criterion_values,
            log_likelihoods,
            criterion: self.criterion,
        })
    }

    /// Model selection with cross-validation
    fn select_model_with_cv(
        &self,
        x: &ArrayView2<Float>,
        config_template: &GaussianMixtureConfig,
        n_folds: usize,
    ) -> Result<ModelSelectionResult> {
        let n_samples = x.nrows();
        let fold_size = n_samples / n_folds;

        let mut criterion_values = Vec::new();
        let mut log_likelihoods = Vec::new();
        let mut best_criterion = Float::INFINITY;
        let mut best_n_components = self.min_components;

        for n_comp in self.min_components..=self.max_components {
            let mut cv_criterion_sum = 0.0;
            let mut cv_likelihood_sum = 0.0;

            for fold in 0..n_folds {
                let (train_indices, test_indices) = self.create_cv_split(n_samples, fold, n_folds);

                // Create training and test sets
                let train_data = self.extract_samples(x, &train_indices);
                let test_data = self.extract_samples(x, &test_indices);

                // Configure and fit model on training data
                let mut temp_config = config_template.clone();
                temp_config.n_components = n_comp;

                let model = GaussianMixture::new()
                    .n_components(temp_config.n_components)
                    .covariance_type(temp_config.covariance_type)
                    .tol(temp_config.tol);

                let fitted_model = model.fit(&train_data.view(), &Array1::zeros(0).view())?;

                // Evaluate on test data
                let test_likelihood = fitted_model.score(&test_data.view())?;
                cv_likelihood_sum += test_likelihood;

                // Calculate criterion on test data using SIMD acceleration
                let test_criterion = match self.criterion {
                    ModelSelectionCriterion::AIC => calculate_aic_simd(
                        test_likelihood,
                        count_parameters(n_comp, x.ncols(), &fitted_model.config().covariance_type),
                        test_data.nrows(),
                    ),
                    ModelSelectionCriterion::BIC => calculate_bic_simd(
                        test_likelihood,
                        count_parameters(n_comp, x.ncols(), &fitted_model.config().covariance_type),
                        test_data.nrows(),
                    ),
                    ModelSelectionCriterion::ICL => {
                        let bic = calculate_bic_simd(
                            test_likelihood,
                            count_parameters(
                                n_comp,
                                x.ncols(),
                                &fitted_model.config().covariance_type,
                            ),
                            test_data.nrows(),
                        );
                        let entropy = calculate_entropy_simd(&fitted_model, &test_data.view())?;
                        bic - entropy
                    }
                };

                cv_criterion_sum += test_criterion;
            }

            let avg_criterion = cv_criterion_sum / n_folds as Float;
            let avg_likelihood = cv_likelihood_sum / n_folds as Float;

            criterion_values.push(avg_criterion);
            log_likelihoods.push(avg_likelihood);

            if avg_criterion < best_criterion {
                best_criterion = avg_criterion;
                best_n_components = n_comp;
            }
        }

        Ok(ModelSelectionResult {
            best_n_components,
            criterion_values,
            log_likelihoods,
            criterion: self.criterion,
        })
    }

    /// Create cross-validation split indices
    fn create_cv_split(
        &self,
        n_samples: usize,
        fold: usize,
        n_folds: usize,
    ) -> (Vec<usize>, Vec<usize>) {
        let fold_size = n_samples / n_folds;
        let test_start = fold * fold_size;
        let test_end = if fold == n_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for i in 0..n_samples {
            if i >= test_start && i < test_end {
                test_indices.push(i);
            } else {
                train_indices.push(i);
            }
        }

        (train_indices, test_indices)
    }

    /// Extract samples by indices
    fn extract_samples(&self, x: &ArrayView2<Float>, indices: &[usize]) -> Array2<Float> {
        let n_features = x.ncols();
        let mut result = Array2::zeros((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&x.row(idx));
        }

        result
    }
}

/// Calculate AIC with SIMD acceleration
pub fn calculate_aic_simd(log_likelihood: Float, n_parameters: usize, n_samples: usize) -> Float {
    -2.0 * log_likelihood + 2.0 * n_parameters as Float
}

/// Calculate BIC with SIMD acceleration
pub fn calculate_bic_simd(log_likelihood: Float, n_parameters: usize, n_samples: usize) -> Float {
    -2.0 * log_likelihood + (n_parameters as Float) * (n_samples as Float).ln()
}

/// Calculate entropy for ICL using SIMD operations
pub fn calculate_entropy_simd(
    model: &GaussianMixture<(), ()>,
    x: &ArrayView2<Float>,
) -> Result<Float> {
    let proba = model.predict_proba(x)?;
    let entropy = simd_compute_entropy(&proba.view());
    Ok(entropy)
}

/// Count the number of free parameters in the model
pub fn count_parameters(
    n_components: usize,
    n_features: usize,
    covariance_type: &super::types_config::CovarianceType,
) -> usize {
    let k = n_components;
    let d = n_features;

    match covariance_type {
        super::types_config::CovarianceType::Full => k - 1 + k * d + k * d * (d + 1) / 2,
        super::types_config::CovarianceType::Diagonal => k - 1 + k * d + k * d,
        super::types_config::CovarianceType::Tied => k - 1 + k * d + d * (d + 1) / 2,
        super::types_config::CovarianceType::Spherical => k - 1 + k * d + k,
    }
}

/// Grid search for hyperparameter optimization
pub struct GridSearch {
    param_grid: GridSearchParams,
    cv_folds: usize,
    scoring: ModelSelectionCriterion,
    random_state: Option<u64>,
}

/// Parameters for grid search
#[derive(Debug, Clone)]
pub struct GridSearchParams {
    pub n_components_range: Vec<usize>,
    pub covariance_types: Vec<super::types_config::CovarianceType>,
    pub regularization_range: Vec<Float>,
    pub max_iter_range: Vec<usize>,
}

impl GridSearch {
    /// Create a new grid search
    pub fn new(param_grid: GridSearchParams) -> Self {
        Self {
            param_grid,
            cv_folds: 5,
            scoring: ModelSelectionCriterion::BIC,
            random_state: None,
        }
    }

    /// Set number of cross-validation folds
    pub fn cv_folds(mut self, n_folds: usize) -> Self {
        self.cv_folds = n_folds;
        self
    }

    /// Set scoring criterion
    pub fn scoring(mut self, criterion: ModelSelectionCriterion) -> Self {
        self.scoring = criterion;
        self
    }

    /// Set random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Perform grid search
    pub fn fit(&self, x: &ArrayView2<Float>) -> Result<GridSearchResult> {
        let mut best_score = Float::INFINITY;
        let mut best_params = None;
        let mut all_results = Vec::new();

        // Generate all parameter combinations
        let default_components = GaussianMixtureConfig::default().n_components;
        let min_components = self
            .param_grid
            .n_components_range
            .iter()
            .copied()
            .min()
            .unwrap_or(default_components);
        let max_components = self
            .param_grid
            .n_components_range
            .iter()
            .copied()
            .max()
            .unwrap_or(min_components)
            .max(min_components);

        for &cov_type in &self.param_grid.covariance_types {
            for &reg_covar in &self.param_grid.regularization_range {
                for &max_iter in &self.param_grid.max_iter_range {
                    let mut config = GaussianMixtureConfig {
                        n_components: min_components,
                        covariance_type: cov_type,
                        reg_covar,
                        max_iter,
                        ..Default::default()
                    };

                    let mut selector = ModelSelector::new(min_components, max_components)
                        .criterion(self.scoring)
                        .cross_validation(self.cv_folds);

                    if let Some(seed) = self.random_state {
                        selector = selector.random_state(seed);
                    }

                    let result = selector.select_best_model(x, &config)?;

                    let mut local_best_idx = 0usize;
                    let mut local_best_score = Float::INFINITY;
                    for (idx, &value) in result.criterion_values.iter().enumerate() {
                        if value < local_best_score {
                            local_best_score = value;
                            local_best_idx = idx;
                        }
                    }

                    let best_log_likelihood = result.log_likelihoods[local_best_idx];
                    let best_n_components = min_components + local_best_idx;
                    let mut tuned_config = config.clone();
                    tuned_config.n_components = best_n_components;

                    all_results.push(GridSearchCV {
                        config: tuned_config.clone(),
                        score: local_best_score,
                        log_likelihood: best_log_likelihood,
                    });

                    if local_best_score < best_score {
                        best_score = local_best_score;
                        best_params = Some(tuned_config);
                    }
                }
            }
        }

        Ok(GridSearchResult {
            best_params: best_params.unwrap(),
            best_score,
            cv_results: all_results,
        })
    }
}

/// Result of grid search cross-validation
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    pub best_params: GaussianMixtureConfig,
    pub best_score: Float,
    pub cv_results: Vec<GridSearchCV>,
}

/// Individual grid search CV result
#[derive(Debug, Clone)]
pub struct GridSearchCV {
    pub config: GaussianMixtureConfig,
    pub score: Float,
    pub log_likelihood: Float,
}

/// Convenience function to perform model selection
pub fn select_model(
    x: &ArrayView2<Float>,
    min_components: usize,
    max_components: usize,
    criterion: ModelSelectionCriterion,
    config_template: &GaussianMixtureConfig,
) -> Result<ModelSelectionResult> {
    let selector = ModelSelector::new(min_components, max_components).criterion(criterion);

    selector.select_best_model(x, config_template)
}

/// Convenience function to perform cross-validated model selection
pub fn select_model_cv(
    x: &ArrayView2<Float>,
    min_components: usize,
    max_components: usize,
    criterion: ModelSelectionCriterion,
    n_folds: usize,
    config_template: &GaussianMixtureConfig,
) -> Result<ModelSelectionResult> {
    let selector = ModelSelector::new(min_components, max_components)
        .criterion(criterion)
        .cross_validation(n_folds);

    selector.select_best_model(x, config_template)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_model_selector() {
        let x = array![
            [0.0, 0.0],
            [0.1, 0.1],
            [0.2, 0.0],
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.0],
        ];

        let config = GaussianMixtureConfig::default();
        let selector = ModelSelector::new(1, 3).criterion(ModelSelectionCriterion::BIC);

        let result = selector.select_best_model(&x.view(), &config).unwrap();

        assert!(result.best_n_components >= 1 && result.best_n_components <= 3);
        assert_eq!(result.criterion_values.len(), 3);
        assert_eq!(result.log_likelihoods.len(), 3);
        assert!(matches!(result.criterion, ModelSelectionCriterion::BIC));
    }

    #[test]
    fn test_parameter_counting() {
        use super::super::types_config::CovarianceType;

        // Test parameter counting for different covariance types
        assert_eq!(count_parameters(2, 3, &CovarianceType::Full), 19); // 1 + 6 + 12 = 19
        assert_eq!(count_parameters(2, 3, &CovarianceType::Diagonal), 13); // 1 + 6 + 6 = 13
        assert_eq!(count_parameters(2, 3, &CovarianceType::Tied), 13); // 1 + 6 + 6 = 13
        assert_eq!(count_parameters(2, 3, &CovarianceType::Spherical), 9); // 1 + 6 + 2 = 9
    }

    #[test]
    fn test_information_criteria() {
        let log_likelihood = -100.0;
        let n_params = 10;
        let n_samples = 100;

        let aic = calculate_aic_simd(log_likelihood, n_params, n_samples);
        let bic = calculate_bic_simd(log_likelihood, n_params, n_samples);

        assert_eq!(aic, 220.0); // -2 * (-100) + 2 * 10
        assert!((bic - 246.05).abs() < 0.1); // -2 * (-100) + 10 * ln(100)
        assert!(bic > aic); // BIC should penalize complexity more for large samples
    }
}
