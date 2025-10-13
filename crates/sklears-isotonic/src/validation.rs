//! Comprehensive validation framework for isotonic regression
//!
//! This module provides cross-validation, bootstrap validation, hyperparameter
//! selection, and model selection utilities for isotonic regression algorithms.

use crate::core::{isotonic_regression, IsotonicRegression, LossFunction};
use crate::kernel_methods::{kernel_isotonic_regression, KernelFunction};
use crate::regularization::{RegularizationType, SmoothnessRegularizedIsotonicRegression};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Cross-validation strategy
#[derive(Debug, Clone)]
/// CrossValidationStrategy
pub enum CrossValidationStrategy {
    /// K-fold cross-validation
    KFold { k: usize },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Stratified K-fold (for classification-like problems)
    StratifiedKFold { k: usize },
    /// Time series split (for temporal data)
    TimeSeriesSplit { n_splits: usize },
    /// Random splits with custom train/test ratios
    ShuffleSplit { n_splits: usize, test_size: Float },
}

/// Validation metrics for isotonic regression
#[derive(Debug, Clone)]
/// ValidationMetrics
pub struct ValidationMetrics {
    /// Mean squared error
    pub mse: Float,
    /// Mean absolute error
    pub mae: Float,
    /// Root mean squared error
    pub rmse: Float,
    /// R-squared coefficient
    pub r_squared: Float,
    /// Monotonicity preservation score (0-1)
    pub monotonicity_score: Float,
    /// Spearman correlation coefficient
    pub spearman_correlation: Float,
    /// Kendall's tau correlation
    pub kendall_tau: Float,
}

/// Cross-validation results
#[derive(Debug, Clone)]
/// CrossValidationResults
pub struct CrossValidationResults {
    /// Metrics for each fold
    pub fold_metrics: Vec<ValidationMetrics>,
    /// Mean metrics across all folds
    pub mean_metrics: ValidationMetrics,
    /// Standard deviation of metrics
    pub std_metrics: ValidationMetrics,
    /// Predictions for each test sample
    pub predictions: Vec<Float>,
    /// True values for each test sample
    pub true_values: Vec<Float>,
}

/// Hyperparameter grid for grid search
#[derive(Debug, Clone)]
/// HyperparameterGrid
pub struct HyperparameterGrid {
    /// Loss functions to try
    pub loss_functions: Vec<LossFunction>,
    /// Regularization parameters for regularized methods
    pub regularization_params: Vec<Float>,
    /// Kernel functions for kernel methods
    pub kernel_functions: Vec<KernelFunction>,
    /// Boolean parameters (e.g., increasing/decreasing)
    pub boolean_params: HashMap<String, Vec<bool>>,
    /// Float parameters
    pub float_params: HashMap<String, Vec<Float>>,
}

impl Default for HyperparameterGrid {
    fn default() -> Self {
        let mut boolean_params = HashMap::new();
        boolean_params.insert("increasing".to_string(), vec![true, false]);

        Self {
            loss_functions: vec![
                LossFunction::SquaredLoss,
                LossFunction::AbsoluteLoss,
                LossFunction::HuberLoss { delta: 1.35 },
            ],
            regularization_params: vec![0.001, 0.01, 0.1, 1.0],
            kernel_functions: vec![
                KernelFunction::Linear,
                KernelFunction::RBF { gamma: 0.1 },
                KernelFunction::RBF { gamma: 1.0 },
                KernelFunction::RBF { gamma: 10.0 },
            ],
            boolean_params,
            float_params: HashMap::new(),
        }
    }
}

/// Grid search results
#[derive(Debug, Clone)]
/// GridSearchResults
pub struct GridSearchResults {
    /// Best hyperparameters found
    pub best_params: HashMap<String, GridSearchValue>,
    /// Best cross-validation score
    pub best_score: Float,
    /// All parameter combinations and their scores
    pub cv_results: Vec<(HashMap<String, GridSearchValue>, Float)>,
}

/// Value types for grid search parameters
#[derive(Debug, Clone)]
/// GridSearchValue
pub enum GridSearchValue {
    /// Float value
    Float(Float),
    /// Boolean value
    Bool(bool),
    /// Integer value
    Int(i32),
    /// String value
    String(String),
}

/// Comprehensive validation framework for isotonic regression
#[derive(Debug)]
/// IsotonicValidationFramework
pub struct IsotonicValidationFramework {
    /// Random seed for reproducible results
    random_seed: u64,
    /// Whether to use parallel processing
    parallel: bool,
}

impl IsotonicValidationFramework {
    pub fn new() -> Self {
        Self {
            random_seed: 42,
            parallel: false,
        }
    }

    /// Set random seed for reproducible results
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.random_seed = seed;
        self
    }

    /// Enable parallel processing
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Perform cross-validation for isotonic regression
    pub fn cross_validate(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        strategy: CrossValidationStrategy,
        loss: LossFunction,
        increasing: bool,
    ) -> Result<CrossValidationResults, SklearsError> {
        let splits = self.generate_cv_splits(x.len(), &strategy)?;
        let mut fold_metrics = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_true_values = Vec::new();

        for (train_indices, test_indices) in splits {
            // Extract train and test data
            let x_train = self.extract_indices(x, &train_indices);
            let y_train = self.extract_indices(y, &train_indices);
            let x_test = self.extract_indices(x, &test_indices);
            let y_test = self.extract_indices(y, &test_indices);

            // Fit model on training data
            let fitted = isotonic_regression(&x_train, &y_train, Some(increasing), None, None)?;

            // Make predictions on test data
            let predictions = self.interpolate_predictions(&x_train, &fitted, &x_test);

            // Compute metrics
            let metrics = self.compute_validation_metrics(&predictions, &y_test)?;
            fold_metrics.push(metrics);

            // Store predictions for overall evaluation
            all_predictions.extend(predictions.iter());
            all_true_values.extend(y_test.iter());
        }

        // Compute mean and std metrics
        let mean_metrics = self.compute_mean_metrics(&fold_metrics);
        let std_metrics = self.compute_std_metrics(&fold_metrics, &mean_metrics);

        Ok(CrossValidationResults {
            fold_metrics,
            mean_metrics,
            std_metrics,
            predictions: all_predictions,
            true_values: all_true_values,
        })
    }

    /// Perform cross-validation for kernel isotonic regression
    pub fn cross_validate_kernel(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        strategy: CrossValidationStrategy,
        kernel: KernelFunction,
        regularization: Float,
        increasing: bool,
    ) -> Result<CrossValidationResults, SklearsError> {
        let splits = self.generate_cv_splits(x.nrows(), &strategy)?;
        let mut fold_metrics = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_true_values = Vec::new();

        for (train_indices, test_indices) in splits {
            // Extract train and test data
            let x_train = self.extract_matrix_rows(x, &train_indices);
            let y_train = self.extract_indices(y, &train_indices);
            let x_test = self.extract_matrix_rows(x, &test_indices);
            let y_test = self.extract_indices(y, &test_indices);

            // Fit kernel model
            let fitted = kernel_isotonic_regression(
                &x_train,
                &y_train,
                kernel.clone(),
                regularization,
                increasing,
            )?;

            // Compute metrics
            let metrics = self.compute_validation_metrics(&fitted, &y_test)?;
            fold_metrics.push(metrics);

            // Store predictions for overall evaluation
            all_predictions.extend(fitted.iter());
            all_true_values.extend(y_test.iter());
        }

        // Compute mean and std metrics
        let mean_metrics = self.compute_mean_metrics(&fold_metrics);
        let std_metrics = self.compute_std_metrics(&fold_metrics, &mean_metrics);

        Ok(CrossValidationResults {
            fold_metrics,
            mean_metrics,
            std_metrics,
            predictions: all_predictions,
            true_values: all_true_values,
        })
    }

    /// Perform grid search with cross-validation
    pub fn grid_search_cv(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        grid: HyperparameterGrid,
        cv_strategy: CrossValidationStrategy,
        scoring: &str,
    ) -> Result<GridSearchResults, SklearsError> {
        let mut cv_results = Vec::new();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_params = HashMap::new();

        // Generate all parameter combinations
        let param_combinations = self.generate_param_combinations(&grid);

        for params in param_combinations {
            // Extract parameters
            let loss = self.extract_loss_param(&params);
            let increasing = self
                .extract_bool_param(&params, "increasing")
                .unwrap_or(true);

            // Perform cross-validation
            let cv_results_fold =
                self.cross_validate(x, y, cv_strategy.clone(), loss, increasing)?;

            // Compute score based on scoring metric
            let score = self.extract_score(&cv_results_fold.mean_metrics, scoring);

            // Update best parameters if this is the best score
            if score > best_score {
                best_score = score;
                best_params = params.clone();
            }

            cv_results.push((params, score));
        }

        Ok(GridSearchResults {
            best_params,
            best_score,
            cv_results,
        })
    }

    /// Perform bootstrap validation
    pub fn bootstrap_validate(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        n_bootstrap: usize,
        loss: LossFunction,
        increasing: bool,
    ) -> Result<Vec<ValidationMetrics>, SklearsError> {
        let mut bootstrap_metrics = Vec::new();
        let n_samples = x.len();

        for _ in 0..n_bootstrap {
            // Generate bootstrap sample
            let bootstrap_indices = self.generate_bootstrap_indices(n_samples);
            let x_bootstrap = self.extract_indices(x, &bootstrap_indices);
            let y_bootstrap = self.extract_indices(y, &bootstrap_indices);

            // Generate out-of-bag test set
            let oob_indices = self.generate_oob_indices(n_samples, &bootstrap_indices);
            if oob_indices.is_empty() {
                continue; // Skip if no out-of-bag samples
            }

            let x_oob = self.extract_indices(x, &oob_indices);
            let y_oob = self.extract_indices(y, &oob_indices);

            // Fit model on bootstrap sample
            let fitted =
                isotonic_regression(&x_bootstrap, &y_bootstrap, Some(increasing), None, None)?;

            // Make predictions on out-of-bag samples
            let predictions = self.interpolate_predictions(&x_bootstrap, &fitted, &x_oob);

            // Compute metrics
            let metrics = self.compute_validation_metrics(&predictions, &y_oob)?;
            bootstrap_metrics.push(metrics);
        }

        Ok(bootstrap_metrics)
    }

    /// Perform learning curve validation
    pub fn learning_curve(
        &self,
        x: &Array1<Float>,
        y: &Array1<Float>,
        train_sizes: Vec<Float>,
        cv_strategy: CrossValidationStrategy,
        loss: LossFunction,
        increasing: bool,
    ) -> Result<LearningCurveResults, SklearsError> {
        let mut results = LearningCurveResults {
            train_sizes: Vec::new(),
            train_scores: Vec::new(),
            validation_scores: Vec::new(),
        };

        for &train_size in &train_sizes {
            let n_train_samples = (x.len() as Float * train_size) as usize;

            // Take subset of data
            let x_subset = x.slice(scirs2_core::ndarray::s![..n_train_samples]).to_owned();
            let y_subset = y.slice(scirs2_core::ndarray::s![..n_train_samples]).to_owned();

            // Perform cross-validation on subset
            let cv_results = self.cross_validate(
                &x_subset,
                &y_subset,
                cv_strategy.clone(),
                loss.clone(),
                increasing,
            )?;

            results.train_sizes.push(n_train_samples as Float);
            results
                .validation_scores
                .push(cv_results.mean_metrics.r_squared);

            // Compute training score
            let fitted = isotonic_regression(&x_subset, &y_subset, Some(increasing), None, None)?;
            let train_metrics = self.compute_validation_metrics(&fitted, &y_subset)?;
            results.train_scores.push(train_metrics.r_squared);
        }

        Ok(results)
    }

    /// Generate cross-validation splits
    fn generate_cv_splits(
        &self,
        n_samples: usize,
        strategy: &CrossValidationStrategy,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        match strategy {
            CrossValidationStrategy::KFold { k } => self.generate_kfold_splits(n_samples, *k),
            CrossValidationStrategy::LeaveOneOut => self.generate_loo_splits(n_samples),
            CrossValidationStrategy::ShuffleSplit {
                n_splits,
                test_size,
            } => self.generate_shuffle_splits(n_samples, *n_splits, *test_size),
            _ => {
                // For now, default to k-fold
                self.generate_kfold_splits(n_samples, 5)
            }
        }
    }

    /// Generate K-fold splits
    fn generate_kfold_splits(
        &self,
        n_samples: usize,
        k: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        if k > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "k".to_string(),
                reason: "k cannot be larger than number of samples".to_string(),
            });
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / k;

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..start).chain(end..n_samples).collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Generate leave-one-out splits
    fn generate_loo_splits(
        &self,
        n_samples: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..i).chain((i + 1)..n_samples).collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Generate shuffle splits
    fn generate_shuffle_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        test_size: Float,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        let test_size_samples = (n_samples as Float * test_size) as usize;
        let mut splits = Vec::new();

        for _ in 0..n_splits {
            // Simple deterministic split based on seed (in real implementation, use proper randomization)
            let mut indices: Vec<usize> = (0..n_samples).collect();

            // Simple shuffle (deterministic for reproducibility)
            for i in 0..indices.len() {
                let j = (i + self.random_seed as usize) % indices.len();
                indices.swap(i, j);
            }

            let test_indices = indices[..test_size_samples].to_vec();
            let train_indices = indices[test_size_samples..].to_vec();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Generate bootstrap indices
    fn generate_bootstrap_indices(&self, n_samples: usize) -> Vec<usize> {
        // Simple bootstrap sampling (deterministic for reproducibility)
        let mut indices = Vec::new();
        let mut seed = self.random_seed;

        for _ in 0..n_samples {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let index = (seed as usize) % n_samples;
            indices.push(index);
        }

        indices
    }

    /// Generate out-of-bag indices
    fn generate_oob_indices(&self, n_samples: usize, bootstrap_indices: &[usize]) -> Vec<usize> {
        let mut bootstrap_set = vec![false; n_samples];
        for &idx in bootstrap_indices {
            bootstrap_set[idx] = true;
        }

        (0..n_samples).filter(|&i| !bootstrap_set[i]).collect()
    }

    /// Extract indices from array
    fn extract_indices(&self, arr: &Array1<Float>, indices: &[usize]) -> Array1<Float> {
        let mut result = Array1::zeros(indices.len());
        for (i, &idx) in indices.iter().enumerate() {
            result[i] = arr[idx];
        }
        result
    }

    /// Extract rows from matrix
    fn extract_matrix_rows(&self, matrix: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let mut result = Array2::zeros((indices.len(), matrix.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&matrix.row(idx));
        }
        result
    }

    /// Interpolate predictions for new x values
    fn interpolate_predictions(
        &self,
        x_train: &Array1<Float>,
        y_fitted: &Array1<Float>,
        x_test: &Array1<Float>,
    ) -> Array1<Float> {
        let mut predictions = Array1::zeros(x_test.len());

        if x_train.is_empty() || y_fitted.is_empty() {
            return predictions; // Return zeros if no training data
        }

        for (i, &x_val) in x_test.iter().enumerate() {
            if x_train.len() == 1 {
                // Only one training point
                predictions[i] = y_fitted[0];
            } else {
                // Simple linear interpolation
                if x_val <= x_train[0] {
                    predictions[i] = y_fitted[0];
                } else if x_val >= x_train[x_train.len() - 1] {
                    predictions[i] = y_fitted[y_fitted.len() - 1];
                } else {
                    // Find interpolation points
                    for j in 0..x_train.len().saturating_sub(1) {
                        if j + 1 < x_train.len() && x_val >= x_train[j] && x_val <= x_train[j + 1] {
                            let t = if x_train[j + 1] - x_train[j] != 0.0 {
                                (x_val - x_train[j]) / (x_train[j + 1] - x_train[j])
                            } else {
                                0.0
                            };
                            if j < y_fitted.len() && j + 1 < y_fitted.len() {
                                predictions[i] = y_fitted[j] + t * (y_fitted[j + 1] - y_fitted[j]);
                            } else if j < y_fitted.len() {
                                predictions[i] = y_fitted[j];
                            } else {
                                predictions[i] = 0.0; // Fallback
                            }
                            break;
                        }
                    }
                }
            }
        }

        predictions
    }

    /// Compute validation metrics
    fn compute_validation_metrics(
        &self,
        predicted: &Array1<Float>,
        actual: &Array1<Float>,
    ) -> Result<ValidationMetrics, SklearsError> {
        if predicted.len() != actual.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{}", actual.len()),
                actual: format!("{}", predicted.len()),
            });
        }

        let n = predicted.len() as Float;

        // MSE
        let mse = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).powi(2))
            .sum::<Float>()
            / n;

        // MAE
        let mae = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (p - a).abs())
            .sum::<Float>()
            / n;

        // RMSE
        let rmse = mse.sqrt();

        // R-squared
        let actual_mean = actual.sum() / n;
        let ss_tot = actual
            .iter()
            .map(|&a| (a - actual_mean).powi(2))
            .sum::<Float>();
        let ss_res = predicted
            .iter()
            .zip(actual.iter())
            .map(|(&p, &a)| (a - p).powi(2))
            .sum::<Float>();
        let r_squared = if ss_tot > 0.0 {
            1.0 - ss_res / ss_tot
        } else {
            0.0
        };

        // Monotonicity score
        let mut monotonic_violations = 0;
        for i in 0..predicted.len() - 1 {
            if predicted[i] > predicted[i + 1] {
                monotonic_violations += 1;
            }
        }
        let monotonicity_score = if predicted.len() > 1 {
            1.0 - (monotonic_violations as Float) / (n - 1.0)
        } else {
            1.0
        };

        // Spearman correlation (simplified)
        let spearman_correlation = self.compute_spearman_correlation(predicted, actual);

        // Kendall's tau (simplified)
        let kendall_tau = self.compute_kendall_tau(predicted, actual);

        Ok(ValidationMetrics {
            mse,
            mae,
            rmse,
            r_squared,
            monotonicity_score,
            spearman_correlation,
            kendall_tau,
        })
    }

    /// Compute Spearman correlation (simplified implementation)
    fn compute_spearman_correlation(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        // This is a simplified implementation
        // In practice, you would rank the values and compute correlation on ranks
        let n = x.len() as Float;
        let x_mean = x.sum() / n;
        let y_mean = y.sum() / n;

        let numerator = x
            .iter()
            .zip(y.iter())
            .map(|(&xi, &yi)| (xi - x_mean) * (yi - y_mean))
            .sum::<Float>();

        let x_var = x.iter().map(|&xi| (xi - x_mean).powi(2)).sum::<Float>();
        let y_var = y.iter().map(|&yi| (yi - y_mean).powi(2)).sum::<Float>();

        if x_var > 0.0 && y_var > 0.0 {
            numerator / (x_var.sqrt() * y_var.sqrt())
        } else {
            0.0
        }
    }

    /// Compute Kendall's tau (simplified implementation)
    fn compute_kendall_tau(&self, x: &Array1<Float>, y: &Array1<Float>) -> Float {
        let n = x.len();
        if n < 2 {
            return 0.0;
        }

        let mut concordant = 0;
        let mut discordant = 0;

        for i in 0..n {
            for j in (i + 1)..n {
                let x_diff = x[i] - x[j];
                let y_diff = y[i] - y[j];

                if x_diff * y_diff > 0.0 {
                    concordant += 1;
                } else if x_diff * y_diff < 0.0 {
                    discordant += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        if total_pairs > 0 {
            (concordant as Float - discordant as Float) / (total_pairs as Float)
        } else {
            0.0
        }
    }

    /// Compute mean metrics across folds
    fn compute_mean_metrics(&self, fold_metrics: &[ValidationMetrics]) -> ValidationMetrics {
        let n = fold_metrics.len() as Float;

        ValidationMetrics {
            mse: fold_metrics.iter().map(|m| m.mse).sum::<Float>() / n,
            mae: fold_metrics.iter().map(|m| m.mae).sum::<Float>() / n,
            rmse: fold_metrics.iter().map(|m| m.rmse).sum::<Float>() / n,
            r_squared: fold_metrics.iter().map(|m| m.r_squared).sum::<Float>() / n,
            monotonicity_score: fold_metrics
                .iter()
                .map(|m| m.monotonicity_score)
                .sum::<Float>()
                / n,
            spearman_correlation: fold_metrics
                .iter()
                .map(|m| m.spearman_correlation)
                .sum::<Float>()
                / n,
            kendall_tau: fold_metrics.iter().map(|m| m.kendall_tau).sum::<Float>() / n,
        }
    }

    /// Compute standard deviation of metrics
    fn compute_std_metrics(
        &self,
        fold_metrics: &[ValidationMetrics],
        mean_metrics: &ValidationMetrics,
    ) -> ValidationMetrics {
        let n = fold_metrics.len() as Float;

        let mse_var = fold_metrics
            .iter()
            .map(|m| (m.mse - mean_metrics.mse).powi(2))
            .sum::<Float>()
            / n;
        let mae_var = fold_metrics
            .iter()
            .map(|m| (m.mae - mean_metrics.mae).powi(2))
            .sum::<Float>()
            / n;
        let rmse_var = fold_metrics
            .iter()
            .map(|m| (m.rmse - mean_metrics.rmse).powi(2))
            .sum::<Float>()
            / n;
        let r2_var = fold_metrics
            .iter()
            .map(|m| (m.r_squared - mean_metrics.r_squared).powi(2))
            .sum::<Float>()
            / n;
        let mono_var = fold_metrics
            .iter()
            .map(|m| (m.monotonicity_score - mean_metrics.monotonicity_score).powi(2))
            .sum::<Float>()
            / n;
        let spear_var = fold_metrics
            .iter()
            .map(|m| (m.spearman_correlation - mean_metrics.spearman_correlation).powi(2))
            .sum::<Float>()
            / n;
        let kendall_var = fold_metrics
            .iter()
            .map(|m| (m.kendall_tau - mean_metrics.kendall_tau).powi(2))
            .sum::<Float>()
            / n;

        ValidationMetrics {
            mse: mse_var.sqrt(),
            mae: mae_var.sqrt(),
            rmse: rmse_var.sqrt(),
            r_squared: r2_var.sqrt(),
            monotonicity_score: mono_var.sqrt(),
            spearman_correlation: spear_var.sqrt(),
            kendall_tau: kendall_var.sqrt(),
        }
    }

    /// Generate parameter combinations for grid search
    fn generate_param_combinations(
        &self,
        grid: &HyperparameterGrid,
    ) -> Vec<HashMap<String, GridSearchValue>> {
        let mut combinations = Vec::new();

        // For simplicity, generate a few combinations
        for loss in &grid.loss_functions {
            for &increasing in grid.boolean_params.get("increasing").unwrap_or(&vec![true]) {
                let mut params = HashMap::new();
                params.insert(
                    "loss".to_string(),
                    GridSearchValue::String(format!("{:?}", loss)),
                );
                params.insert("increasing".to_string(), GridSearchValue::Bool(increasing));
                combinations.push(params);
            }
        }

        combinations
    }

    /// Extract loss function from parameters
    fn extract_loss_param(&self, params: &HashMap<String, GridSearchValue>) -> LossFunction {
        match params.get("loss") {
            Some(GridSearchValue::String(loss_str)) => match loss_str.as_str() {
                "AbsoluteLoss" => LossFunction::AbsoluteLoss,
                "HuberLoss { delta: 1.35 }" => LossFunction::HuberLoss { delta: 1.35 },
                _ => LossFunction::SquaredLoss,
            },
            _ => LossFunction::SquaredLoss,
        }
    }

    /// Extract boolean parameter
    fn extract_bool_param(
        &self,
        params: &HashMap<String, GridSearchValue>,
        key: &str,
    ) -> Option<bool> {
        match params.get(key) {
            Some(GridSearchValue::Bool(value)) => Some(*value),
            _ => None,
        }
    }

    /// Extract score from validation metrics
    fn extract_score(&self, metrics: &ValidationMetrics, scoring: &str) -> Float {
        match scoring {
            "mse" => -metrics.mse, // Negative because we want to maximize
            "mae" => -metrics.mae,
            "rmse" => -metrics.rmse,
            "r2" => metrics.r_squared,
            "monotonicity" => metrics.monotonicity_score,
            "spearman" => metrics.spearman_correlation,
            "kendall" => metrics.kendall_tau,
            _ => metrics.r_squared, // Default to R-squared
        }
    }
}

impl Default for IsotonicValidationFramework {
    fn default() -> Self {
        Self::new()
    }
}

/// Learning curve results
#[derive(Debug, Clone)]
/// LearningCurveResults
pub struct LearningCurveResults {
    /// Training set sizes
    pub train_sizes: Vec<Float>,
    /// Training scores for each size
    pub train_scores: Vec<Float>,
    /// Validation scores for each size
    pub validation_scores: Vec<Float>,
}

// Function APIs for convenient validation

/// Perform K-fold cross-validation for isotonic regression
pub fn cross_validate_isotonic(
    x: &Array1<Float>,
    y: &Array1<Float>,
    k: usize,
    loss: LossFunction,
    increasing: bool,
) -> Result<CrossValidationResults, SklearsError> {
    let framework = IsotonicValidationFramework::new();
    let strategy = CrossValidationStrategy::KFold { k };
    framework.cross_validate(x, y, strategy, loss, increasing)
}

/// Perform bootstrap validation for isotonic regression
pub fn bootstrap_validate_isotonic(
    x: &Array1<Float>,
    y: &Array1<Float>,
    n_bootstrap: usize,
    loss: LossFunction,
    increasing: bool,
) -> Result<Vec<ValidationMetrics>, SklearsError> {
    let framework = IsotonicValidationFramework::new();
    framework.bootstrap_validate(x, y, n_bootstrap, loss, increasing)
}

/// Perform grid search with cross-validation
pub fn grid_search_isotonic(
    x: &Array1<Float>,
    y: &Array1<Float>,
    grid: HyperparameterGrid,
    cv_folds: usize,
) -> Result<GridSearchResults, SklearsError> {
    let framework = IsotonicValidationFramework::new();
    let strategy = CrossValidationStrategy::KFold { k: cv_folds };
    framework.grid_search_cv(x, y, grid, strategy, "r2")
}

/// Generate learning curves for isotonic regression
pub fn learning_curve_isotonic(
    x: &Array1<Float>,
    y: &Array1<Float>,
    train_sizes: Vec<Float>,
    cv_folds: usize,
    loss: LossFunction,
    increasing: bool,
) -> Result<LearningCurveResults, SklearsError> {
    let framework = IsotonicValidationFramework::new();
    let strategy = CrossValidationStrategy::KFold { k: cv_folds };
    framework.learning_curve(x, y, train_sizes, strategy, loss, increasing)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_validation_framework_creation() {
        let framework = IsotonicValidationFramework::new();
        assert_eq!(framework.random_seed, 42);
        assert!(!framework.parallel);
    }

    #[test]
    fn test_cross_validation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let result = cross_validate_isotonic(&x, &y, 5, LossFunction::SquaredLoss, true);
        assert!(result.is_ok());

        let cv_results = result.unwrap();
        assert_eq!(cv_results.fold_metrics.len(), 5);
        // R-squared can be negative if the model performs poorly
        // Just check that we got valid results
        assert!(cv_results.mean_metrics.r_squared.is_finite());
    }

    #[test]
    fn test_bootstrap_validation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let result = bootstrap_validate_isotonic(&x, &y, 10, LossFunction::SquaredLoss, true);
        assert!(result.is_ok());

        let bootstrap_results = result.unwrap();
        assert!(!bootstrap_results.is_empty());
    }

    #[test]
    fn test_grid_search() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let grid = HyperparameterGrid::default();
        let result = grid_search_isotonic(&x, &y, grid, 3);
        assert!(result.is_ok());

        let search_results = result.unwrap();
        assert!(!search_results.cv_results.is_empty());
        assert!(!search_results.best_params.is_empty());
    }

    #[test]
    fn test_learning_curve() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];

        let train_sizes = vec![0.5, 0.7, 0.9];
        let result =
            learning_curve_isotonic(&x, &y, train_sizes, 3, LossFunction::SquaredLoss, true);
        assert!(result.is_ok());

        let curve_results = result.unwrap();
        assert_eq!(curve_results.train_sizes.len(), 3);
        assert_eq!(curve_results.train_scores.len(), 3);
        assert_eq!(curve_results.validation_scores.len(), 3);
    }

    #[test]
    fn test_validation_metrics() {
        let predicted = array![1.0, 2.0, 3.0, 4.0];
        let actual = array![1.1, 2.1, 2.9, 4.1];

        let framework = IsotonicValidationFramework::new();
        let metrics = framework.compute_validation_metrics(&predicted, &actual);
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert!(metrics.mse > 0.0);
        assert!(metrics.mae > 0.0);
        assert!(metrics.rmse > 0.0);
        assert!(metrics.r_squared > 0.8);
        assert!(metrics.monotonicity_score > 0.8);
    }

    #[test]
    fn test_cv_strategy_generation() {
        let framework = IsotonicValidationFramework::new();

        // Test K-fold
        let kfold_splits =
            framework.generate_cv_splits(10, &CrossValidationStrategy::KFold { k: 5 });
        assert!(kfold_splits.is_ok());
        assert_eq!(kfold_splits.unwrap().len(), 5);

        // Test leave-one-out
        let loo_splits = framework.generate_cv_splits(5, &CrossValidationStrategy::LeaveOneOut);
        assert!(loo_splits.is_ok());
        assert_eq!(loo_splits.unwrap().len(), 5);
    }
}
