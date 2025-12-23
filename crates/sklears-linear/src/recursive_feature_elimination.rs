//! Recursive Feature Elimination (RFE) for linear models
//!
//! This module provides recursive feature elimination, a wrapper feature selection
//! method that fits a model and recursively removes features based on their importance,
//! retraining the model with remaining features until the desired number is reached.

use crate::lasso_cv::{KFold, LassoCV, LassoCVConfig};
use crate::linear_regression::{LinearRegression, LinearRegressionConfig};
use crate::ridge_cv::{RidgeCV, RidgeCVConfig};
use crate::Penalty;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::SklearsError,
    traits::{Fit, Predict, Trained},
};
use std::cmp::Ordering;

/// Base estimator types for RFE
#[derive(Debug, Clone)]
pub enum RFEEstimator {
    /// Linear regression
    LinearRegression { config: LinearRegressionConfig },
    /// Ridge regression with cross-validation
    RidgeCV { config: RidgeCVConfig },
    /// Lasso regression with cross-validation
    LassoCV { config: LassoCVConfig },
}

/// RFE configuration
#[derive(Debug, Clone)]
pub struct RFEConfig {
    /// Base estimator to use for feature importance
    pub estimator: RFEEstimator,
    /// Number of features to select
    pub n_features_to_select: Option<usize>,
    /// Number of features to remove at each iteration
    pub step: usize,
    /// Whether to use cross-validation for scoring
    pub cv: Option<usize>,
    /// Scoring metric for cross-validation
    pub scoring: ScoringMetric,
    /// Verbose output
    pub verbose: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for RFEConfig {
    fn default() -> Self {
        Self {
            estimator: RFEEstimator::LinearRegression {
                config: LinearRegressionConfig::default(),
            },
            n_features_to_select: None,
            step: 1,
            cv: Some(5),
            scoring: ScoringMetric::R2,
            verbose: false,
            random_state: None,
        }
    }
}

/// Scoring metrics for RFE
#[derive(Debug, Clone, PartialEq)]
pub enum ScoringMetric {
    /// R-squared (coefficient of determination)
    R2,
    /// Mean squared error (negative)
    NegMeanSquaredError,
    /// Mean absolute error (negative)
    NegMeanAbsoluteError,
    /// Root mean squared error (negative)
    NegRootMeanSquaredError,
}

/// RFE feature information
#[derive(Debug, Clone)]
pub struct RFEFeatureInfo {
    /// Original feature index
    pub feature_index: usize,
    /// Elimination ranking (1 = selected, higher = eliminated earlier)
    pub ranking: usize,
    /// Feature importance at elimination
    pub importance: f64,
    /// Whether feature is selected
    pub selected: bool,
}

/// RFE result
#[derive(Debug, Clone)]
pub struct RFEResult {
    /// Selected feature indices
    pub selected_features: Vec<usize>,
    /// Feature information for all features
    pub feature_info: Vec<RFEFeatureInfo>,
    /// Cross-validation scores at each step
    pub cv_scores: Vec<f64>,
    /// Number of features at each elimination step
    pub n_features_steps: Vec<usize>,
    /// Final estimator fitted on selected features
    pub estimator_coefficients: Vec<f64>,
    /// Number of original features
    pub n_features_in: usize,
    /// Number of selected features
    pub n_features_out: usize,
    /// Configuration used
    pub config: RFEConfig,
}

/// Recursive Feature Elimination selector
pub struct RecursiveFeatureElimination {
    config: RFEConfig,
    is_fitted: bool,
    rfe_result: Option<RFEResult>,
}

impl RecursiveFeatureElimination {
    /// Create a new RFE selector with default configuration
    pub fn new() -> Self {
        Self {
            config: RFEConfig::default(),
            is_fitted: false,
            rfe_result: None,
        }
    }

    /// Create an RFE selector with custom configuration
    pub fn with_config(config: RFEConfig) -> Self {
        Self {
            config,
            is_fitted: false,
            rfe_result: None,
        }
    }

    /// Set the base estimator
    pub fn with_estimator(mut self, estimator: RFEEstimator) -> Self {
        self.config.estimator = estimator;
        self
    }

    /// Set the number of features to select
    pub fn with_n_features_to_select(mut self, n_features: Option<usize>) -> Self {
        self.config.n_features_to_select = n_features;
        self
    }

    /// Set the elimination step size
    pub fn with_step(mut self, step: usize) -> Self {
        self.config.step = step.max(1);
        self
    }

    /// Enable cross-validation with specified folds
    pub fn with_cv(mut self, cv: Option<usize>) -> Self {
        self.config.cv = cv;
        self
    }

    /// Set the scoring metric
    pub fn with_scoring(mut self, scoring: ScoringMetric) -> Self {
        self.config.scoring = scoring;
        self
    }

    /// Fit the RFE selector to data
    pub fn fit(&mut self, x: &[Vec<f64>], y: &[f64]) -> Result<(), SklearsError> {
        if x.is_empty() || y.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit RFE on empty dataset".to_string(),
            ));
        }

        let n_samples = x.len();
        let n_features = x[0].len();

        if y.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("target.len() == {}", n_samples),
                actual: format!("target.len() == {}", y.len()),
            });
        }

        // Validate that all rows have the same number of features
        for (i, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("row[{}].len() == {}", i, n_features),
                    actual: format!("row[{}].len() == {}", i, row.len()),
                });
            }
        }

        // Determine target number of features
        let n_features_to_select = self
            .config
            .n_features_to_select
            .unwrap_or(n_features / 2)
            .min(n_features)
            .max(1);

        if self.config.verbose {
            eprintln!(
                "RFE: Starting with {} features, selecting {}",
                n_features, n_features_to_select
            );
        }

        // Initialize feature tracking
        let mut remaining_features: Vec<usize> = (0..n_features).collect();
        let mut feature_info: Vec<RFEFeatureInfo> = (0..n_features)
            .map(|i| RFEFeatureInfo {
                feature_index: i,
                ranking: 0,
                importance: 0.0,
                selected: false,
            })
            .collect();

        let mut cv_scores = Vec::new();
        let mut n_features_steps = Vec::new();
        let mut elimination_step = 0;

        // Recursive feature elimination loop
        while remaining_features.len() > n_features_to_select {
            if self.config.verbose {
                eprintln!(
                    "RFE: Step {}, {} features remaining",
                    elimination_step,
                    remaining_features.len()
                );
            }

            // Create subset of data with remaining features
            let x_subset = self.create_feature_subset(x, &remaining_features);

            // Fit estimator and get feature importance
            let importance_scores = self.fit_estimator_and_get_importance(&x_subset, y)?;

            // Calculate cross-validation score if requested
            if let Some(cv_folds) = self.config.cv {
                let cv_score = self.cross_validate_score(&x_subset, y, cv_folds)?;
                cv_scores.push(cv_score);
                n_features_steps.push(remaining_features.len());
            }

            // Determine number of features to eliminate this step
            let n_to_eliminate = self
                .config
                .step
                .min(remaining_features.len() - n_features_to_select);

            // Find features with lowest importance
            let mut feature_importance_pairs: Vec<(usize, f64)> = importance_scores
                .iter()
                .enumerate()
                .map(|(local_idx, &importance)| (remaining_features[local_idx], importance))
                .collect();

            feature_importance_pairs
                .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));

            // Eliminate features with lowest importance
            let eliminated_features: Vec<usize> = feature_importance_pairs
                .iter()
                .take(n_to_eliminate)
                .map(|(global_idx, importance)| {
                    feature_info[*global_idx].ranking = remaining_features.len() - elimination_step;
                    feature_info[*global_idx].importance = *importance;
                    *global_idx
                })
                .collect();

            // Remove eliminated features from remaining set
            remaining_features.retain(|&f| !eliminated_features.contains(&f));

            elimination_step += 1;
        }

        // Mark selected features
        for &feature_idx in &remaining_features {
            feature_info[feature_idx].selected = true;
            feature_info[feature_idx].ranking = 1; // Selected features get rank 1
        }

        // Fit final estimator on selected features
        let x_final = self.create_feature_subset(x, &remaining_features);
        let final_coefficients = self.fit_final_estimator(&x_final, y)?;

        // Add final cross-validation score
        if let Some(cv_folds) = self.config.cv {
            let final_cv_score = self.cross_validate_score(&x_final, y, cv_folds)?;
            cv_scores.push(final_cv_score);
            n_features_steps.push(remaining_features.len());
        }

        self.rfe_result = Some(RFEResult {
            selected_features: remaining_features,
            feature_info,
            cv_scores,
            n_features_steps,
            estimator_coefficients: final_coefficients,
            n_features_in: n_features,
            n_features_out: n_features_to_select,
            config: self.config.clone(),
        });

        if self.config.verbose {
            eprintln!("RFE: Completed. Selected {} features", n_features_to_select);
        }

        self.is_fitted = true;
        Ok(())
    }

    /// Transform data by selecting only the chosen features
    pub fn transform(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, SklearsError> {
        if !self.is_fitted {
            return Err(SklearsError::NotFitted {
                operation: "transform".to_string(),
            });
        }

        let rfe_result = self.rfe_result.as_ref().unwrap();

        if x.is_empty() {
            return Ok(Vec::new());
        }

        let n_features_in = x[0].len();
        if n_features_in != rfe_result.n_features_in {
            return Err(SklearsError::FeatureMismatch {
                expected: rfe_result.n_features_in,
                actual: n_features_in,
            });
        }

        Ok(self.create_feature_subset(x, &rfe_result.selected_features))
    }

    /// Fit and transform data in one step
    pub fn fit_transform(
        &mut self,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<Vec<f64>>, SklearsError> {
        self.fit(x, y)?;
        self.transform(x)
    }

    /// Get the RFE result
    pub fn get_rfe_result(&self) -> Option<&RFEResult> {
        self.rfe_result.as_ref()
    }

    /// Get selected feature indices
    pub fn get_selected_features(&self) -> Option<&Vec<usize>> {
        self.rfe_result.as_ref().map(|r| &r.selected_features)
    }

    /// Get feature rankings
    pub fn get_feature_ranking(&self) -> Option<Vec<usize>> {
        self.rfe_result
            .as_ref()
            .map(|r| r.feature_info.iter().map(|info| info.ranking).collect())
    }

    /// Get cross-validation scores
    pub fn get_cv_scores(&self) -> Option<&Vec<f64>> {
        self.rfe_result.as_ref().map(|r| &r.cv_scores)
    }

    /// Create a subset of features from the data
    fn create_feature_subset(&self, x: &[Vec<f64>], feature_indices: &[usize]) -> Vec<Vec<f64>> {
        x.iter()
            .map(|row| feature_indices.iter().map(|&idx| row[idx]).collect())
            .collect()
    }

    /// Fit estimator and return feature importance scores
    fn fit_estimator_and_get_importance(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
    ) -> Result<Vec<f64>, SklearsError> {
        match &self.config.estimator {
            RFEEstimator::LinearRegression { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_linear_with_fallback(&x_array, &y_array, config)?;
                Ok(trained.coef().iter().map(|&coef| coef.abs()).collect())
            }

            RFEEstimator::RidgeCV { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_ridge(&x_array, &y_array, config)?;
                Ok(trained.coef().iter().map(|&coef| coef.abs()).collect())
            }

            RFEEstimator::LassoCV { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_lasso(&x_array, &y_array, config)?;
                Ok(trained.coef().iter().map(|&coef| coef.abs()).collect())
            }
        }
    }

    /// Fit final estimator and return coefficients
    fn fit_final_estimator(&self, x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, SklearsError> {
        match &self.config.estimator {
            RFEEstimator::LinearRegression { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_linear_with_fallback(&x_array, &y_array, config)?;
                Ok(trained.coef().to_vec())
            }

            RFEEstimator::RidgeCV { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_ridge(&x_array, &y_array, config)?;
                Ok(trained.coef().to_vec())
            }

            RFEEstimator::LassoCV { config } => {
                let x_array = Self::to_array2(x)?;
                let y_array = Array1::from_vec(y.to_vec());
                let trained = Self::fit_lasso(&x_array, &y_array, config)?;
                Ok(trained.coef().to_vec())
            }
        }
    }

    /// Perform cross-validation and return average score
    fn cross_validate_score(
        &self,
        x: &[Vec<f64>],
        y: &[f64],
        cv_folds: usize,
    ) -> Result<f64, SklearsError> {
        if cv_folds == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "cv".to_string(),
                reason: "must be at least 1".to_string(),
            });
        }

        let n_samples = x.len();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot perform cross-validation on empty dataset".to_string(),
            ));
        }

        let x_array = Self::to_array2(x)?;
        let y_array = Array1::from_vec(y.to_vec());
        let n_folds = cv_folds.min(n_samples);
        let kfold = KFold::new(n_folds);
        let splits = kfold.split(n_samples, None);

        if splits.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cross-validation produced no folds".to_string(),
            ));
        }

        let mut scores = Vec::new();

        for (train_idx, test_idx) in splits {
            if train_idx.is_empty() || test_idx.is_empty() {
                continue;
            }

            let x_train = x_array.select(Axis(0), &train_idx);
            let y_train = y_array.select(Axis(0), &train_idx);
            let x_val = x_array.select(Axis(0), &test_idx);
            let y_val = y_array.select(Axis(0), &test_idx);

            let predictions = match &self.config.estimator {
                RFEEstimator::LinearRegression { config } => {
                    let model = Self::fit_linear_with_fallback(&x_train, &y_train, config)?;
                    model.predict(&x_val)?
                }
                RFEEstimator::RidgeCV { config } => {
                    let model = Self::fit_ridge(&x_train, &y_train, config)?;
                    model.predict(&x_val)?
                }
                RFEEstimator::LassoCV { config } => {
                    let model = Self::fit_lasso(&x_train, &y_train, config)?;
                    model.predict(&x_val)?
                }
            };

            let score =
                self.calculate_score(y_val.as_slice().unwrap(), predictions.as_slice().unwrap())?;
            scores.push(score);
        }

        if scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cross-validation produced no valid folds".to_string(),
            ));
        }

        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Calculate score based on scoring metric
    fn calculate_score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64, SklearsError> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "y_true.len() == y_pred.len()".to_string(),
                actual: format!(
                    "y_true.len() == {}, y_pred.len() == {}",
                    y_true.len(),
                    y_pred.len()
                ),
            });
        }

        match self.config.scoring {
            ScoringMetric::R2 => {
                let y_mean = y_true.iter().sum::<f64>() / y_true.len() as f64;
                let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
                let ss_res: f64 = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
                    .sum();

                if ss_tot < 1e-10 {
                    Ok(1.0)
                } else {
                    Ok(1.0 - ss_res / ss_tot)
                }
            }

            ScoringMetric::NegMeanSquaredError => {
                let mse: f64 = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mse)
            }

            ScoringMetric::NegMeanAbsoluteError => {
                let mae: f64 = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).abs())
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mae)
            }

            ScoringMetric::NegRootMeanSquaredError => {
                let mse: f64 = y_true
                    .iter()
                    .zip(y_pred.iter())
                    .map(|(&y_t, &y_p)| (y_t - y_p).powi(2))
                    .sum::<f64>()
                    / y_true.len() as f64;
                Ok(-mse.sqrt())
            }
        }
    }
}

impl Default for RecursiveFeatureElimination {
    fn default() -> Self {
        Self::new()
    }
}

impl RecursiveFeatureElimination {
    fn to_array2(x: &[Vec<f64>]) -> Result<Array2<f64>, SklearsError> {
        if x.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot convert empty feature matrix".to_string(),
            ));
        }

        let n_features = x[0].len();
        let mut flat = Vec::with_capacity(x.len() * n_features);

        for (row_idx, row) in x.iter().enumerate() {
            if row.len() != n_features {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("row[{row_idx}].len() == {n_features}"),
                    actual: format!("row[{row_idx}].len() == {}", row.len()),
                });
            }
            flat.extend_from_slice(row);
        }

        Array2::from_shape_vec((x.len(), n_features), flat)
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))
    }

    fn fit_linear_with_fallback(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &LinearRegressionConfig,
    ) -> Result<LinearRegression<Trained>, SklearsError> {
        let estimator = LinearRegression::new()
            .fit_intercept(config.fit_intercept)
            .penalty(config.penalty)
            .solver(config.solver)
            .max_iter(config.max_iter);

        match estimator.fit(x, y) {
            Ok(model) => Ok(model),
            Err(err) => match err {
                SklearsError::NumericalError(detail) if matches!(config.penalty, Penalty::None) => {
                    let fallback = LinearRegression::new()
                        .fit_intercept(config.fit_intercept)
                        .penalty(Penalty::L2(1e-6))
                        .solver(config.solver)
                        .max_iter(config.max_iter);

                    match fallback.fit(x, y) {
                        Ok(model) => Ok(model),
                        Err(ridge_err) => Err(SklearsError::NumericalError(format!(
                            "OLS failed ({detail}); ridge fallback failed ({ridge_err})"
                        ))),
                    }
                }
                _ => Err(err),
            },
        }
    }

    fn fit_ridge(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &RidgeCVConfig,
    ) -> Result<RidgeCV<Trained>, SklearsError> {
        let mut estimator = RidgeCV::new().fit_intercept(config.fit_intercept);

        if let Some(alphas) = &config.alphas {
            estimator = estimator.alphas(alphas.clone());
        }

        if let Some(cv) = config.cv {
            estimator = estimator.cv(cv);
        }

        if config.store_cv_values {
            estimator = estimator.store_cv_values(true);
        }

        estimator.fit(x, y)
    }

    fn fit_lasso(
        x: &Array2<f64>,
        y: &Array1<f64>,
        config: &LassoCVConfig,
    ) -> Result<LassoCV<Trained>, SklearsError> {
        let mut estimator = LassoCV::new()
            .fit_intercept(config.fit_intercept)
            .max_iter(config.max_iter)
            .tol(config.tol)
            .store_cv_values(config.store_cv_values);

        if let Some(alphas) = &config.alphas {
            estimator = estimator.alphas(alphas.clone());
        }

        if let Some(cv) = config.cv {
            estimator = estimator.cv(cv);
        }

        estimator.fit(x, y)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    fn create_sample_data() -> (Vec<Vec<f64>>, Vec<f64>) {
        // Create data where first 3 features are relevant, last 2 are noise
        let x = vec![
            vec![1.0, 2.0, 3.0, 0.1, 0.9],
            vec![2.0, 3.0, 4.0, 0.2, 0.8],
            vec![3.0, 4.0, 5.0, 0.1, 0.7],
            vec![4.0, 5.0, 6.0, 0.3, 0.6],
            vec![5.0, 6.0, 7.0, 0.2, 0.5],
            vec![6.0, 7.0, 8.0, 0.1, 0.4],
            vec![7.0, 8.0, 9.0, 0.2, 0.3],
            vec![8.0, 9.0, 10.0, 0.1, 0.2],
        ];
        // Target is combination of first 3 features
        let y = vec![6.0, 9.0, 12.0, 15.0, 18.0, 21.0, 24.0, 27.0];
        (x, y)
    }

    #[test]
    fn test_rfe_basic() {
        let mut rfe = RecursiveFeatureElimination::new()
            .with_n_features_to_select(Some(3))
            .with_step(1);

        let (x, y) = create_sample_data();
        let result = rfe.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select 3 features
        assert_eq!(transformed[0].len(), 3);
        assert_eq!(transformed.len(), 8);
    }

    #[test]
    fn test_rfe_with_ridge() {
        let mut rfe = RecursiveFeatureElimination::new()
            .with_estimator(RFEEstimator::RidgeCV {
                config: RidgeCVConfig::default(),
            })
            .with_n_features_to_select(Some(2))
            .with_step(2);

        let (x, y) = create_sample_data();
        let result = rfe.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select 2 features
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_rfe_with_lasso() {
        // Use minimal parameters to avoid timeout
        // RFE: 2 iterations (5→4→3 features)
        // LassoCV: 5 alphas × 2 folds = 10 fits per iteration
        // Total: 2 × 10 = 20 coordinate descent runs
        let lasso_config = LassoCVConfig {
            max_iter: 500,
            tol: 1e-2,
            fit_intercept: true,
            alphas: Some(vec![0.001, 0.01, 0.1, 1.0, 10.0]), // Explicit alphas
            cv: Some(2), // Must be Some, not None (None defaults to 5 folds)
            n_alphas: 5,
            store_cv_values: false,
        };

        let mut rfe = RecursiveFeatureElimination::new()
            .with_estimator(RFEEstimator::LassoCV {
                config: lasso_config,
            })
            .with_n_features_to_select(Some(3))
            .with_cv(Some(2)); // Reduced CV folds for faster execution

        let (x, y) = create_sample_data();
        let result = rfe.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select 3 features
        assert_eq!(transformed[0].len(), 3);
    }

    #[test]
    fn test_rfe_feature_ranking() {
        let mut rfe = RecursiveFeatureElimination::new().with_n_features_to_select(Some(3));

        let (x, y) = create_sample_data();
        rfe.fit(&x, &y).unwrap();

        let ranking = rfe.get_feature_ranking().unwrap();
        assert_eq!(ranking.len(), 5); // 5 original features

        // Selected features should have rank 1
        let selected_features = rfe.get_selected_features().unwrap();
        for &feature_idx in selected_features {
            assert_eq!(ranking[feature_idx], 1);
        }
    }

    #[test]
    fn test_rfe_cv_scores() {
        let mut rfe = RecursiveFeatureElimination::new()
            .with_n_features_to_select(Some(2))
            .with_cv(Some(3));

        let (x, y) = create_sample_data();
        rfe.fit(&x, &y).unwrap();

        let cv_scores = rfe.get_cv_scores().unwrap();
        assert!(!cv_scores.is_empty());

        // Scores should be finite
        for &score in cv_scores {
            assert!(score.is_finite());
        }
    }

    #[test]
    fn test_rfe_different_scoring_metrics() {
        let scoring_metrics = vec![
            ScoringMetric::R2,
            ScoringMetric::NegMeanSquaredError,
            ScoringMetric::NegMeanAbsoluteError,
            ScoringMetric::NegRootMeanSquaredError,
        ];

        let (x, y) = create_sample_data();

        for scoring in scoring_metrics {
            let mut rfe = RecursiveFeatureElimination::new()
                .with_n_features_to_select(Some(3))
                .with_scoring(scoring.clone())
                .with_cv(Some(3));

            let result = rfe.fit_transform(&x, &y);
            assert!(result.is_ok(), "Failed with scoring metric: {:?}", scoring);

            let transformed = result.unwrap();
            assert_eq!(transformed[0].len(), 3);
        }
    }

    #[test]
    fn test_rfe_step_size() {
        let mut rfe = RecursiveFeatureElimination::new()
            .with_n_features_to_select(Some(2))
            .with_step(2); // Eliminate 2 features at a time

        let (x, y) = create_sample_data();
        let result = rfe.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should still select 2 features
        assert_eq!(transformed[0].len(), 2);
    }

    #[test]
    fn test_rfe_empty_data_error() {
        let mut rfe = RecursiveFeatureElimination::new();
        let x: Vec<Vec<f64>> = vec![];
        let y: Vec<f64> = vec![];

        let result = rfe.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfe_dimension_mismatch_error() {
        let mut rfe = RecursiveFeatureElimination::new();
        let (x, _) = create_sample_data();
        let wrong_y = vec![1.0, 2.0]; // Wrong length

        let result = rfe.fit(&x, &wrong_y);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfe_transform_before_fit_error() {
        let rfe = RecursiveFeatureElimination::new();
        let (x, _) = create_sample_data();

        let result = rfe.transform(&x);
        assert!(result.is_err());
    }

    #[test]
    fn test_rfe_auto_feature_selection() {
        let mut rfe = RecursiveFeatureElimination::new().with_n_features_to_select(None); // Auto-select half

        let (x, y) = create_sample_data();
        let result = rfe.fit_transform(&x, &y);

        assert!(result.is_ok());
        let transformed = result.unwrap();

        // Should select half of the features (5/2 = 2)
        assert_eq!(transformed[0].len(), 2);
    }
}
