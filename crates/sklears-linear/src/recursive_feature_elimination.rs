//! Recursive Feature Elimination (RFE) for linear models
//!
//! This module provides recursive feature elimination, a wrapper feature selection
//! method that fits a model and recursively removes features based on their importance,
//! retraining the model with remaining features until the desired number is reached.

use crate::lasso_cv::{LassoCV, LassoCVConfig};
use crate::linear_regression::{LinearRegression, LinearRegressionConfig};
use crate::ridge_cv::{RidgeCV, RidgeCVConfig};
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::SklearsError,
    traits::{Estimator, Fit, Predict},
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
                let estimator = LinearRegression::new()
                    .fit_intercept(config.fit_intercept)
                    .penalty(config.penalty.clone())
                    .solver(config.solver.clone())
                    .max_iter(config.max_iter);
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

                // For linear regression, use absolute coefficients as importance
                let coefficients = trained.coef();

                Ok(coefficients.iter().map(|&coef| coef.abs()).collect())
            }

            RFEEstimator::RidgeCV { config } => {
                let estimator = RidgeCV::new();
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

                // For ridge regression, use absolute coefficients as importance
                let coefficients = trained.coef();

                Ok(coefficients.iter().map(|&coef| coef.abs()).collect())
            }

            RFEEstimator::LassoCV { config } => {
                let estimator = LassoCV::new();
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

                // For lasso regression, use absolute coefficients as importance
                let coefficients = trained.coef();

                Ok(coefficients.iter().map(|&coef| coef.abs()).collect())
            }
        }
    }

    /// Fit final estimator and return coefficients
    fn fit_final_estimator(&self, x: &[Vec<f64>], y: &[f64]) -> Result<Vec<f64>, SklearsError> {
        match &self.config.estimator {
            RFEEstimator::LinearRegression { config } => {
                let estimator = LinearRegression::new()
                    .fit_intercept(config.fit_intercept)
                    .penalty(config.penalty.clone())
                    .solver(config.solver.clone())
                    .max_iter(config.max_iter);
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

                Ok(trained.coef().to_vec())
            }

            RFEEstimator::RidgeCV { config } => {
                let estimator = RidgeCV::new();
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

                Ok(trained.coef().to_vec())
            }

            RFEEstimator::LassoCV { config } => {
                let estimator = LassoCV::new();
                // Convert Vec<Vec<f64>> to Array2<f64>
                let x_array = Array2::from_shape_vec(
                    (x.len(), x[0].len()),
                    x.iter().flatten().cloned().collect(),
                )
                .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

                // Convert Vec<f64> to Array1<f64>
                let y_array = Array1::from_vec(y.to_vec());

                let trained = estimator.fit(&x_array, &y_array)?;

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
        let n_samples = x.len();
        let fold_size = n_samples / cv_folds;
        let mut scores = Vec::new();

        for fold in 0..cv_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == cv_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };

            // Create train/validation splits
            let mut x_train = Vec::new();
            let mut y_train = Vec::new();
            let mut x_val = Vec::new();
            let mut y_val = Vec::new();

            for (i, (x_row, &y_val_single)) in x.iter().zip(y.iter()).enumerate() {
                if i >= start_idx && i < end_idx {
                    x_val.push(x_row.clone());
                    y_val.push(y_val_single);
                } else {
                    x_train.push(x_row.clone());
                    y_train.push(y_val_single);
                }
            }

            // Convert training data
            let x_train_array = Array2::from_shape_vec(
                (x_train.len(), x_train[0].len()),
                x_train.iter().flatten().cloned().collect(),
            )
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;
            let y_train_array = Array1::from_vec(y_train.to_vec());

            // Convert validation data
            let x_val_array = Array2::from_shape_vec(
                (x_val.len(), x_val[0].len()),
                x_val.iter().flatten().cloned().collect(),
            )
            .map_err(|e| SklearsError::InvalidInput(format!("Shape error: {}", e)))?;

            // Fit and predict for each estimator type
            let y_pred = match &self.config.estimator {
                RFEEstimator::LinearRegression { config } => {
                    let estimator = LinearRegression::new()
                        .fit_intercept(config.fit_intercept)
                        .penalty(config.penalty.clone())
                        .solver(config.solver.clone())
                        .max_iter(config.max_iter);
                    let trained_model = estimator.fit(&x_train_array, &y_train_array)?;
                    trained_model.predict(&x_val_array)?
                }

                RFEEstimator::RidgeCV { config } => {
                    let estimator = RidgeCV::new();
                    let trained_model = estimator.fit(&x_train_array, &y_train_array)?;
                    trained_model.predict(&x_val_array)?
                }

                RFEEstimator::LassoCV { config } => {
                    let estimator = LassoCV::new();
                    let trained_model = estimator.fit(&x_train_array, &y_train_array)?;
                    trained_model.predict(&x_val_array)?
                }
            };

            // Calculate score
            let score = self.calculate_score(&y_val, y_pred.as_slice().unwrap())?;
            scores.push(score);
        }

        // Return average score
        Ok(scores.iter().sum::<f64>() / scores.len() as f64)
    }

    /// Calculate score based on scoring metric
    fn calculate_score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64, SklearsError> {
        if y_true.len() != y_pred.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("y_true.len() == y_pred.len()"),
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
        let mut rfe = RecursiveFeatureElimination::new()
            .with_estimator(RFEEstimator::LassoCV {
                config: LassoCVConfig::default(),
            })
            .with_n_features_to_select(Some(3))
            .with_cv(Some(3));

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
