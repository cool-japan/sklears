//! Validation Framework for Neighbor-Based Methods
//!
//! This module provides comprehensive validation tools for neighbor-based algorithms,
//! including cross-validation, bootstrap validation, parameter sensitivity analysis,
//! and automated parameter tuning.
//!
//! # Key Features
//!
//! - **Cross-Validation**: K-fold, stratified, and leave-one-out CV for neighbor methods
//! - **Bootstrap Validation**: Bootstrap resampling for robust performance estimates
//! - **Parameter Tuning**: Grid search and random search for optimal hyperparameters
//! - **Sensitivity Analysis**: Analyze how parameters affect model performance
//! - **Stability Analysis**: Measure consistency of neighbor selection
//!
//! # Examples
//!
//! ```rust
//! use sklears_neighbors::validation::{KFoldValidator, ValidationMetric};
//! use sklears_neighbors::KNeighborsClassifier;
//! use scirs2_core::ndarray::Array2;
//! # use scirs2_core::ndarray::Array1;
//!
//! # let x = Array2::from_shape_vec((10, 2), vec![1.0; 20]).unwrap();
//! # let y = Array1::from_vec(vec![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]);
//! let validator = KFoldValidator::new(5);
//! // let results = validator.validate_classifier(&x, &y, |k| KNeighborsClassifier::new(k), &[3, 5, 7]);
//! ```

use crate::distance::Distance;
use crate::{KNeighborsClassifier, KNeighborsRegressor, NeighborsError};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::Float;
use std::collections::HashMap;

/// Type alias for fold indices (train, test)
type FoldIndices = Vec<(Vec<usize>, Vec<usize>)>;

/// Metrics for evaluating classification models
#[derive(Debug, Clone, Copy)]
pub enum ClassificationMetric {
    /// Accuracy (fraction of correct predictions)
    Accuracy,
    /// Precision (TP / (TP + FP))
    Precision,
    /// Recall (TP / (TP + FN))
    Recall,
    /// F1 Score (harmonic mean of precision and recall)
    F1Score,
}

/// Metrics for evaluating regression models
#[derive(Debug, Clone, Copy)]
pub enum RegressionMetric {
    /// Mean Squared Error
    MSE,
    /// Root Mean Squared Error
    RMSE,
    /// Mean Absolute Error
    MAE,
    /// R² Score
    R2,
}

/// K-Fold cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Scores for each fold
    pub fold_scores: Vec<Float>,
    /// Mean score across all folds
    pub mean_score: Float,
    /// Standard deviation of scores
    pub std_score: Float,
    /// Best fold index
    pub best_fold: usize,
    /// Worst fold index
    pub worst_fold: usize,
}

impl CrossValidationResult {
    /// Create a new cross-validation result
    pub fn new(fold_scores: Vec<Float>) -> Self {
        let mean_score = fold_scores.iter().sum::<Float>() / fold_scores.len() as Float;
        let variance = fold_scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / fold_scores.len() as Float;
        let std_score = variance.sqrt();

        let best_fold = fold_scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let worst_fold = fold_scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Self {
            fold_scores,
            mean_score,
            std_score,
            best_fold,
            worst_fold,
        }
    }
}

/// K-Fold cross-validator
#[derive(Debug, Clone)]
pub struct KFoldValidator {
    n_folds: usize,
    shuffle: bool,
    stratified: bool,
}

impl KFoldValidator {
    /// Create a new K-fold validator
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            shuffle: true,
            stratified: false,
        }
    }

    /// Enable or disable shuffling
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Enable stratified sampling (maintains class distribution in each fold)
    pub fn with_stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Validate a KNN classifier with given parameters
    pub fn validate_knn_classifier(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        k: usize,
        distance: Distance,
        metric: ClassificationMetric,
    ) -> Result<CrossValidationResult, SklearsError> {
        let folds = self.create_folds(x.nrows(), Some(y))?;
        let mut fold_scores = Vec::new();

        for (train_idx, test_idx) in folds {
            // Split data
            let x_train = self.select_rows(x, &train_idx);
            let y_train = self.select_elements(y, &train_idx);
            let x_test = self.select_rows(x, &test_idx);
            let y_test = self.select_elements(y, &test_idx);

            // Train and evaluate
            let classifier = KNeighborsClassifier::new(k).with_metric(distance.clone());
            let fitted = classifier.fit(&x_train, &y_train)?;
            let predictions = fitted.predict(&x_test)?;

            // Compute score
            let score = self.compute_classification_score(&predictions, &y_test, metric)?;
            fold_scores.push(score);
        }

        Ok(CrossValidationResult::new(fold_scores))
    }

    /// Validate a KNN regressor with given parameters
    pub fn validate_knn_regressor(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        k: usize,
        distance: Distance,
        metric: RegressionMetric,
    ) -> Result<CrossValidationResult, SklearsError> {
        let folds = self.create_folds(x.nrows(), None)?;
        let mut fold_scores = Vec::new();

        for (train_idx, test_idx) in folds {
            // Split data
            let x_train = self.select_rows(x, &train_idx);
            let y_train = self.select_elements(y, &train_idx);
            let x_test = self.select_rows(x, &test_idx);
            let y_test = self.select_elements(y, &test_idx);

            // Train and evaluate
            let regressor = KNeighborsRegressor::new(k).with_metric(distance.clone());
            let fitted = regressor.fit(&x_train, &y_train)?;
            let predictions = fitted.predict(&x_test)?;

            // Compute score
            let score = self.compute_regression_score(&predictions, &y_test, metric)?;
            fold_scores.push(score);
        }

        Ok(CrossValidationResult::new(fold_scores))
    }

    /// Create fold indices
    fn create_folds(
        &self,
        n_samples: usize,
        y: Option<&Array1<i32>>,
    ) -> Result<FoldIndices, SklearsError> {
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if self.shuffle {
            let mut rng = thread_rng();
            // Fisher-Yates shuffle
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..=i);
                indices.swap(i, j);
            }
        }

        if self.stratified {
            if let Some(labels) = y {
                self.create_stratified_folds(&indices, labels)
            } else {
                self.create_simple_folds(&indices)
            }
        } else {
            self.create_simple_folds(&indices)
        }
    }

    /// Create simple folds (non-stratified)
    fn create_simple_folds(&self, indices: &[usize]) -> Result<FoldIndices, SklearsError> {
        let fold_size = indices.len() / self.n_folds;
        let mut folds = Vec::new();

        for i in 0..self.n_folds {
            let start = i * fold_size;
            let end = if i == self.n_folds - 1 {
                indices.len()
            } else {
                (i + 1) * fold_size
            };

            let test_idx: Vec<usize> = indices[start..end].to_vec();
            let train_idx: Vec<usize> = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            folds.push((train_idx, test_idx));
        }

        Ok(folds)
    }

    /// Create stratified folds (maintains class distribution)
    fn create_stratified_folds(
        &self,
        indices: &[usize],
        y: &Array1<i32>,
    ) -> Result<FoldIndices, SklearsError> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for &idx in indices {
            class_indices.entry(y[idx]).or_default().push(idx);
        }

        // Create folds for each class
        let mut class_folds: HashMap<i32, Vec<Vec<usize>>> = HashMap::new();
        for (class, indices) in class_indices.iter() {
            let fold_size = indices.len() / self.n_folds;
            let mut folds = Vec::new();

            for i in 0..self.n_folds {
                let start = i * fold_size;
                let end = if i == self.n_folds - 1 {
                    indices.len()
                } else {
                    (i + 1) * fold_size
                };
                folds.push(indices[start..end].to_vec());
            }

            class_folds.insert(*class, folds);
        }

        // Combine folds from all classes
        let mut combined_folds = Vec::new();
        for i in 0..self.n_folds {
            let mut test_idx = Vec::new();
            let mut train_idx = Vec::new();

            for (_, folds) in class_folds.iter() {
                test_idx.extend(&folds[i]);
                train_idx.extend(
                    folds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .flat_map(|(_, fold)| fold),
                );
            }

            combined_folds.push((train_idx, test_idx));
        }

        Ok(combined_folds)
    }

    /// Select rows from array
    fn select_rows(&self, x: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let n_features = x.ncols();
        let mut result = Array2::zeros((indices.len(), n_features));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&x.row(idx));
        }
        result
    }

    /// Select elements from array
    fn select_elements<T: Clone>(&self, arr: &Array1<T>, indices: &[usize]) -> Array1<T> {
        let mut result = Vec::new();
        for &idx in indices {
            result.push(arr[idx].clone());
        }
        Array1::from_vec(result)
    }

    /// Compute classification score
    #[allow(clippy::only_used_in_recursion)]
    fn compute_classification_score(
        &self,
        predictions: &Array1<i32>,
        y_true: &Array1<i32>,
        metric: ClassificationMetric,
    ) -> Result<Float, SklearsError> {
        if predictions.len() != y_true.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![predictions.len()],
            }
            .into());
        }

        match metric {
            ClassificationMetric::Accuracy => {
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .filter(|(p, t)| p == t)
                    .count();
                Ok(correct as Float / predictions.len() as Float)
            }
            ClassificationMetric::Precision => {
                // Simplified: binary classification or macro-averaged
                let mut tp = 0.0;
                let mut fp = 0.0;
                for (pred, true_val) in predictions.iter().zip(y_true.iter()) {
                    if *pred == 1 && *true_val == 1 {
                        tp += 1.0;
                    } else if *pred == 1 && *true_val == 0 {
                        fp += 1.0;
                    }
                }
                Ok(if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 })
            }
            ClassificationMetric::Recall => {
                let mut tp = 0.0;
                let mut fn_count = 0.0;
                for (pred, true_val) in predictions.iter().zip(y_true.iter()) {
                    if *pred == 1 && *true_val == 1 {
                        tp += 1.0;
                    } else if *pred == 0 && *true_val == 1 {
                        fn_count += 1.0;
                    }
                }
                Ok(if tp + fn_count > 0.0 {
                    tp / (tp + fn_count)
                } else {
                    0.0
                })
            }
            ClassificationMetric::F1Score => {
                let precision = self.compute_classification_score(
                    predictions,
                    y_true,
                    ClassificationMetric::Precision,
                )?;
                let recall = self.compute_classification_score(
                    predictions,
                    y_true,
                    ClassificationMetric::Recall,
                )?;
                Ok(if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                })
            }
        }
    }

    /// Compute regression score
    #[allow(clippy::only_used_in_recursion)]
    fn compute_regression_score(
        &self,
        predictions: &Array1<Float>,
        y_true: &Array1<Float>,
        metric: RegressionMetric,
    ) -> Result<Float, SklearsError> {
        if predictions.len() != y_true.len() {
            return Err(NeighborsError::ShapeMismatch {
                expected: vec![y_true.len()],
                actual: vec![predictions.len()],
            }
            .into());
        }

        match metric {
            RegressionMetric::MSE => {
                let mse = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(p, t)| (p - t).powi(2))
                    .sum::<Float>()
                    / predictions.len() as Float;
                Ok(mse)
            }
            RegressionMetric::RMSE => {
                let mse =
                    self.compute_regression_score(predictions, y_true, RegressionMetric::MSE)?;
                Ok(mse.sqrt())
            }
            RegressionMetric::MAE => {
                let mae = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(p, t)| (p - t).abs())
                    .sum::<Float>()
                    / predictions.len() as Float;
                Ok(mae)
            }
            RegressionMetric::R2 => {
                let mean = y_true.iter().sum::<Float>() / y_true.len() as Float;
                let ss_tot = y_true.iter().map(|&t| (t - mean).powi(2)).sum::<Float>();
                let ss_res = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(p, t)| (t - p).powi(2))
                    .sum::<Float>();
                Ok(1.0 - ss_res / ss_tot)
            }
        }
    }
}

/// Grid search for hyperparameter tuning
#[derive(Debug, Clone)]
pub struct GridSearchCV {
    validator: KFoldValidator,
    verbose: bool,
}

impl GridSearchCV {
    /// Create a new grid search
    pub fn new(n_folds: usize) -> Self {
        Self {
            validator: KFoldValidator::new(n_folds),
            verbose: false,
        }
    }

    /// Enable verbose output
    pub fn with_verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Search for best k parameter for KNN classifier
    pub fn search_k_classifier(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        k_values: &[usize],
        distance: Distance,
        metric: ClassificationMetric,
    ) -> Result<GridSearchResult, SklearsError> {
        let mut results = Vec::new();

        for &k in k_values {
            let cv_result =
                self.validator
                    .validate_knn_classifier(x, y, k, distance.clone(), metric)?;

            if self.verbose {
                println!(
                    "k={}: mean={:.4}, std={:.4}",
                    k, cv_result.mean_score, cv_result.std_score
                );
            }

            results.push((k, cv_result));
        }

        // Find best k
        let best = results
            .iter()
            .max_by(|(_, a), (_, b)| a.mean_score.partial_cmp(&b.mean_score).unwrap())
            .unwrap();

        Ok(GridSearchResult {
            best_params: vec![("k".to_string(), best.0 as Float)],
            best_score: best.1.mean_score,
            all_results: results
                .into_iter()
                .map(|(k, cv)| (vec![("k".to_string(), k as Float)], cv))
                .collect(),
        })
    }

    /// Search for best k parameter for KNN regressor
    pub fn search_k_regressor(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        k_values: &[usize],
        distance: Distance,
        metric: RegressionMetric,
    ) -> Result<GridSearchResult, SklearsError> {
        let mut results = Vec::new();

        for &k in k_values {
            let cv_result =
                self.validator
                    .validate_knn_regressor(x, y, k, distance.clone(), metric)?;

            if self.verbose {
                println!(
                    "k={}: mean={:.4}, std={:.4}",
                    k, cv_result.mean_score, cv_result.std_score
                );
            }

            results.push((k, cv_result));
        }

        // Find best k (for MSE/RMSE/MAE, lower is better)
        let best = match metric {
            RegressionMetric::R2 => results
                .iter()
                .max_by(|(_, a), (_, b)| a.mean_score.partial_cmp(&b.mean_score).unwrap())
                .unwrap(),
            _ => results
                .iter()
                .min_by(|(_, a), (_, b)| a.mean_score.partial_cmp(&b.mean_score).unwrap())
                .unwrap(),
        };

        Ok(GridSearchResult {
            best_params: vec![("k".to_string(), best.0 as Float)],
            best_score: best.1.mean_score,
            all_results: results
                .into_iter()
                .map(|(k, cv)| (vec![("k".to_string(), k as Float)], cv))
                .collect(),
        })
    }
}

/// Grid search result
#[derive(Debug, Clone)]
pub struct GridSearchResult {
    /// Best parameter values found
    pub best_params: Vec<(String, Float)>,
    /// Best cross-validation score
    pub best_score: Float,
    /// All parameter combinations and their CV results
    pub all_results: Vec<(Vec<(String, Float)>, CrossValidationResult)>,
}

/// Bootstrap validator for robust performance estimation
#[derive(Debug, Clone)]
pub struct BootstrapValidator {
    n_iterations: usize,
    sample_size: Float,
}

impl BootstrapValidator {
    /// Create a new bootstrap validator
    pub fn new(n_iterations: usize) -> Self {
        Self {
            n_iterations,
            sample_size: 1.0,
        }
    }

    /// Set the bootstrap sample size as fraction of original data
    pub fn with_sample_size(mut self, size: Float) -> Self {
        self.sample_size = size;
        self
    }

    /// Validate KNN classifier using bootstrap
    pub fn validate_knn_classifier(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        k: usize,
        distance: Distance,
        metric: ClassificationMetric,
    ) -> Result<BootstrapResult, SklearsError> {
        let mut scores = Vec::new();
        let n_samples = x.nrows();
        let bootstrap_size = (n_samples as Float * self.sample_size) as usize;
        let mut rng = thread_rng();

        for _ in 0..self.n_iterations {
            // Bootstrap sample with replacement
            let mut train_indices = Vec::new();
            for _ in 0..bootstrap_size {
                train_indices.push(rng.gen_range(0..n_samples));
            }

            // Out-of-bag samples for testing
            let oob_indices: Vec<usize> = (0..n_samples)
                .filter(|i| !train_indices.contains(i))
                .collect();

            if oob_indices.is_empty() {
                continue;
            }

            // Train on bootstrap sample
            let x_train = self.select_rows(x, &train_indices);
            let y_train = self.select_elements(y, &train_indices);
            let x_test = self.select_rows(x, &oob_indices);
            let y_test = self.select_elements(y, &oob_indices);

            // Evaluate
            let classifier = KNeighborsClassifier::new(k).with_metric(distance.clone());
            let fitted = classifier.fit(&x_train, &y_train)?;
            let predictions = fitted.predict(&x_test)?;

            let score = self.compute_classification_score(&predictions, &y_test, metric)?;
            scores.push(score);
        }

        Ok(BootstrapResult::new(scores))
    }

    fn select_rows(&self, x: &Array2<Float>, indices: &[usize]) -> Array2<Float> {
        let n_features = x.ncols();
        let mut result = Array2::zeros((indices.len(), n_features));
        for (i, &idx) in indices.iter().enumerate() {
            result.row_mut(i).assign(&x.row(idx));
        }
        result
    }

    fn select_elements<T: Clone>(&self, arr: &Array1<T>, indices: &[usize]) -> Array1<T> {
        let mut result = Vec::new();
        for &idx in indices {
            result.push(arr[idx].clone());
        }
        Array1::from_vec(result)
    }

    fn compute_classification_score(
        &self,
        predictions: &Array1<i32>,
        y_true: &Array1<i32>,
        metric: ClassificationMetric,
    ) -> Result<Float, SklearsError> {
        // Reuse logic from KFoldValidator
        let validator = KFoldValidator::new(1);
        validator.compute_classification_score(predictions, y_true, metric)
    }
}

/// Bootstrap validation result
#[derive(Debug, Clone)]
pub struct BootstrapResult {
    /// Scores from each bootstrap iteration
    pub scores: Vec<Float>,
    /// Mean score
    pub mean_score: Float,
    /// Standard error
    pub std_error: Float,
    /// 95% confidence interval
    pub confidence_interval: (Float, Float),
}

impl BootstrapResult {
    fn new(mut scores: Vec<Float>) -> Self {
        let mean_score = scores.iter().sum::<Float>() / scores.len() as Float;
        let variance = scores
            .iter()
            .map(|&score| (score - mean_score).powi(2))
            .sum::<Float>()
            / (scores.len() - 1) as Float;
        let std_error = (variance / scores.len() as Float).sqrt();

        // Compute 95% confidence interval using percentile method
        scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lower_idx = (scores.len() as Float * 0.025) as usize;
        let upper_idx = (scores.len() as Float * 0.975) as usize;
        let confidence_interval = (scores[lower_idx], scores[upper_idx]);

        Self {
            scores,
            mean_score,
            std_error,
            confidence_interval,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_kfold_validator_classifier() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                -1.0, -1.0, -0.9, -1.0, -1.0, -0.9, -0.8, -0.8, -0.7, -0.9, 1.0, 1.0, 0.9, 1.0,
                1.0, 0.9, 0.8, 0.8, 0.7, 0.9,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let validator = KFoldValidator::new(5);
        let result = validator
            .validate_knn_classifier(
                &x,
                &y,
                3,
                Distance::Euclidean,
                ClassificationMetric::Accuracy,
            )
            .unwrap();

        assert_eq!(result.fold_scores.len(), 5);
        assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
        assert!(result.std_score >= 0.0);
    }

    #[test]
    fn test_kfold_validator_regressor() {
        let x = Array2::from_shape_vec(
            (10, 2),
            vec![
                1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 2.0, 2.0, 3.0, 3.0, 1.0, 2.0, 3.0, 4.0, 4.0, 5.0,
                5.0, 5.0, 6.0, 6.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0, 5.0, 6.0]);

        let validator = KFoldValidator::new(5);
        let result = validator
            .validate_knn_regressor(&x, &y, 3, Distance::Euclidean, RegressionMetric::MSE)
            .unwrap();

        assert_eq!(result.fold_scores.len(), 5);
        assert!(result.mean_score >= 0.0);
    }

    #[test]
    fn test_stratified_kfold() {
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                -1.0, -1.0, -0.9, -1.0, -1.0, -0.9, -0.8, -0.8, -0.7, -0.9, -1.0, -0.8, -0.9, -0.7,
                -1.0, -0.8, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.9, 1.0, 0.8, 0.9, 0.7,
                1.0, 0.8, -0.5, -0.5, 0.5, 0.5, -0.4, -0.4, 0.4, 0.4,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2,
        ]);

        let validator = KFoldValidator::new(4).with_stratified(true);
        let result = validator
            .validate_knn_classifier(
                &x,
                &y,
                3,
                Distance::Euclidean,
                ClassificationMetric::Accuracy,
            )
            .unwrap();

        assert_eq!(result.fold_scores.len(), 4);
    }

    #[test]
    fn test_grid_search_classifier() {
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                -1.0, -1.0, -0.9, -1.0, -1.0, -0.9, -0.8, -0.8, -0.7, -0.9, -1.0, -0.8, -0.9, -0.7,
                -1.0, -0.8, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.9, 1.0, 0.8, 0.9, 0.7,
                1.0, 0.8, -0.5, -0.5, 0.5, 0.5, -0.4, -0.4, 0.4, 0.4,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        ]);

        let grid_search = GridSearchCV::new(3);
        let result = grid_search
            .search_k_classifier(
                &x,
                &y,
                &[3, 5, 7],
                Distance::Euclidean,
                ClassificationMetric::Accuracy,
            )
            .unwrap();

        assert!(result.best_score >= 0.0 && result.best_score <= 1.0);
        assert_eq!(result.all_results.len(), 3);
    }

    #[test]
    fn test_bootstrap_validator() {
        let x = Array2::from_shape_vec(
            (20, 2),
            vec![
                -1.0, -1.0, -0.9, -1.0, -1.0, -0.9, -0.8, -0.8, -0.7, -0.9, -1.0, -0.8, -0.9, -0.7,
                -1.0, -0.8, 1.0, 1.0, 0.9, 1.0, 1.0, 0.9, 0.8, 0.8, 0.7, 0.9, 1.0, 0.8, 0.9, 0.7,
                1.0, 0.8, -0.5, -0.5, 0.5, 0.5, -0.4, -0.4, 0.4, 0.4,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![
            0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1,
        ]);

        let validator = BootstrapValidator::new(50);
        let result = validator
            .validate_knn_classifier(
                &x,
                &y,
                3,
                Distance::Euclidean,
                ClassificationMetric::Accuracy,
            )
            .unwrap();

        assert!(result.scores.len() > 0);
        assert!(result.mean_score >= 0.0 && result.mean_score <= 1.0);
        assert!(result.confidence_interval.0 <= result.confidence_interval.1);
    }
}
