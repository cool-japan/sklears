use scirs2_core::ndarray::{Array1, Array2, Array3, Axis, ScalarOperand};
use scirs2_core::numeric::{Float, One, ToPrimitive};
use scirs2_core::random::{Rng, SeedableRng};
use std::fmt::Debug;

use crate::activation::Activation;
use crate::mlp_classifier::MLPClassifier;
use crate::mlp_regressor::MLPRegressor;
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::FloatBounds;
use std::collections::HashMap;

/// Model comparison and selection utilities for neural networks
///
/// This module provides tools for comparing different neural network architectures,
/// hyperparameter optimization, cross-validation, and model selection.

/// Model performance metrics
#[derive(Debug, Clone)]
pub struct ModelMetrics<T: Float> {
    /// Model identifier/name
    pub model_name: String,
    /// Training accuracy/error
    pub train_score: T,
    /// Validation accuracy/error
    pub validation_score: T,
    /// Test accuracy/error (if available)
    pub test_score: Option<T>,
    /// Training time in seconds
    pub training_time: T,
    /// Number of parameters
    pub num_parameters: usize,
    /// Model complexity score
    pub complexity_score: T,
    /// Additional metrics
    pub additional_metrics: HashMap<String, T>,
}

impl<T: Float> ModelMetrics<T> {
    /// Create new model metrics
    pub fn new(model_name: String) -> Self {
        Self {
            model_name,
            train_score: T::zero(),
            validation_score: T::zero(),
            test_score: None,
            training_time: T::zero(),
            num_parameters: 0,
            complexity_score: T::zero(),
            additional_metrics: HashMap::new(),
        }
    }

    /// Add an additional metric
    pub fn add_metric(&mut self, name: String, value: T) {
        self.additional_metrics.insert(name, value);
    }

    /// Get metric by name
    pub fn get_metric(&self, name: &str) -> Option<T> {
        match name {
            "train_score" => Some(self.train_score),
            "validation_score" => Some(self.validation_score),
            "test_score" => self.test_score,
            "training_time" => Some(self.training_time),
            "complexity_score" => Some(self.complexity_score),
            _ => self.additional_metrics.get(name).copied(),
        }
    }
}

/// Cross-validation configuration
#[derive(Debug, Clone)]
pub struct CrossValidationConfig {
    /// Number of folds
    pub n_folds: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Whether to shuffle data before folding
    pub shuffle: bool,
    /// Stratify for classification (maintain class distribution)
    pub stratify: bool,
}

impl Default for CrossValidationConfig {
    fn default() -> Self {
        Self {
            n_folds: 5,
            random_seed: Some(42),
            shuffle: true,
            stratify: true,
        }
    }
}

/// Cross-validation results
#[derive(Debug, Clone)]
pub struct CrossValidationResults<T: Float> {
    /// Scores for each fold
    pub fold_scores: Vec<T>,
    /// Mean score across folds
    pub mean_score: T,
    /// Standard deviation of scores
    pub std_score: T,
    /// Best fold index
    pub best_fold: usize,
    /// Worst fold index
    pub worst_fold: usize,
}

impl<T: Float + std::iter::Sum> CrossValidationResults<T> {
    /// Create from fold scores
    pub fn from_scores(scores: Vec<T>) -> Self {
        let mean_score = scores.iter().cloned().sum::<T>() / T::from(scores.len()).unwrap();

        let variance = scores
            .iter()
            .map(|&score| {
                let diff = score - mean_score;
                diff * diff
            })
            .sum::<T>()
            / T::from(scores.len()).unwrap();

        let std_score = variance.sqrt();

        let best_fold = scores
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let worst_fold = scores
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        Self {
            fold_scores: scores,
            mean_score,
            std_score,
            best_fold,
            worst_fold,
        }
    }
}

/// Grid search configuration for hyperparameter optimization
#[derive(Debug, Clone)]
pub struct GridSearchConfig<T: Float> {
    /// Hidden layer sizes to try
    pub hidden_layer_sizes: Vec<Vec<usize>>,
    /// Learning rates to try
    pub learning_rates: Vec<T>,
    /// Regularization strengths to try
    pub alphas: Vec<T>,
    /// Activation functions to try
    pub activations: Vec<Activation>,
    /// Maximum iterations to try
    pub max_iters: Vec<usize>,
    /// Cross-validation configuration
    pub cv_config: CrossValidationConfig,
    /// Scoring metric
    pub scoring: String,
}

impl<T: Float> Default for GridSearchConfig<T> {
    fn default() -> Self {
        Self {
            hidden_layer_sizes: vec![vec![100], vec![50, 50], vec![100, 50], vec![100, 100]],
            learning_rates: vec![
                T::from(0.001).unwrap(),
                T::from(0.01).unwrap(),
                T::from(0.1).unwrap(),
            ],
            alphas: vec![
                T::from(0.0001).unwrap(),
                T::from(0.001).unwrap(),
                T::from(0.01).unwrap(),
            ],
            activations: vec![Activation::Relu, Activation::Tanh, Activation::Logistic],
            max_iters: vec![200, 500, 1000],
            cv_config: CrossValidationConfig::default(),
            scoring: "accuracy".to_string(),
        }
    }
}

/// Hyperparameter combination
#[derive(Debug, Clone)]
pub struct HyperparameterSet<T: Float> {
    pub hidden_layer_sizes: Vec<usize>,
    pub learning_rate: T,
    pub alpha: T,
    pub activation: Activation,
    pub max_iter: usize,
}

/// Grid search results
#[derive(Debug, Clone)]
pub struct GridSearchResults<T: Float> {
    /// Best hyperparameters found
    pub best_params: HyperparameterSet<T>,
    /// Best cross-validation score
    pub best_score: T,
    /// All parameter combinations tried
    pub all_results: Vec<(HyperparameterSet<T>, CrossValidationResults<T>)>,
    /// Best model (trained on full dataset)
    pub best_estimator_metrics: ModelMetrics<T>,
}

/// Model selection and comparison framework
#[derive(Debug)]
pub struct ModelSelector<T: FloatBounds + ScalarOperand + ToPrimitive + std::iter::Sum> {
    /// Cross-validation configuration
    cv_config: CrossValidationConfig,
    /// Comparison results
    results: Vec<ModelMetrics<T>>,
}

impl<T: FloatBounds + ScalarOperand + ToPrimitive + std::iter::Sum> ModelSelector<T> {
    /// Create a new model selector
    pub fn new(cv_config: CrossValidationConfig) -> Self {
        Self {
            cv_config,
            results: Vec::new(),
        }
    }

    /// Perform k-fold cross-validation on a classifier
    pub fn cross_validate_classifier(
        &self,
        X: &Array2<T>,
        y: &[usize],
        hidden_layer_sizes: &[usize],
        activation: Activation,
        learning_rate: T,
        alpha: T,
        max_iter: usize,
    ) -> Result<CrossValidationResults<T>, SklearsError> {
        let folds = self.create_folds(X, Some(y))?;
        let mut scores = Vec::new();

        for (train_indices, val_indices) in folds {
            // Create training and validation sets
            let X_train = self.select_rows(X, &train_indices)?;
            let y_train: Vec<usize> = train_indices.iter().map(|&i| y[i]).collect();
            let X_val = self.select_rows(X, &val_indices)?;
            let y_val: Vec<usize> = val_indices.iter().map(|&i| y[i]).collect();

            // Train classifier
            let classifier = MLPClassifier::new()
                .hidden_layer_sizes(hidden_layer_sizes)
                .activation(activation.clone())
                .learning_rate_init(learning_rate.to_f64().unwrap())
                .alpha(alpha.to_f64().unwrap())
                .max_iter(max_iter);

            // Convert T arrays to f64 for neural network models
            let X_train_f64 = X_train.mapv(|x| x.to_f64().unwrap_or(0.0));
            let X_val_f64 = X_val.mapv(|x| x.to_f64().unwrap_or(0.0));

            let trained_classifier = classifier.fit(&X_train_f64, &y_train)?;
            let predictions = trained_classifier.predict(&X_val_f64)?;

            // Compute accuracy
            let accuracy = self.compute_accuracy(&predictions, &y_val);
            scores.push(T::from(accuracy).unwrap());
        }

        Ok(CrossValidationResults::from_scores(scores))
    }

    /// Perform k-fold cross-validation on a regressor
    pub fn cross_validate_regressor(
        &self,
        X: &Array2<T>,
        y: &Array1<T>,
        hidden_layer_sizes: &[usize],
        activation: Activation,
        learning_rate: T,
        alpha: T,
        max_iter: usize,
    ) -> Result<CrossValidationResults<T>, SklearsError> {
        let folds = self.create_folds(X, None)?;
        let mut scores = Vec::new();

        for (train_indices, val_indices) in folds {
            // Create training and validation sets
            let X_train = self.select_rows(X, &train_indices)?;
            let y_train = self.select_elements(y, &train_indices)?;
            let X_val = self.select_rows(X, &val_indices)?;
            let y_val = self.select_elements(y, &val_indices)?;

            // Train regressor
            let regressor = MLPRegressor::new()
                .hidden_layer_sizes(hidden_layer_sizes)
                .activation(activation.clone())
                .learning_rate_init(learning_rate.to_f64().unwrap())
                .alpha(alpha.to_f64().unwrap())
                .max_iter(max_iter);

            // Convert T arrays to f64 for neural network models
            let X_train_f64 = X_train.mapv(|x| x.to_f64().unwrap_or(0.0));
            let X_val_f64 = X_val.mapv(|x| x.to_f64().unwrap_or(0.0));
            let y_train_f64 = y_train.mapv(|x| x.to_f64().unwrap_or(0.0));

            // Reshape y for neural network (expects 2D)
            let y_train_2d = y_train_f64.insert_axis(scirs2_core::ndarray::Axis(1));

            let trained_regressor = regressor.fit(&X_train_f64, &y_train_2d)?;
            let predictions = trained_regressor.predict(&X_val_f64)?;

            // Convert predictions back to 1D for scoring and convert to T
            let predictions_1d_f64 = predictions.column(0).to_owned();
            let predictions_1d =
                predictions_1d_f64.mapv(|x| T::from(x).unwrap_or_else(|| T::zero()));

            // Compute R² score
            let r2_score = self.compute_r2_score(&predictions_1d, &y_val);
            scores.push(T::from(r2_score).unwrap());
        }

        Ok(CrossValidationResults::from_scores(scores))
    }

    /// Perform grid search for classifier
    pub fn grid_search_classifier(
        &self,
        X: &Array2<T>,
        y: &[usize],
        config: GridSearchConfig<T>,
    ) -> Result<GridSearchResults<T>, SklearsError> {
        let mut best_score = T::neg_infinity();
        let mut best_params = None;
        let mut all_results = Vec::new();

        // Try all parameter combinations
        for hidden_layers in &config.hidden_layer_sizes {
            for &learning_rate in &config.learning_rates {
                for &alpha in &config.alphas {
                    for activation in &config.activations {
                        for &max_iter in &config.max_iters {
                            let params = HyperparameterSet {
                                hidden_layer_sizes: hidden_layers.clone(),
                                learning_rate,
                                alpha,
                                activation: activation.clone(),
                                max_iter,
                            };

                            let cv_results = self.cross_validate_classifier(
                                X,
                                y,
                                hidden_layers,
                                activation.clone(),
                                learning_rate,
                                alpha,
                                max_iter,
                            )?;

                            if cv_results.mean_score > best_score {
                                best_score = cv_results.mean_score;
                                best_params = Some(params.clone());
                            }

                            all_results.push((params, cv_results));
                        }
                    }
                }
            }
        }

        let best_params = best_params.ok_or_else(|| SklearsError::InvalidParameter {
            name: "grid_search".to_string(),
            reason: "No valid parameter combinations found".to_string(),
        })?;

        // Train final model with best parameters
        let final_classifier = MLPClassifier::new()
            .hidden_layer_sizes(&best_params.hidden_layer_sizes)
            .activation(best_params.activation.clone())
            .learning_rate_init(best_params.learning_rate.to_f64().unwrap())
            .alpha(best_params.alpha.to_f64().unwrap())
            .max_iter(best_params.max_iter);

        // Convert to f64 for neural network model
        let x_f64 = X.mapv(|x| x.to_f64().unwrap_or(0.0));
        let y_vec = y.to_vec();
        let _trained_final = final_classifier.fit(&x_f64, &y_vec)?;

        let best_estimator_metrics = ModelMetrics {
            model_name: "Best_MLP_Classifier".to_string(),
            train_score: best_score,
            validation_score: best_score,
            test_score: None,
            training_time: T::zero(),
            num_parameters: 0,
            complexity_score: T::zero(),
            additional_metrics: HashMap::new(),
        };

        Ok(GridSearchResults {
            best_params,
            best_score,
            all_results,
            best_estimator_metrics,
        })
    }

    /// Compare multiple models
    pub fn compare_models(&mut self, models: Vec<ModelMetrics<T>>) -> Vec<ModelMetrics<T>> {
        self.results.extend(models);

        // Sort by validation score (descending)
        self.results
            .sort_by(|a, b| b.validation_score.partial_cmp(&a.validation_score).unwrap());

        self.results.clone()
    }

    /// Get model rankings
    pub fn get_rankings(&self) -> Vec<(usize, &ModelMetrics<T>)> {
        self.results.iter().enumerate().collect()
    }

    /// Create k-fold splits
    fn create_folds(
        &self,
        X: &Array2<T>,
        y: Option<&[usize]>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, SklearsError> {
        let n_samples = X.nrows();
        let fold_size = n_samples / self.cv_config.n_folds;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.cv_config.shuffle {
            if let Some(seed) = self.cv_config.random_seed {
                let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(seed);
                for i in (1..indices.len()).rev() {
                    let j = rng.gen_range(0..=i);
                    indices.swap(i, j);
                }
            }
        }

        let mut folds = Vec::new();

        for fold in 0..self.cv_config.n_folds {
            let start = fold * fold_size;
            let end = if fold == self.cv_config.n_folds - 1 {
                n_samples // Include remaining samples in last fold
            } else {
                (fold + 1) * fold_size
            };

            let val_indices = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices[0..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            folds.push((train_indices, val_indices));
        }

        Ok(folds)
    }

    /// Select rows by indices
    fn select_rows(&self, X: &Array2<T>, indices: &[usize]) -> Result<Array2<T>, SklearsError> {
        let mut selected = Array2::zeros((indices.len(), X.ncols()));

        for (i, &idx) in indices.iter().enumerate() {
            if idx < X.nrows() {
                selected.row_mut(i).assign(&X.row(idx));
            }
        }

        Ok(selected)
    }

    /// Select elements by indices
    fn select_elements(&self, y: &Array1<T>, indices: &[usize]) -> Result<Array1<T>, SklearsError> {
        let mut selected = Array1::zeros(indices.len());

        for (i, &idx) in indices.iter().enumerate() {
            if idx < y.len() {
                selected[i] = y[idx];
            }
        }

        Ok(selected)
    }

    /// Compute classification accuracy
    fn compute_accuracy(&self, predictions: &[usize], y_true: &[usize]) -> f64 {
        if predictions.len() != y_true.len() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .filter(|(pred, true_)| pred == true_)
            .count();

        correct as f64 / predictions.len() as f64
    }

    /// Compute R² score for regression
    fn compute_r2_score(&self, predictions: &Array1<T>, y_true: &Array1<T>) -> f64 {
        if predictions.len() != y_true.len() {
            return 0.0;
        }

        // Convert to f64 for calculations
        let pred_f64: Vec<f64> = predictions
            .iter()
            .map(|&x| x.to_f64().unwrap_or(0.0))
            .collect();
        let true_f64: Vec<f64> = y_true.iter().map(|&x| x.to_f64().unwrap_or(0.0)).collect();

        let mean_true = true_f64.iter().sum::<f64>() / true_f64.len() as f64;

        let ss_res: f64 = pred_f64
            .iter()
            .zip(true_f64.iter())
            .map(|(pred, true_)| {
                let diff = true_ - pred;
                diff * diff
            })
            .sum();

        let ss_tot: f64 = true_f64
            .iter()
            .map(|true_| {
                let diff = true_ - mean_true;
                diff * diff
            })
            .sum();

        if ss_tot == 0.0 {
            0.0
        } else {
            1.0 - (ss_res / ss_tot)
        }
    }
}

/// Learning curve analysis
#[derive(Debug, Clone)]
pub struct LearningCurveAnalyzer<T: Float> {
    /// Training set sizes to analyze
    pub train_sizes: Vec<usize>,
    /// Training scores
    pub train_scores: Vec<Vec<T>>,
    /// Validation scores
    pub validation_scores: Vec<Vec<T>>,
}

impl<T: Float + std::iter::Sum> LearningCurveAnalyzer<T> {
    /// Create new learning curve analyzer
    pub fn new() -> Self {
        Self {
            train_sizes: Vec::new(),
            train_scores: Vec::new(),
            validation_scores: Vec::new(),
        }
    }

    /// Analyze learning curves
    pub fn analyze_learning_curve(
        &mut self,
        X: &Array2<T>,
        y: &[usize],
        hidden_layer_sizes: &[usize],
        activation: Activation,
        learning_rate: f64,
        alpha: f64,
        max_iter: usize,
        train_sizes: Vec<usize>,
    ) -> Result<(), SklearsError> {
        self.train_sizes = train_sizes.clone();
        self.train_scores.clear();
        self.validation_scores.clear();

        for &train_size in &train_sizes {
            if train_size > X.nrows() {
                continue;
            }

            // Create subset of data
            let X_subset = X
                .slice(scirs2_core::ndarray::s![..train_size, ..])
                .to_owned();
            let y_subset = &y[..train_size];

            // Split into train/validation
            let val_size = train_size / 5; // 20% for validation
            let train_end = train_size - val_size;

            let X_train = X_subset
                .slice(scirs2_core::ndarray::s![..train_end, ..])
                .to_owned();
            let y_train = &y_subset[..train_end];
            let X_val = X_subset
                .slice(scirs2_core::ndarray::s![train_end.., ..])
                .to_owned();
            let y_val = &y_subset[train_end..];

            // Train model
            let classifier = MLPClassifier::new()
                .hidden_layer_sizes(hidden_layer_sizes)
                .activation(activation.clone())
                .learning_rate_init(learning_rate)
                .alpha(alpha)
                .max_iter(max_iter);

            // Convert to f64 for neural network model
            let X_train_f64 = X_train.mapv(|x| x.to_f64().unwrap_or(0.0));
            let X_val_f64 = X_val.mapv(|x| x.to_f64().unwrap_or(0.0));
            let y_train_vec = y_train.to_vec();

            let trained = classifier.fit(&X_train_f64, &y_train_vec)?;

            // Evaluate on training set
            let train_pred = trained.predict(&X_train_f64)?;
            let train_accuracy = self.compute_accuracy(&train_pred, y_train);

            // Evaluate on validation set
            let val_pred = trained.predict(&X_val_f64)?;
            let val_accuracy = self.compute_accuracy(&val_pred, y_val);

            self.train_scores
                .push(vec![T::from(train_accuracy).unwrap()]);
            self.validation_scores
                .push(vec![T::from(val_accuracy).unwrap()]);
        }

        Ok(())
    }

    /// Get learning curve statistics
    pub fn get_statistics(&self) -> Result<(Vec<T>, Vec<T>, Vec<T>, Vec<T>), SklearsError> {
        let train_means: Vec<T> = self
            .train_scores
            .iter()
            .map(|scores| scores.iter().cloned().sum::<T>() / T::from(scores.len()).unwrap())
            .collect();

        let train_stds: Vec<T> = self
            .train_scores
            .iter()
            .zip(train_means.iter())
            .map(|(scores, &mean)| {
                let variance = scores
                    .iter()
                    .map(|&score| {
                        let diff = score - mean;
                        diff * diff
                    })
                    .sum::<T>()
                    / T::from(scores.len()).unwrap();
                variance.sqrt()
            })
            .collect();

        let val_means: Vec<T> = self
            .validation_scores
            .iter()
            .map(|scores| scores.iter().cloned().sum::<T>() / T::from(scores.len()).unwrap())
            .collect();

        let val_stds: Vec<T> = self
            .validation_scores
            .iter()
            .zip(val_means.iter())
            .map(|(scores, &mean)| {
                let variance = scores
                    .iter()
                    .map(|&score| {
                        let diff = score - mean;
                        diff * diff
                    })
                    .sum::<T>()
                    / T::from(scores.len()).unwrap();
                variance.sqrt()
            })
            .collect();

        Ok((train_means, train_stds, val_means, val_stds))
    }

    /// Compute accuracy
    fn compute_accuracy(&self, predictions: &[usize], y_true: &[usize]) -> f64 {
        if predictions.len() != y_true.len() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .filter(|(pred, true_)| pred == true_)
            .count();

        correct as f64 / predictions.len() as f64
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_model_metrics_creation() {
        let mut metrics = ModelMetrics::<f32>::new("TestModel".to_string());
        metrics.train_score = 0.95;
        metrics.validation_score = 0.90;
        metrics.add_metric("precision".to_string(), 0.88);

        assert_eq!(metrics.model_name, "TestModel");
        assert_eq!(metrics.train_score, 0.95);
        assert_eq!(metrics.get_metric("precision"), Some(0.88));
    }

    #[test]
    fn test_cross_validation_results() {
        let scores = vec![0.8, 0.85, 0.82, 0.87, 0.83];
        let results = CrossValidationResults::from_scores(scores);

        assert_abs_diff_eq!(results.mean_score, 0.834, epsilon = 1e-3);
        assert!(results.std_score > 0.0);
        assert_eq!(results.best_fold, 3); // Index of 0.87
    }

    #[test]
    fn test_model_selector_creation() {
        let cv_config = CrossValidationConfig::default();
        let selector = ModelSelector::<f32>::new(cv_config);
        assert_eq!(selector.cv_config.n_folds, 5);
    }

    #[test]
    fn test_fold_creation() {
        let cv_config = CrossValidationConfig {
            n_folds: 3,
            random_seed: Some(42),
            shuffle: false,
            stratify: false,
        };
        let selector = ModelSelector::<f32>::new(cv_config);

        let X = Array2::from_shape_vec((9, 2), vec![1.0; 18]).unwrap();
        let folds = selector.create_folds(&X, None).unwrap();

        assert_eq!(folds.len(), 3);
        assert_eq!(folds[0].1.len(), 3); // First fold validation size
        assert_eq!(folds[0].0.len(), 6); // First fold training size
    }

    #[test]
    fn test_accuracy_computation() {
        let selector = ModelSelector::<f32>::new(CrossValidationConfig::default());

        let predictions = vec![0, 1, 1, 0, 1];
        let y_true = vec![0, 1, 0, 0, 1];

        let accuracy = selector.compute_accuracy(&predictions, &y_true);
        assert_abs_diff_eq!(accuracy, 0.8, epsilon = 1e-6);
    }

    #[test]
    fn test_r2_score_computation() {
        let selector = ModelSelector::<f32>::new(CrossValidationConfig::default());

        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let y_true = Array1::from_vec(vec![1.1, 1.9, 3.1, 3.9]);

        let r2 = selector.compute_r2_score(&predictions, &y_true);
        assert!(r2 > 0.9); // Should be high for good predictions
    }

    #[test]
    fn test_grid_search_config_default() {
        let config = GridSearchConfig::<f32>::default();
        assert!(!config.hidden_layer_sizes.is_empty());
        assert!(!config.learning_rates.is_empty());
        assert!(!config.activations.is_empty());
    }

    #[test]
    fn test_hyperparameter_set() {
        let params = HyperparameterSet {
            hidden_layer_sizes: vec![100, 50],
            learning_rate: 0.01,
            alpha: 0.001,
            activation: Activation::Relu,
            max_iter: 1000,
        };

        assert_eq!(params.hidden_layer_sizes, vec![100, 50]);
        assert_eq!(params.learning_rate, 0.01);
    }

    #[test]
    fn test_learning_curve_analyzer() {
        let mut analyzer = LearningCurveAnalyzer::<f32>::new();
        assert!(analyzer.train_sizes.is_empty());
        assert!(analyzer.train_scores.is_empty());
    }

    #[test]
    fn test_model_comparison() {
        let cv_config = CrossValidationConfig::default();
        let mut selector = ModelSelector::<f32>::new(cv_config);

        let model1 = ModelMetrics {
            model_name: "Model1".to_string(),
            train_score: 0.90,
            validation_score: 0.85,
            test_score: None,
            training_time: 10.0,
            num_parameters: 1000,
            complexity_score: 0.5,
            additional_metrics: HashMap::new(),
        };

        let model2 = ModelMetrics {
            model_name: "Model2".to_string(),
            train_score: 0.88,
            validation_score: 0.87,
            test_score: None,
            training_time: 15.0,
            num_parameters: 1500,
            complexity_score: 0.7,
            additional_metrics: HashMap::new(),
        };

        let results = selector.compare_models(vec![model1, model2]);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].model_name, "Model2"); // Better validation score
    }
}
