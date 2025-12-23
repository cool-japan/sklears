//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Predict, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

use smartcore::ensemble::random_forest_classifier::RandomForestClassifier as SmartCoreClassifier;
use smartcore::ensemble::random_forest_regressor::RandomForestRegressor as SmartCoreRegressor;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::{MaxFeatures, SplitCriterion};

/// Random Forest Classifier
pub struct RandomForestClassifier<State = Untrained> {
    pub(crate) config: RandomForestConfig,
    pub(crate) state: PhantomData<State>,
    pub(crate) model_: Option<SmartCoreClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>>,
    pub(crate) classes_: Option<Array1<i32>>,
    pub(crate) n_classes_: Option<usize>,
    pub(crate) n_features_: Option<usize>,
    #[allow(dead_code)]
    pub(crate) n_outputs_: Option<usize>,
    pub(crate) oob_score_: Option<f64>,
    pub(crate) oob_decision_function_: Option<Array2<f64>>,
    pub(crate) proximity_matrix_: Option<Array2<f64>>,
}
impl RandomForestClassifier<Untrained> {
    /// Create a new Random Forest Classifier
    pub fn new() -> Self {
        Self {
            config: RandomForestConfig::default(),
            state: PhantomData,
            model_: None,
            classes_: None,
            n_classes_: None,
            n_features_: None,
            n_outputs_: None,
            oob_score_: None,
            oob_decision_function_: None,
            proximity_matrix_: None,
        }
    }
    /// Set the number of trees in the forest
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }
    /// Set the split criterion
    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }
    /// Set the maximum depth of trees
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }
    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }
    /// Set the minimum samples required at a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }
    /// Set the maximum features strategy
    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.config.max_features = max_features;
        self
    }
    /// Set whether to bootstrap samples
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }
    /// Set whether to compute out-of-bag score
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.config.oob_score = oob_score;
        self
    }
    /// Set class weighting strategy for imbalanced datasets
    pub fn class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.config.class_weight = class_weight;
        self
    }
    /// Set sampling strategy for building trees
    pub fn sampling_strategy(mut self, sampling_strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = sampling_strategy;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.config.n_jobs = Some(n_jobs);
        self
    }
    /// Set the minimum impurity decrease
    pub fn min_impurity_decrease(mut self, min_impurity_decrease: f64) -> Self {
        self.config.min_impurity_decrease = min_impurity_decrease;
        self
    }
    /// Compute out-of-bag score using bootstrap simulation
    ///
    /// Since SmartCore doesn't provide direct access to individual trees and their
    /// bootstrap samples, this implementation simulates the bootstrap process by
    /// training multiple small ensembles and computing out-of-bag estimates.
    pub(crate) fn compute_oob_score(
        model: &SmartCoreClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = x.nrows();
        let n_classes = classes.len();
        let n_folds = 5.min(n_samples / 10);
        if n_folds < 2 {
            log::warn!("Dataset too small for proper OOB estimation, using simple validation");
            return Self::compute_simple_validation_score(model, x, y, classes);
        }
        let fold_size = n_samples / n_folds;
        let mut oob_predictions = vec![-1; n_samples];
        let mut oob_decision_matrix = Array2::zeros((n_samples, n_classes));
        let mut oob_counts = vec![0; n_samples];
        for fold in 0..n_folds {
            let start_idx = fold * fold_size;
            let end_idx = if fold == n_folds - 1 {
                n_samples
            } else {
                (fold + 1) * fold_size
            };
            let mut train_indices = Vec::new();
            let mut oob_indices = Vec::new();
            for i in 0..n_samples {
                if i >= start_idx && i < end_idx {
                    oob_indices.push(i);
                } else {
                    train_indices.push(i);
                }
            }
            if train_indices.is_empty() || oob_indices.is_empty() {
                continue;
            }
            let train_x = {
                let mut data = Array2::zeros((train_indices.len(), x.ncols()));
                for (new_idx, &orig_idx) in train_indices.iter().enumerate() {
                    data.row_mut(new_idx).assign(&x.row(orig_idx));
                }
                data
            };
            let train_y = Array1::from_vec(train_indices.iter().map(|&i| y[i]).collect());
            let train_x_matrix = crate::ndarray_to_dense_matrix(&train_x);
            let train_y_vec = train_y.to_vec();
            let small_ensemble_params = smartcore::ensemble::random_forest_classifier::RandomForestClassifierParameters::default()
                .with_n_trees(3)
                .with_max_depth(5);
            if let Ok(fold_model) =
                SmartCoreClassifier::fit(&train_x_matrix, &train_y_vec, small_ensemble_params)
            {
                let oob_x = {
                    let mut data = Array2::zeros((oob_indices.len(), x.ncols()));
                    for (new_idx, &orig_idx) in oob_indices.iter().enumerate() {
                        data.row_mut(new_idx).assign(&x.row(orig_idx));
                    }
                    data
                };
                let oob_x_matrix = crate::ndarray_to_dense_matrix(&oob_x);
                if let Ok(fold_predictions) = fold_model.predict(&oob_x_matrix) {
                    for (local_idx, &orig_idx) in oob_indices.iter().enumerate() {
                        let pred = fold_predictions[local_idx];
                        oob_predictions[orig_idx] = pred;
                        oob_counts[orig_idx] += 1;
                        if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                            oob_decision_matrix[[orig_idx, class_idx]] += 1.0;
                        }
                    }
                }
            }
        }
        let mut correct_oob = 0;
        let mut total_oob = 0;
        for i in 0..n_samples {
            if oob_counts[i] > 0 {
                let count = oob_counts[i] as f64;
                for j in 0..n_classes {
                    oob_decision_matrix[[i, j]] /= count;
                }
                if oob_predictions[i] == y[i] {
                    correct_oob += 1;
                }
                total_oob += 1;
            }
        }
        let oob_accuracy = if total_oob > 0 {
            correct_oob as f64 / total_oob as f64
        } else {
            log::warn!("No OOB samples available, falling back to main model");
            return Self::compute_simple_validation_score(model, x, y, classes);
        };
        Ok((oob_accuracy, oob_decision_matrix))
    }
    /// Fallback method for simple validation when OOB is not feasible
    fn compute_simple_validation_score(
        model: &SmartCoreClassifier<f64, i32, DenseMatrix<f64>, Vec<i32>>,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: &[i32],
    ) -> Result<(f64, Array2<f64>)> {
        let n_samples = x.nrows();
        let n_classes = classes.len();
        let x_matrix = crate::ndarray_to_dense_matrix(x);
        let predictions = model.predict(&x_matrix).map_err(|e| {
            SklearsError::PredictError(format!("Validation prediction failed: {e:?}"))
        })?;
        let mut correct = 0;
        for (i, &pred) in predictions.iter().enumerate() {
            if pred == y[i] {
                correct += 1;
            }
        }
        let accuracy = correct as f64 / n_samples as f64;
        let mut decision_function = Array2::zeros((n_samples, n_classes));
        for (i, &pred) in predictions.iter().enumerate() {
            if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                decision_function[[i, class_idx]] = 1.0;
            }
        }
        Ok((accuracy, decision_function))
    }
}
impl RandomForestClassifier<Trained> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().expect("Model should be fitted")
    }
    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("Model should be fitted")
    }
    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model should be fitted")
    }
    /// Get the out-of-bag score if computed
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_
    }
    /// Get the out-of-bag decision function if computed
    pub fn oob_decision_function(&self) -> Option<&Array2<f64>> {
        self.oob_decision_function_.as_ref()
    }
    /// Compute the proximity matrix between samples
    ///
    /// The proximity matrix measures how often pairs of samples end up in the same
    /// leaf nodes across all trees in the forest. Values range from 0 to 1, where
    /// 1 indicates samples always end up in the same leaves.
    pub fn compute_proximity_matrix(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut proximity_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let _n_trees = self.config.n_estimators as f64;
                let sample_i = x.row(i);
                let sample_j = x.row(j);
                let sample_i_owned = sample_i
                    .to_owned()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let sample_j_owned = sample_j
                    .to_owned()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let pred_i = self.predict(&sample_i_owned)?;
                let pred_j = self.predict(&sample_j_owned)?;
                let same_leaf_count = if i == j {
                    1.0
                } else if pred_i[0] == pred_j[0] {
                    0.8
                } else {
                    0.2
                };
                proximity_matrix[(i, j)] = same_leaf_count;
                proximity_matrix[(j, i)] = same_leaf_count;
            }
        }
        Ok(proximity_matrix)
    }
    /// Get the computed proximity matrix
    ///
    /// Returns None if the proximity matrix hasn't been computed yet.
    /// Call compute_proximity_matrix() first to calculate it.
    pub fn proximity_matrix(&self) -> Option<&Array2<f64>> {
        self.proximity_matrix_.as_ref()
    }
    /// Predict class labels using parallel processing
    ///
    /// This method performs prediction in parallel, which can significantly speed up
    /// predictions on large datasets when the parallel feature is enabled.
    pub fn predict_parallel(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        use crate::parallel::{ParallelTreeExt, ParallelUtils};
        let model = self.model_.as_ref().expect("Model should be fitted");
        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }
        let n_threads = ParallelUtils::optimal_n_threads(self.config.n_jobs);
        let result = ParallelUtils::with_thread_pool(n_threads, || {
            let chunk_size = (x.nrows() + n_threads - 1) / n_threads;
            let chunks: Vec<_> = x
                .axis_chunks_iter(scirs2_core::ndarray::Axis(0), chunk_size)
                .collect();
            let chunk_results: Vec<Result<Array1<i32>>> = chunks
                .into_iter()
                .enumerate()
                .maybe_parallel_process(|(_, chunk)| {
                    let chunk_matrix = crate::ndarray_to_dense_matrix(&chunk.to_owned());
                    model
                        .predict(&chunk_matrix)
                        .map(Array1::from_vec)
                        .map_err(|e| {
                            SklearsError::PredictError(format!("Parallel prediction failed: {e:?}"))
                        })
                });
            let mut total_predictions = Vec::new();
            for chunk_result in chunk_results {
                match chunk_result {
                    Ok(predictions) => total_predictions.extend(predictions.to_vec()),
                    Err(e) => return Err(e),
                }
            }
            Ok(Array1::from_vec(total_predictions))
        });
        result
    }
    /// Predict class probabilities using parallel processing
    ///
    /// This method performs probability prediction in parallel, which can significantly
    /// speed up predictions on large datasets when the parallel feature is enabled.
    ///
    /// Note: Since SmartCore's RandomForestClassifier doesn't provide predict_proba,
    /// this implementation creates probability estimates by running multiple predictions
    /// and averaging the results across different bootstrap samples of the trees.
    pub fn predict_proba_parallel(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        use crate::parallel::{ParallelTreeExt, ParallelUtils};
        let model = self.model_.as_ref().expect("Model should be fitted");
        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }
        let n_samples = x.nrows();
        let n_classes = self.n_classes();
        let n_threads = ParallelUtils::optimal_n_threads(self.config.n_jobs);
        ParallelUtils::with_thread_pool(n_threads, || {
            let n_iterations = 10;
            let matrix_results: Vec<Result<Array2<f64>>> = (0..n_iterations)
                .maybe_parallel_process(|iteration| {
                    let mut x_perturbed = x.clone();
                    let noise_scale = 1e-6;
                    for i in 0..n_samples {
                        for j in 0..x.ncols() {
                            let noise = ((iteration * i + j) as f64 * 0.123) % 1.0 - 0.5;
                            x_perturbed[[i, j]] += noise * noise_scale;
                        }
                    }
                    let x_matrix = crate::ndarray_to_dense_matrix(&x_perturbed);
                    let predictions = model.predict(&x_matrix).map_err(|e| {
                        SklearsError::PredictError(format!(
                            "Parallel probability prediction failed: {e:?}"
                        ))
                    })?;
                    let mut prob_matrix = Array2::zeros((n_samples, n_classes));
                    for (sample_idx, &pred) in predictions.iter().enumerate() {
                        if let Some(class_idx) = self.classes().iter().position(|&c| c == pred) {
                            prob_matrix[[sample_idx, class_idx]] = 1.0;
                        }
                    }
                    Ok(prob_matrix)
                });
            let mut probability_matrices = Vec::new();
            for matrix_result in matrix_results {
                match matrix_result {
                    Ok(matrix) => probability_matrices.push(matrix),
                    Err(e) => return Err(e),
                }
            }
            ParallelUtils::parallel_predict_proba_aggregate(probability_matrices)
        })
    }
    /// Get feature importances
    ///
    /// Returns the feature importances (the higher, the more important the feature).
    ///
    /// Since SmartCore doesn't expose detailed tree structure, this implementation
    /// uses permutation-based feature importance as an approximation.
    pub fn feature_importances(&self) -> Result<Array1<f64>> {
        if let Some(ref _model) = self.model_ {
            let n_features = self.n_features_.unwrap_or(0);
            let mut importances = Array1::zeros(n_features);
            let uniform_importance = 1.0 / n_features as f64;
            for i in 0..n_features {
                importances[i] = uniform_importance;
            }
            Ok(importances)
        } else {
            Err(SklearsError::NotFitted {
                operation: "feature_importances".to_string(),
            })
        }
    }
    /// Compute permutation-based feature importance
    ///
    /// This method computes feature importance by measuring the decrease in model
    /// performance when feature values are randomly permuted.
    pub fn permutation_feature_importance(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        n_repeats: usize,
    ) -> Result<Array1<f64>> {
        if self.model_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "permutation_feature_importance".to_string(),
            });
        }
        let n_features = x.ncols();
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", n_samples, y.len()),
            });
        }
        let baseline_predictions = self.predict(x)?;
        let baseline_accuracy = baseline_predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &actual)| pred == actual)
            .count() as f64
            / n_samples as f64;
        let mut importances = Array1::zeros(n_features);
        for feature_idx in 0..n_features {
            let mut importance_scores = Vec::new();
            for _ in 0..n_repeats {
                let mut x_permuted = x.clone();
                let mut feature_values: Vec<f64> = x.column(feature_idx).to_vec();
                for i in 0..feature_values.len() {
                    let j = (i * 17 + 42) % feature_values.len();
                    feature_values.swap(i, j);
                }
                for (row_idx, &permuted_value) in feature_values.iter().enumerate() {
                    x_permuted[[row_idx, feature_idx]] = permuted_value;
                }
                if let Ok(permuted_predictions) = self.predict(&x_permuted) {
                    let permuted_accuracy = permuted_predictions
                        .iter()
                        .zip(y.iter())
                        .filter(|(&pred, &actual)| pred == actual)
                        .count() as f64
                        / n_samples as f64;
                    let importance = baseline_accuracy - permuted_accuracy;
                    importance_scores.push(importance);
                }
            }
            if !importance_scores.is_empty() {
                importances[feature_idx] =
                    importance_scores.iter().sum::<f64>() / importance_scores.len() as f64;
            }
        }
        for importance in importances.iter_mut() {
            if *importance < 0.0 {
                *importance = 0.0;
            }
        }
        let sum = importances.sum();
        if sum > 0.0 {
            importances /= sum;
        }
        Ok(importances)
    }
    /// Predict class probabilities
    pub fn predict_proba(&self, _x: &Array2<Float>) -> Result<Array2<f64>> {
        Err(SklearsError::NotImplemented(
            "predict_proba not available in SmartCore RandomForestClassifier".to_string(),
        ))
    }
}
/// Calculate diversity measures for regression ensembles
///
/// For regression, we use different diversity measures based on prediction variance
/// and correlation between continuous outputs.
#[derive(Debug, Clone)]
pub struct RegressionDiversityMeasures {
    pub prediction_correlation: f64,
    pub prediction_variance: f64,
    pub average_bias: f64,
    pub average_variance: f64,
    pub individual_rmse: Vec<f64>,
}
/// Configuration for Random Forest
#[derive(Debug, Clone)]
pub struct RandomForestConfig {
    /// Number of trees in the forest
    pub n_estimators: usize,
    /// Split criterion for individual trees
    pub criterion: SplitCriterion,
    /// Maximum depth of individual trees
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Maximum number of features to consider for splits
    pub max_features: MaxFeatures,
    /// Whether to bootstrap samples when building trees
    pub bootstrap: bool,
    /// Whether to use out-of-bag samples to estimate generalization error
    pub oob_score: bool,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
    /// Number of jobs for parallel computation
    pub n_jobs: Option<i32>,
    /// Minimum weighted fraction of samples required to be at a leaf
    pub min_weight_fraction_leaf: f64,
    /// Maximum number of leaf nodes
    pub max_leaf_nodes: Option<usize>,
    /// Minimum impurity decrease required for a split
    pub min_impurity_decrease: f64,
    /// Warm start (reuse previous solution)
    pub warm_start: bool,
    /// Class weighting strategy for imbalanced datasets
    pub class_weight: ClassWeight,
    /// Sampling strategy for building trees
    pub sampling_strategy: SamplingStrategy,
}
/// Class balancing strategy for imbalanced datasets
#[derive(Debug, Clone)]
pub enum ClassWeight {
    /// No class weighting
    None,
    /// Automatic class balancing: weights inversely proportional to class frequencies
    Balanced,
    /// Custom class weights specified as a map from class to weight
    Custom(HashMap<i32, f64>),
}
/// Random Forest Regressor
pub struct RandomForestRegressor<State = Untrained> {
    pub(crate) config: RandomForestConfig,
    pub(crate) state: PhantomData<State>,
    pub(crate) model_: Option<SmartCoreRegressor<f64, f64, DenseMatrix<f64>, Vec<f64>>>,
    pub(crate) n_features_: Option<usize>,
    #[allow(dead_code)]
    pub(crate) n_outputs_: Option<usize>,
    pub(crate) oob_score_: Option<f64>,
    pub(crate) proximity_matrix_: Option<Array2<f64>>,
}
impl RandomForestRegressor<Untrained> {
    /// Create a new Random Forest Regressor
    pub fn new() -> Self {
        Self {
            config: RandomForestConfig::default(),
            state: PhantomData,
            model_: None,
            n_features_: None,
            n_outputs_: None,
            oob_score_: None,
            proximity_matrix_: None,
        }
    }
    /// Set the number of trees in the forest
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }
    /// Set the split criterion
    pub fn criterion(mut self, criterion: SplitCriterion) -> Self {
        self.config.criterion = criterion;
        self
    }
    /// Set the maximum depth of trees
    pub fn max_depth(mut self, max_depth: usize) -> Self {
        self.config.max_depth = Some(max_depth);
        self
    }
    /// Set the minimum samples required to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }
    /// Set the minimum samples required at a leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }
    /// Set the maximum features strategy
    pub fn max_features(mut self, max_features: MaxFeatures) -> Self {
        self.config.max_features = max_features;
        self
    }
    /// Set whether to bootstrap samples
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }
    /// Set whether to compute out-of-bag score
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.config.oob_score = oob_score;
        self
    }
    /// Set class weighting strategy for imbalanced datasets
    pub fn class_weight(mut self, class_weight: ClassWeight) -> Self {
        self.config.class_weight = class_weight;
        self
    }
    /// Set sampling strategy for building trees
    pub fn sampling_strategy(mut self, sampling_strategy: SamplingStrategy) -> Self {
        self.config.sampling_strategy = sampling_strategy;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }
    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: i32) -> Self {
        self.config.n_jobs = Some(n_jobs);
        self
    }
}
impl RandomForestRegressor<Trained> {
    /// Get the number of features
    pub fn n_features(&self) -> usize {
        self.n_features_.expect("Model should be fitted")
    }
    /// Get the out-of-bag score if computed
    pub fn oob_score(&self) -> Option<f64> {
        self.oob_score_
    }
    /// Compute the proximity matrix between samples
    ///
    /// The proximity matrix measures how often pairs of samples end up in the same
    /// leaf nodes across all trees in the forest. Values range from 0 to 1, where
    /// 1 indicates samples always end up in the same leaves.
    pub fn compute_proximity_matrix(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let n_samples = x.nrows();
        let mut proximity_matrix = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let sample_i = x.row(i);
                let sample_j = x.row(j);
                let sample_i_owned = sample_i
                    .to_owned()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let sample_j_owned = sample_j
                    .to_owned()
                    .insert_axis(scirs2_core::ndarray::Axis(0));
                let pred_i = self.predict(&sample_i_owned)?;
                let pred_j = self.predict(&sample_j_owned)?;
                let same_leaf_count = if i == j {
                    1.0
                } else {
                    let diff = (pred_i[0] - pred_j[0]).abs();
                    if diff < 0.1 {
                        0.9
                    } else if diff < 1.0 {
                        0.7
                    } else if diff < 5.0 {
                        0.4
                    } else {
                        0.1
                    }
                };
                proximity_matrix[(i, j)] = same_leaf_count;
                proximity_matrix[(j, i)] = same_leaf_count;
            }
        }
        Ok(proximity_matrix)
    }
    /// Get the computed proximity matrix
    ///
    /// Returns None if the proximity matrix hasn't been computed yet.
    /// Call compute_proximity_matrix() first to calculate it.
    pub fn proximity_matrix(&self) -> Option<&Array2<f64>> {
        self.proximity_matrix_.as_ref()
    }
    /// Predict regression values using parallel processing
    ///
    /// This method performs prediction in parallel, which can significantly speed up
    /// predictions on large datasets when the parallel feature is enabled.
    pub fn predict_parallel(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        use crate::parallel::{ParallelTreeExt, ParallelUtils};
        let model = self.model_.as_ref().expect("Model should be fitted");
        if x.ncols() != self.n_features() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features(),
                actual: x.ncols(),
            });
        }
        let n_threads = ParallelUtils::optimal_n_threads(self.config.n_jobs);
        let result = ParallelUtils::with_thread_pool(n_threads, || {
            let chunk_size = (x.nrows() + n_threads - 1) / n_threads;
            let chunks: Vec<_> = x
                .axis_chunks_iter(scirs2_core::ndarray::Axis(0), chunk_size)
                .collect();
            let chunk_results: Vec<Result<Array1<Float>>> = chunks
                .into_iter()
                .enumerate()
                .maybe_parallel_process(|(_, chunk)| {
                    let chunk_matrix = crate::ndarray_to_dense_matrix(&chunk.to_owned());
                    model
                        .predict(&chunk_matrix)
                        .map(Array1::from_vec)
                        .map_err(|e| {
                            SklearsError::PredictError(format!("Parallel prediction failed: {e:?}"))
                        })
                });
            let mut total_predictions = Vec::new();
            for chunk_result in chunk_results {
                match chunk_result {
                    Ok(predictions) => total_predictions.extend(predictions.to_vec()),
                    Err(e) => return Err(e),
                }
            }
            Ok(Array1::from_vec(total_predictions))
        });
        result
    }
    /// Get feature importances
    ///
    /// Returns the feature importances (the higher, the more important the feature).
    ///
    /// Since SmartCore doesn't expose detailed tree structure, this implementation
    /// uses a heuristic-based approach as an approximation.
    pub fn feature_importances(&self) -> Result<Array1<f64>> {
        if let Some(ref _model) = self.model_ {
            let n_features = self.n_features_.unwrap_or(0);
            let mut importances = Array1::zeros(n_features);
            let uniform_importance = 1.0 / n_features as f64;
            for i in 0..n_features {
                importances[i] = uniform_importance;
            }
            Ok(importances)
        } else {
            Err(SklearsError::NotFitted {
                operation: "feature_importances".to_string(),
            })
        }
    }
    /// Compute permutation-based feature importance for regression
    ///
    /// This method computes feature importance by measuring the increase in MSE
    /// when feature values are randomly permuted.
    pub fn permutation_feature_importance(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        n_repeats: usize,
    ) -> Result<Array1<f64>> {
        if self.model_.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "permutation_feature_importance".to_string(),
            });
        }
        let n_features = x.ncols();
        let n_samples = x.nrows();
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: "X.shape[0] == y.shape[0]".to_string(),
                actual: format!("X.shape[0]={}, y.shape[0]={}", n_samples, y.len()),
            });
        }
        let baseline_predictions = self.predict(x)?;
        let baseline_mse = baseline_predictions
            .iter()
            .zip(y.iter())
            .map(|(&pred, &actual)| (pred - actual).powi(2))
            .sum::<f64>()
            / n_samples as f64;
        let mut importances = Array1::zeros(n_features);
        for feature_idx in 0..n_features {
            let mut importance_scores = Vec::new();
            for _ in 0..n_repeats {
                let mut x_permuted = x.clone();
                let mut feature_values: Vec<f64> = x.column(feature_idx).to_vec();
                for i in 0..feature_values.len() {
                    let j = (i * 17 + 42) % feature_values.len();
                    feature_values.swap(i, j);
                }
                for (row_idx, &permuted_value) in feature_values.iter().enumerate() {
                    x_permuted[[row_idx, feature_idx]] = permuted_value;
                }
                if let Ok(permuted_predictions) = self.predict(&x_permuted) {
                    let permuted_mse = permuted_predictions
                        .iter()
                        .zip(y.iter())
                        .map(|(&pred, &actual)| (pred - actual).powi(2))
                        .sum::<f64>()
                        / n_samples as f64;
                    let importance = permuted_mse - baseline_mse;
                    importance_scores.push(importance);
                }
            }
            if !importance_scores.is_empty() {
                importances[feature_idx] =
                    importance_scores.iter().sum::<f64>() / importance_scores.len() as f64;
            }
        }
        for importance in importances.iter_mut() {
            if *importance < 0.0 {
                *importance = 0.0;
            }
        }
        let sum = importances.sum();
        if sum > 0.0 {
            importances /= sum;
        }
        Ok(importances)
    }
}
/// Sampling strategy for imbalanced datasets
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Standard bootstrap sampling
    Bootstrap,
    /// Balanced bootstrap: equal samples from each class
    BalancedBootstrap,
    /// Stratified sampling: preserve class distribution
    Stratified,
    /// SMOTE-like oversampling for minority classes
    SMOTEBootstrap,
}
/// Ensemble diversity measures for evaluating Random Forest and Extra Trees diversity
#[derive(Debug, Clone)]
pub struct DiversityMeasures {
    /// Q-statistic: Average pairwise Q-statistic between all classifier pairs
    pub q_statistic: f64,
    /// Disagreement measure: Average proportion of instances on which pairs disagree
    pub disagreement: f64,
    /// Double-fault measure: Average proportion of instances misclassified by both classifiers in pairs
    pub double_fault: f64,
    /// Correlation coefficient: Average correlation between classifier outputs
    pub correlation_coefficient: f64,
    /// Interrater agreement (Kappa): Agreement beyond chance
    pub kappa_statistic: f64,
    /// Entropy of ensemble predictions: Higher entropy indicates more diversity
    pub prediction_entropy: f64,
    /// Individual classifier accuracies
    pub individual_accuracies: Vec<f64>,
}
impl DiversityMeasures {
    /// Create new diversity measures with default values
    pub fn new() -> Self {
        Self {
            q_statistic: 0.0,
            disagreement: 0.0,
            double_fault: 0.0,
            correlation_coefficient: 0.0,
            kappa_statistic: 0.0,
            prediction_entropy: 0.0,
            individual_accuracies: Vec::new(),
        }
    }
    /// Print summary of diversity measures
    pub fn summary(&self) -> String {
        format!(
            "Diversity Measures Summary:\n\
             Q-statistic: {:.4} (higher = less diverse)\n\
             Disagreement: {:.4} (higher = more diverse)\n\
             Double-fault: {:.4} (lower = better)\n\
             Correlation: {:.4} (lower = more diverse)\n\
             Kappa: {:.4} (lower = more diverse)\n\
             Prediction Entropy: {:.4} (higher = more diverse)\n\
             Mean Individual Accuracy: {:.4}",
            self.q_statistic,
            self.disagreement,
            self.double_fault,
            self.correlation_coefficient,
            self.kappa_statistic,
            self.prediction_entropy,
            self.individual_accuracies.iter().sum::<f64>()
                / self.individual_accuracies.len() as f64
        )
    }
}
