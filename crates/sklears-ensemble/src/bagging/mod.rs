// Copyright 2026 COOLJAPAN OU (Team KitaSan)
// SPDX-License-Identifier: Apache-2.0

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::prelude::*;
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::Predict;
use sklears_core::traits::{Estimator, Fit, Trained, Untrained};
use sklears_core::types::{Float, Int};
use crate::adaboost::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};
#[allow(unused_imports)]
use std::collections::HashSet;
use std::marker::PhantomData;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
/// Convenience type alias for bootstrap sample data tuple
type BootstrapSampleData = (Array2<Float>, Array1<Int>, Vec<usize>, Vec<usize>);
/// Convenience type alias for trained classifier ensemble results
type ClassifierEnsembleResult = (
    Vec<DecisionTreeClassifier<Trained>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);
/// Convenience type alias for regressor bootstrap sample data tuple
type RegressorBootstrapSampleData = (
    Array2<Float>,
    Array1<Float>,
    Vec<usize>,
    Vec<usize>,
);
/// Convenience type alias for trained regressor ensemble results
type RegressorEnsembleResult = (
    Vec<DecisionTreeRegressor<Trained>>,
    Vec<Vec<usize>>,
    Vec<Vec<usize>>,
);
/// Configuration for bagging ensemble
#[derive(Debug, Clone)]
pub struct BaggingConfig {
    /// Number of base estimators in the ensemble
    pub n_estimators: usize,
    /// Number of samples to draw from X to train each base estimator
    pub max_samples: Option<usize>,
    /// Number of features to draw from X to train each base estimator
    pub max_features: Option<usize>,
    /// Whether to use replacement when sampling
    pub bootstrap: bool,
    /// Whether to use replacement when sampling features
    pub bootstrap_features: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Out-of-bag score calculation
    pub oob_score: bool,
    /// Number of jobs for parallel execution
    pub n_jobs: Option<i32>,
    /// Maximum depth for decision tree base estimators
    pub max_depth: Option<usize>,
    /// Minimum samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Bootstrap confidence level for intervals
    pub confidence_level: Float,
    /// Use extra randomization (Extremely Randomized Trees)
    pub extra_randomized: bool,
}
impl Default for BaggingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_samples: None,
            max_features: None,
            bootstrap: true,
            bootstrap_features: false,
            random_state: None,
            oob_score: false,
            n_jobs: None,
            max_depth: None,
            min_samples_split: 2,
            min_samples_leaf: 1,
            confidence_level: 0.95,
            extra_randomized: false,
        }
    }
}
/// Enhanced Bagging classifier with OOB estimation and feature bagging
#[allow(dead_code)]
pub struct BaggingClassifier<State = Untrained> {
    config: BaggingConfig,
    state: PhantomData<State>,
    estimators_: Option<Vec<DecisionTreeClassifier<Trained>>>,
    estimators_features_: Option<Vec<Vec<usize>>>,
    estimators_samples_: Option<Vec<Vec<usize>>>,
    oob_score_: Option<Float>,
    oob_prediction_: Option<Array1<Float>>,
    classes_: Option<Array1<Int>>,
    n_classes_: Option<usize>,
    n_features_in_: Option<usize>,
    feature_importances_: Option<Array1<Float>>,
}
impl BaggingClassifier<Untrained> {
    /// Create a new bagging classifier
    pub fn new() -> Self {
        Self {
            config: BaggingConfig::default(),
            state: PhantomData,
            estimators_: None,
            estimators_features_: None,
            estimators_samples_: None,
            oob_score_: None,
            oob_prediction_: None,
            classes_: None,
            n_classes_: None,
            n_features_in_: None,
            feature_importances_: None,
        }
    }
    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }
    /// Set the maximum number of samples per estimator
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }
    /// Set the maximum number of features per estimator
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }
    /// Set whether to use bootstrap sampling
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }
    /// Set whether to use bootstrap feature sampling
    pub fn bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.config.bootstrap_features = bootstrap_features;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
    /// Set whether to calculate out-of-bag score
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.config.oob_score = oob_score;
        self
    }
    /// Set maximum depth for base estimators
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }
    /// Set minimum samples to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }
    /// Set minimum samples at leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }
    /// Set confidence level for bootstrap intervals
    pub fn confidence_level(mut self, confidence_level: Float) -> Self {
        self.config.confidence_level = confidence_level;
        self
    }
    /// Set number of parallel jobs for training (None for automatic detection)
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }
    /// Enable parallel training with automatic job detection
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }
    /// Enable extra randomization (Extremely Randomized Trees)
    pub fn extra_randomized(mut self, extra_randomized: bool) -> Self {
        self.config.extra_randomized = extra_randomized;
        self
    }
    /// Enable extra randomization (convenient shorthand)
    pub fn extremely_randomized(mut self) -> Self {
        self.config.extra_randomized = true;
        self.config.bootstrap = false;
        self
    }
    /// Generate bootstrap samples with optional out-of-bag tracking
    /// For extra randomized trees, returns the full dataset instead of bootstrap samples
    fn bootstrap_sample(
        &self,
        x: &Array2<Float>,
        y: &Array1<Int>,
        rng: &mut StdRng,
    ) -> Result<(Array2<Float>, Array1<Int>, Vec<usize>)> {
        let n_samples = x.nrows();
        if self.config.extra_randomized {
            let sample_indices: Vec<usize> = (0..n_samples).collect();
            return Ok((x.clone(), y.clone(), sample_indices));
        }
        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        let mut class_indices: std::collections::HashMap<Int, Vec<usize>> = std::collections::HashMap::new();
        for (idx, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_default().push(idx);
        }
        let mut sample_indices: Vec<usize> = if self.config.bootstrap {
            (0..max_samples).map(|_| rng.gen_range(0..n_samples)).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(max_samples);
            indices
        };
        let mut sampled_classes = std::collections::HashSet::new();
        for &idx in &sample_indices {
            sampled_classes.insert(y[idx]);
        }
        if sampled_classes.len() < 2 && class_indices.len() >= 2 {
            let mut other_classes: Vec<Int> = class_indices
                .keys()
                .filter(|&&c| !sampled_classes.contains(&c))
                .cloned()
                .collect();
            other_classes.sort();
            if !other_classes.is_empty() {
                let other_class = other_classes[0];
                if let Some(other_indices) = class_indices.get(&other_class) {
                    if !other_indices.is_empty() {
                        let replacement_idx = other_indices[0];
                        if let Some(last) = sample_indices.last_mut() {
                            *last = replacement_idx;
                        }
                    }
                }
            }
        }
        let mut x_bootstrap = Array2::zeros((max_samples, x.ncols()));
        let mut y_bootstrap = Array1::zeros(max_samples);
        for (i, &idx) in sample_indices.iter().enumerate() {
            x_bootstrap.row_mut(i).assign(&x.row(idx));
            y_bootstrap[i] = y[idx];
        }
        Ok((x_bootstrap, y_bootstrap, sample_indices))
    }
    /// Train ensemble using parallel processing with work-stealing when available
    fn train_ensemble_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<Int>,
        _rng: &mut StdRng,
        n_features: usize,
    ) -> Result<ClassifierEnsembleResult> {
        let mut bootstrap_data = Vec::new();
        for i in 0..self.config.n_estimators {
            let mut local_rng = StdRng::seed_from_u64(
                self.config.random_state.unwrap_or(42) + i as u64,
            );
            let (x_bootstrap, y_bootstrap, sample_indices) = self
                .bootstrap_sample(x, y, &mut local_rng)?;
            let feature_indices = self.get_feature_indices(n_features, &mut local_rng);
            bootstrap_data
                .push((x_bootstrap, y_bootstrap, sample_indices, feature_indices));
        }
        let use_parallel = self.should_use_parallel();
        if use_parallel {
            #[cfg(feature = "parallel")]
            {
                let results: Result<Vec<_>> = bootstrap_data
                    .into_par_iter()
                    .enumerate()
                    .map(|
                        (i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices))|
                    {
                        self.fit_single_estimator(
                                &x_bootstrap,
                                &y_bootstrap,
                                &feature_indices,
                                i,
                            )
                            .map(|estimator| (
                                estimator,
                                feature_indices,
                                sample_indices,
                            ))
                    })
                    .collect();
                let fitted_data = results?;
                let (estimators, estimators_features, estimators_samples) = fitted_data
                    .into_iter()
                    .fold(
                        (Vec::new(), Vec::new(), Vec::new()),
                        |(mut e, mut ef, mut es), (estimator, features, samples)| {
                            e.push(estimator);
                            ef.push(features);
                            es.push(samples);
                            (e, ef, es)
                        },
                    );
                Ok((estimators, estimators_features, estimators_samples))
            }
            #[cfg(not(feature = "parallel"))]
            { self.train_ensemble_sequential(bootstrap_data) }
        } else {
            self.train_ensemble_sequential(bootstrap_data)
        }
    }
    /// Train ensemble sequentially
    fn train_ensemble_sequential(
        &self,
        bootstrap_data: Vec<BootstrapSampleData>,
    ) -> Result<ClassifierEnsembleResult> {
        let mut estimators = Vec::new();
        let mut estimators_features = Vec::new();
        let mut estimators_samples = Vec::new();
        for (i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices)) in bootstrap_data
            .into_iter()
            .enumerate()
        {
            let fitted_tree = self
                .fit_single_estimator(&x_bootstrap, &y_bootstrap, &feature_indices, i)?;
            estimators.push(fitted_tree);
            estimators_features.push(feature_indices);
            estimators_samples.push(sample_indices);
        }
        Ok((estimators, estimators_features, estimators_samples))
    }
    /// Fit a single estimator with given bootstrap sample and feature indices
    fn fit_single_estimator(
        &self,
        x_bootstrap: &Array2<Float>,
        y_bootstrap: &Array1<Int>,
        feature_indices: &[usize],
        estimator_index: usize,
    ) -> Result<DecisionTreeClassifier<Trained>> {
        let mut x_features = Array2::zeros((x_bootstrap.nrows(), feature_indices.len()));
        for (j, &feature_idx) in feature_indices.iter().enumerate() {
            x_features.column_mut(j).assign(&x_bootstrap.column(feature_idx));
        }
        let mut tree = DecisionTreeClassifier::new()
            .criterion(SplitCriterion::Gini)
            .min_samples_split(self.config.min_samples_split)
            .min_samples_leaf(self.config.min_samples_leaf);
        if let Some(max_depth) = self.config.max_depth {
            tree = tree.max_depth(max_depth);
        }
        if let Some(seed) = self.config.random_state.map(|s| s + estimator_index as u64)
        {
            tree = tree.random_state(Some(seed));
        }
        tree.fit(&x_features, y_bootstrap)
    }
    /// Determine whether to use parallel processing based on configuration
    fn should_use_parallel(&self) -> bool {
        match self.config.n_jobs {
            Some(n) if n != 1 => true,
            None => false,
            _ => false,
        }
    }
    /// Generate feature indices for feature bagging
    fn get_feature_indices(&self, n_features: usize, rng: &mut StdRng) -> Vec<usize> {
        let max_features = self.config.max_features.unwrap_or(n_features);
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        if self.config.bootstrap_features {
            feature_indices = (0..max_features)
                .map(|_| rng.gen_range(0..n_features))
                .collect();
        } else {
            feature_indices.shuffle(rng);
            feature_indices.truncate(max_features);
        }
        feature_indices.sort_unstable();
        feature_indices
    }
    /// Calculate out-of-bag predictions for OOB score
    fn calculate_oob_predictions(
        &self,
        x: &Array2<Float>,
        y: &Array1<Int>,
        estimators: &[DecisionTreeClassifier<Trained>],
        estimators_features: &[Vec<usize>],
        estimators_samples: &[Vec<usize>],
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut oob_predictions: Array1<Float> = Array1::zeros(n_samples);
        let mut oob_counts: Array1<Float> = Array1::zeros(n_samples);
        for (estimator, (features, samples)) in estimators
            .iter()
            .zip(estimators_features.iter().zip(estimators_samples.iter()))
        {
            let mut oob_mask = vec![true; n_samples];
            for &sample_idx in samples {
                if sample_idx < n_samples {
                    oob_mask[sample_idx] = false;
                }
            }
            for (sample_idx, &is_oob) in oob_mask.iter().enumerate() {
                if is_oob {
                    let x_sample = x.row(sample_idx);
                    let x_features = Array2::from_shape_vec(
                            (1, features.len()),
                            features.iter().map(|&f| x_sample[f]).collect(),
                        )
                        .map_err(|_| {
                            SklearsError::InvalidInput(
                                "Failed to create feature subset".to_string(),
                            )
                        })?;
                    let pred = estimator.predict(&x_features)?;
                    oob_predictions[sample_idx] += pred[0] as Float;
                    oob_counts[sample_idx] += 1.0;
                }
            }
        }
        let mut correct = 0;
        let mut total = 0;
        for i in 0..n_samples {
            if oob_counts[i] > 0.0 {
                let ratio: Float = oob_predictions[i] / oob_counts[i];
                let predicted_class: Int = ratio.round() as Int;
                if predicted_class == y[i] {
                    correct += 1;
                }
                total += 1;
            }
        }
        if total == 0 { Ok(0.0) } else { Ok(correct as Float / total as Float) }
    }
}
impl Fit<Array2<Float>, Array1<Int>> for BaggingClassifier<Untrained> {
    type Fitted = BaggingClassifier<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<Int>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {}", n_samples),
                actual: format!("y.shape[0] = {}", y.len()),
            });
        }
        if n_samples == 0 {
            return Err(
                SklearsError::InvalidInput(
                    "Cannot fit bagging on empty dataset".to_string(),
                ),
            );
        }
        let mut unique_classes: Vec<Int> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();
        if n_classes < 2 {
            return Err(
                SklearsError::InvalidInput(
                    "Bagging requires at least 2 classes".to_string(),
                ),
            );
        }
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };
        let (estimators, estimators_features, estimators_samples) = self
            .train_ensemble_parallel(x, y, &mut rng, n_features)?;
        let oob_score = if self.config.oob_score {
            Some(
                self
                    .calculate_oob_predictions(
                        x,
                        y,
                        &estimators,
                        &estimators_features,
                        &estimators_samples,
                    )?,
            )
        } else {
            None
        };
        let mut feature_importances = Array1::zeros(n_features);
        for (_estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            let tree_importance = 1.0 / features.len() as Float;
            for &feature_idx in features {
                feature_importances[feature_idx] += tree_importance;
            }
        }
        let total_importance = feature_importances.sum();
        if total_importance > 0.0 {
            feature_importances /= total_importance;
        }
        Ok(BaggingClassifier {
            config: self.config,
            state: PhantomData,
            estimators_: Some(estimators),
            estimators_features_: Some(estimators_features.to_vec()),
            estimators_samples_: Some(estimators_samples.to_vec()),
            oob_score_: oob_score,
            oob_prediction_: None,
            classes_: Some(classes),
            n_classes_: Some(n_classes),
            n_features_in_: Some(n_features),
            feature_importances_: Some(feature_importances),
        })
    }
}
impl BaggingClassifier<Trained> {
    /// Get the fitted base estimators
    pub fn estimators(&self) -> &[DecisionTreeClassifier<Trained>] {
        self.estimators_.as_ref().expect("BaggingClassifier should be fitted")
    }
    /// Get the feature indices used by each estimator
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        self.estimators_features_.as_ref().expect("BaggingClassifier should be fitted")
    }
    /// Get the sample indices used by each estimator
    pub fn estimators_samples(&self) -> &[Vec<usize>] {
        self.estimators_samples_.as_ref().expect("BaggingClassifier should be fitted")
    }
    /// Get the out-of-bag score if calculated
    pub fn oob_score(&self) -> Option<Float> {
        self.oob_score_
    }
    /// Get the classes
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_.as_ref().expect("BaggingClassifier should be fitted")
    }
    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("BaggingClassifier should be fitted")
    }
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("BaggingClassifier should be fitted")
    }
    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_.as_ref().expect("BaggingClassifier should be fitted")
    }
    /// Calculate bootstrap confidence intervals for predictions
    pub fn predict_with_confidence(
        &self,
        x: &Array2<Float>,
    ) -> Result<(Array1<Int>, Array2<Float>)> {
        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }
        let estimators = self.estimators();
        let estimators_features = self.estimators_features();
        let classes = self.classes();
        let n_classes = self.n_classes();
        let n_estimators = estimators.len();
        let mut all_predictions = Array2::zeros((n_samples, n_estimators));
        for (estimator_idx, (estimator, features)) in estimators
            .iter()
            .zip(estimators_features.iter())
            .enumerate()
        {
            let mut x_features = Array2::zeros((n_samples, features.len()));
            for (j, &feature_idx) in features.iter().enumerate() {
                x_features.column_mut(j).assign(&x.column(feature_idx));
            }
            let predictions = estimator.predict(&x_features)?;
            if predictions.len() != n_samples {
                return Err(SklearsError::ShapeMismatch {
                    expected: format!("{} predictions", n_samples),
                    actual: format!("{} predictions", predictions.len()),
                });
            }
            for i in 0..n_samples {
                all_predictions[[i, estimator_idx]] = predictions[i] as Float;
            }
        }
        let mut final_predictions = Array1::zeros(n_samples);
        let mut confidence_intervals = Array2::zeros((n_samples, 2));
        for i in 0..n_samples {
            let sample_predictions = all_predictions.row(i);
            let mut class_counts = vec![0; n_classes];
            for &pred in sample_predictions {
                let class_idx = classes
                    .iter()
                    .position(|&c| c == pred as Int)
                    .unwrap_or(0);
                class_counts[class_idx] += 1;
            }
            let max_class_idx = class_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            final_predictions[i] = classes[max_class_idx];
            let mut sorted_predictions: Vec<Float> = sample_predictions.to_vec();
            sorted_predictions
                .sort_by(|a, b| a.partial_cmp(b).expect("operation should succeed"));
            let alpha = 1.0 - self.config.confidence_level;
            let lower_idx = ((alpha / 2.0) * n_estimators as Float) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n_estimators as Float) as usize;
            confidence_intervals[[i, 0]] = sorted_predictions[lower_idx
                .min(n_estimators - 1)];
            confidence_intervals[[i, 1]] = sorted_predictions[upper_idx
                .min(n_estimators - 1)];
        }
        Ok((final_predictions, confidence_intervals))
    }
}
impl Predict<Array2<Float>, Array1<Int>> for BaggingClassifier<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Int>> {
        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }
        let estimators = self.estimators();
        let estimators_features = self.estimators_features();
        let classes = self.classes();
        let n_classes = self.n_classes();
        let mut class_votes = Array2::zeros((n_samples, n_classes));
        for (estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            let mut x_features = Array2::zeros((n_samples, features.len()));
            for (j, &feature_idx) in features.iter().enumerate() {
                x_features.column_mut(j).assign(&x.column(feature_idx));
            }
            let predictions = estimator.predict(&x_features)?;
            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                    class_votes[[i, class_idx]] += 1.0;
                }
            }
        }
        let mut final_predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let max_idx = class_votes
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b): &(_, &Float)| {
                    a.partial_cmp(b).expect("operation should succeed")
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            final_predictions[i] = classes[max_idx];
        }
        Ok(final_predictions)
    }
}
impl Default for BaggingClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
impl<State> Estimator<State> for BaggingClassifier<State> {
    type Config = BaggingConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn validate_config(&self) -> Result<()> {
        if self.config.n_estimators == 0 {
            return Err(
                SklearsError::InvalidInput(
                    "n_estimators must be greater than 0".to_string(),
                ),
            );
        }
        if let Some(max_samples) = self.config.max_samples {
            if max_samples == 0 {
                return Err(
                    SklearsError::InvalidInput(
                        "max_samples must be greater than 0".to_string(),
                    ),
                );
            }
        }
        if let Some(max_features) = self.config.max_features {
            if max_features == 0 {
                return Err(
                    SklearsError::InvalidInput(
                        "max_features must be greater than 0".to_string(),
                    ),
                );
            }
        }
        if self.config.min_samples_split < 2 {
            return Err(
                SklearsError::InvalidInput(
                    "min_samples_split must be at least 2".to_string(),
                ),
            );
        }
        if self.config.min_samples_leaf < 1 {
            return Err(
                SklearsError::InvalidInput(
                    "min_samples_leaf must be at least 1".to_string(),
                ),
            );
        }
        if self.config.confidence_level <= 0.0 || self.config.confidence_level >= 1.0 {
            return Err(
                SklearsError::InvalidInput(
                    "confidence_level must be between 0.0 and 1.0".to_string(),
                ),
            );
        }
        Ok(())
    }
    fn metadata(&self) -> sklears_core::traits::EstimatorMetadata {
        sklears_core::traits::EstimatorMetadata {
            name: "BaggingClassifier".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Bootstrap aggregating (bagging) classifier".to_string(),
            supports_sparse: false,
            supports_multiclass: true,
            supports_multilabel: false,
            requires_positive_input: false,
            supports_online_learning: false,
            supports_feature_importance: true,
            memory_complexity: sklears_core::traits::MemoryComplexity::Linear,
            time_complexity: sklears_core::traits::TimeComplexity::LogLinear,
        }
    }
}
/// Enhanced Bagging regressor with OOB estimation and feature bagging
#[allow(dead_code)]
pub struct BaggingRegressor<State = Untrained> {
    config: BaggingConfig,
    state: PhantomData<State>,
    estimators_: Option<Vec<DecisionTreeRegressor<Trained>>>,
    estimators_features_: Option<Vec<Vec<usize>>>,
    estimators_samples_: Option<Vec<Vec<usize>>>,
    oob_score_: Option<Float>,
    n_features_in_: Option<usize>,
    feature_importances_: Option<Array1<Float>>,
}
impl BaggingRegressor<Untrained> {
    /// Create a new bagging regressor
    pub fn new() -> Self {
        Self {
            config: BaggingConfig::default(),
            state: PhantomData,
            estimators_: None,
            estimators_features_: None,
            estimators_samples_: None,
            oob_score_: None,
            n_features_in_: None,
            feature_importances_: None,
        }
    }
    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }
    /// Set the random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.config.random_state = Some(random_state);
        self
    }
    /// Set whether to calculate out-of-bag score
    pub fn oob_score(mut self, oob_score: bool) -> Self {
        self.config.oob_score = oob_score;
        self
    }
    /// Set the maximum number of samples per estimator
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }
    /// Set the maximum number of features per estimator
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.config.max_features = max_features;
        self
    }
    /// Set whether to use bootstrap sampling
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
        self
    }
    /// Set whether to use bootstrap feature sampling
    pub fn bootstrap_features(mut self, bootstrap_features: bool) -> Self {
        self.config.bootstrap_features = bootstrap_features;
        self
    }
    /// Set maximum depth for base estimators
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }
    /// Set minimum samples to split
    pub fn min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.config.min_samples_split = min_samples_split;
        self
    }
    /// Set minimum samples at leaf
    pub fn min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.config.min_samples_leaf = min_samples_leaf;
        self
    }
    /// Set confidence level for bootstrap intervals
    pub fn confidence_level(mut self, confidence_level: Float) -> Self {
        self.config.confidence_level = confidence_level;
        self
    }
    /// Set number of parallel jobs for training (None for automatic detection)
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }
    /// Enable parallel training with automatic job detection
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }
    /// Enable extra randomization (Extremely Randomized Trees)
    pub fn extra_randomized(mut self, extra_randomized: bool) -> Self {
        self.config.extra_randomized = extra_randomized;
        self
    }
    /// Enable extra randomization (convenient shorthand)
    pub fn extremely_randomized(mut self) -> Self {
        self.config.extra_randomized = true;
        self.config.bootstrap = false;
        self
    }
    /// Generate bootstrap samples with optional out-of-bag tracking
    /// For extra randomized trees, returns the full dataset instead of bootstrap samples
    fn bootstrap_sample(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        rng: &mut StdRng,
    ) -> Result<(Array2<Float>, Array1<Float>, Vec<usize>)> {
        let n_samples = x.nrows();
        if self.config.extra_randomized {
            let sample_indices: Vec<usize> = (0..n_samples).collect();
            return Ok((x.clone(), y.clone(), sample_indices));
        }
        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        let sample_indices: Vec<usize> = if self.config.bootstrap {
            (0..max_samples).map(|_| rng.gen_range(0..n_samples)).collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(max_samples);
            indices
        };
        let mut x_bootstrap = Array2::zeros((max_samples, x.ncols()));
        let mut y_bootstrap = Array1::zeros(max_samples);
        for (i, &idx) in sample_indices.iter().enumerate() {
            x_bootstrap.row_mut(i).assign(&x.row(idx));
            y_bootstrap[i] = y[idx];
        }
        Ok((x_bootstrap, y_bootstrap, sample_indices))
    }
    /// Train ensemble using parallel processing with work-stealing when available
    fn train_ensemble_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        _rng: &mut StdRng,
        n_features: usize,
    ) -> Result<RegressorEnsembleResult> {
        let mut bootstrap_data = Vec::new();
        for i in 0..self.config.n_estimators {
            let mut local_rng = StdRng::seed_from_u64(
                self.config.random_state.unwrap_or(42) + i as u64,
            );
            let (x_bootstrap, y_bootstrap, sample_indices) = self
                .bootstrap_sample(x, y, &mut local_rng)?;
            let feature_indices = self.get_feature_indices(n_features, &mut local_rng);
            bootstrap_data
                .push((x_bootstrap, y_bootstrap, sample_indices, feature_indices));
        }
        let use_parallel = self.should_use_parallel();
        if use_parallel {
            #[cfg(feature = "parallel")]
            {
                let results: Result<Vec<_>> = bootstrap_data
                    .into_par_iter()
                    .enumerate()
                    .map(|
                        (i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices))|
                    {
                        self.fit_single_estimator(
                                &x_bootstrap,
                                &y_bootstrap,
                                &feature_indices,
                                i,
                            )
                            .map(|estimator| (
                                estimator,
                                feature_indices,
                                sample_indices,
                            ))
                    })
                    .collect();
                let fitted_data = results?;
                let (estimators, estimators_features, estimators_samples) = fitted_data
                    .into_iter()
                    .fold(
                        (Vec::new(), Vec::new(), Vec::new()),
                        |(mut e, mut ef, mut es), (estimator, features, samples)| {
                            e.push(estimator);
                            ef.push(features);
                            es.push(samples);
                            (e, ef, es)
                        },
                    );
                Ok((estimators, estimators_features, estimators_samples))
            }
            #[cfg(not(feature = "parallel"))]
            { self.train_ensemble_sequential(bootstrap_data) }
        } else {
            self.train_ensemble_sequential(bootstrap_data)
        }
    }
    /// Train ensemble sequentially
    fn train_ensemble_sequential(
        &self,
        bootstrap_data: Vec<RegressorBootstrapSampleData>,
    ) -> Result<RegressorEnsembleResult> {
        let mut estimators = Vec::new();
        let mut estimators_features = Vec::new();
        let mut estimators_samples = Vec::new();
        for (i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices)) in bootstrap_data
            .into_iter()
            .enumerate()
        {
            let fitted_tree = self
                .fit_single_estimator(&x_bootstrap, &y_bootstrap, &feature_indices, i)?;
            estimators.push(fitted_tree);
            estimators_features.push(feature_indices);
            estimators_samples.push(sample_indices);
        }
        Ok((estimators, estimators_features, estimators_samples))
    }
    /// Fit a single estimator with given bootstrap sample and feature indices
    fn fit_single_estimator(
        &self,
        x_bootstrap: &Array2<Float>,
        y_bootstrap: &Array1<Float>,
        feature_indices: &[usize],
        estimator_index: usize,
    ) -> Result<DecisionTreeRegressor<Trained>> {
        let mut x_features = Array2::zeros((x_bootstrap.nrows(), feature_indices.len()));
        for (j, &feature_idx) in feature_indices.iter().enumerate() {
            x_features.column_mut(j).assign(&x_bootstrap.column(feature_idx));
        }
        let mut tree = DecisionTreeRegressor::new()
            .min_samples_split(self.config.min_samples_split)
            .min_samples_leaf(self.config.min_samples_leaf);
        if let Some(max_depth) = self.config.max_depth {
            tree = tree.max_depth(max_depth);
        }
        if let Some(seed) = self.config.random_state.map(|s| s + estimator_index as u64)
        {
            tree = tree.random_state(Some(seed));
        }
        tree.fit(&x_features, y_bootstrap)
    }
    /// Determine whether to use parallel processing based on configuration
    fn should_use_parallel(&self) -> bool {
        match self.config.n_jobs {
            Some(n) if n != 1 => true,
            None => false,
            _ => false,
        }
    }
    /// Generate feature indices for feature bagging
    fn get_feature_indices(&self, n_features: usize, rng: &mut StdRng) -> Vec<usize> {
        let max_features = self.config.max_features.unwrap_or(n_features);
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        if self.config.bootstrap_features {
            feature_indices = (0..max_features)
                .map(|_| rng.gen_range(0..n_features))
                .collect();
        } else {
            feature_indices.shuffle(rng);
            feature_indices.truncate(max_features);
        }
        feature_indices.sort_unstable();
        feature_indices
    }
    /// Calculate out-of-bag predictions and return the OOB R^2 score
    fn calculate_oob_predictions(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        estimators: &[DecisionTreeRegressor<Trained>],
        estimators_features: &[Vec<usize>],
        estimators_samples: &[Vec<usize>],
    ) -> Result<Float> {
        let n_samples = x.nrows();
        let mut oob_predictions: Array1<Float> = Array1::zeros(n_samples);
        let mut oob_counts: Array1<Float> = Array1::zeros(n_samples);
        for (estimator, (features, samples)) in estimators
            .iter()
            .zip(estimators_features.iter().zip(estimators_samples.iter()))
        {
            let mut oob_mask = vec![true; n_samples];
            for &sample_idx in samples {
                if sample_idx < n_samples {
                    oob_mask[sample_idx] = false;
                }
            }
            for (sample_idx, &is_oob) in oob_mask.iter().enumerate() {
                if is_oob {
                    let x_sample = x.row(sample_idx);
                    let x_features = Array2::from_shape_vec(
                            (1, features.len()),
                            features.iter().map(|&f| x_sample[f]).collect(),
                        )
                        .map_err(|_| {
                            SklearsError::InvalidInput(
                                "Failed to create feature subset".to_string(),
                            )
                        })?;
                    let pred = estimator.predict(&x_features)?;
                    oob_predictions[sample_idx] += pred[0];
                    oob_counts[sample_idx] += 1.0;
                }
            }
        }
        let mut y_oob_actual = Vec::new();
        let mut y_oob_pred = Vec::new();
        for i in 0..n_samples {
            if oob_counts[i] > 0.0 {
                y_oob_actual.push(y[i]);
                y_oob_pred.push(oob_predictions[i] / oob_counts[i]);
            }
        }
        if y_oob_actual.is_empty() {
            return Ok(0.0);
        }
        let mean_y = y_oob_actual.iter().sum::<Float>() / y_oob_actual.len() as Float;
        let ss_tot: Float = y_oob_actual.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        let ss_res: Float = y_oob_actual
            .iter()
            .zip(y_oob_pred.iter())
            .map(|(&yi, &pi)| (yi - pi).powi(2))
            .sum();
        if ss_tot.abs() < 1e-12 {
            Ok(if ss_res.abs() < 1e-12 { 1.0 } else { 0.0 })
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}
impl Fit<Array2<Float>, Array1<Float>> for BaggingRegressor<Untrained> {
    type Fitted = BaggingRegressor<Trained>;
    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        if n_samples != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("X.shape[0] = {}", n_samples),
                actual: format!("y.shape[0] = {}", y.len()),
            });
        }
        if n_samples == 0 {
            return Err(
                SklearsError::InvalidInput(
                    "Cannot fit bagging on empty dataset".to_string(),
                ),
            );
        }
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };
        let (estimators, estimators_features, estimators_samples) = self
            .train_ensemble_parallel(x, y, &mut rng, n_features)?;
        let oob_score = if self.config.oob_score {
            Some(
                self
                    .calculate_oob_predictions(
                        x,
                        y,
                        &estimators,
                        &estimators_features,
                        &estimators_samples,
                    )?,
            )
        } else {
            None
        };
        let mut feature_importances = Array1::zeros(n_features);
        for (_estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            let tree_importance = 1.0 / features.len() as Float;
            for &feature_idx in features {
                feature_importances[feature_idx] += tree_importance;
            }
        }
        let total_importance = feature_importances.sum();
        if total_importance > 0.0 {
            feature_importances /= total_importance;
        }
        Ok(BaggingRegressor {
            config: self.config,
            state: PhantomData,
            estimators_: Some(estimators),
            estimators_features_: Some(estimators_features),
            estimators_samples_: Some(estimators_samples),
            oob_score_: oob_score,
            n_features_in_: Some(n_features),
            feature_importances_: Some(feature_importances),
        })
    }
}
impl BaggingRegressor<Trained> {
    /// Get the fitted base estimators
    pub fn estimators(&self) -> &[DecisionTreeRegressor<Trained>] {
        self.estimators_.as_ref().expect("BaggingRegressor should be fitted")
    }
    /// Get the feature indices used by each estimator
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        self.estimators_features_.as_ref().expect("BaggingRegressor should be fitted")
    }
    /// Get the sample indices used by each estimator
    pub fn estimators_samples(&self) -> &[Vec<usize>] {
        self.estimators_samples_.as_ref().expect("BaggingRegressor should be fitted")
    }
    /// Get the out-of-bag score if calculated
    pub fn oob_score(&self) -> Option<Float> {
        self.oob_score_
    }
    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_.expect("BaggingRegressor should be fitted")
    }
    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_.as_ref().expect("BaggingRegressor should be fitted")
    }
}
impl Predict<Array2<Float>, Array1<Float>> for BaggingRegressor<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = x.dim();
        if n_features != self.n_features_in() {
            return Err(SklearsError::FeatureMismatch {
                expected: self.n_features_in(),
                actual: n_features,
            });
        }
        let estimators = self.estimators();
        let estimators_features = self.estimators_features();
        let n_estimators = estimators.len();
        let mut sum_predictions: Array1<Float> = Array1::zeros(n_samples);
        for (estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            let mut x_features = Array2::zeros((n_samples, features.len()));
            for (j, &feature_idx) in features.iter().enumerate() {
                x_features.column_mut(j).assign(&x.column(feature_idx));
            }
            let predictions = estimator.predict(&x_features)?;
            sum_predictions += &predictions;
        }
        Ok(sum_predictions / n_estimators as Float)
    }
}
impl Default for BaggingRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}
impl<State> Estimator<State> for BaggingRegressor<State> {
    type Config = BaggingConfig;
    type Error = SklearsError;
    type Float = Float;
    fn config(&self) -> &Self::Config {
        &self.config
    }
    fn validate_config(&self) -> Result<()> {
        if self.config.n_estimators == 0 {
            return Err(
                SklearsError::InvalidInput(
                    "n_estimators must be greater than 0".to_string(),
                ),
            );
        }
        if let Some(max_samples) = self.config.max_samples {
            if max_samples == 0 {
                return Err(
                    SklearsError::InvalidInput(
                        "max_samples must be greater than 0".to_string(),
                    ),
                );
            }
        }
        if let Some(max_features) = self.config.max_features {
            if max_features == 0 {
                return Err(
                    SklearsError::InvalidInput(
                        "max_features must be greater than 0".to_string(),
                    ),
                );
            }
        }
        if self.config.min_samples_split < 2 {
            return Err(
                SklearsError::InvalidInput(
                    "min_samples_split must be at least 2".to_string(),
                ),
            );
        }
        if self.config.min_samples_leaf < 1 {
            return Err(
                SklearsError::InvalidInput(
                    "min_samples_leaf must be at least 1".to_string(),
                ),
            );
        }
        if self.config.confidence_level <= 0.0 || self.config.confidence_level >= 1.0 {
            return Err(
                SklearsError::InvalidInput(
                    "confidence_level must be between 0.0 and 1.0".to_string(),
                ),
            );
        }
        Ok(())
    }
    fn metadata(&self) -> sklears_core::traits::EstimatorMetadata {
        sklears_core::traits::EstimatorMetadata {
            name: "BaggingRegressor".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            description: "Bootstrap aggregating (bagging) regressor".to_string(),
            supports_sparse: false,
            supports_multiclass: false,
            supports_multilabel: false,
            requires_positive_input: false,
            supports_online_learning: false,
            supports_feature_importance: true,
            memory_complexity: sklears_core::traits::MemoryComplexity::Linear,
            time_complexity: sklears_core::traits::TimeComplexity::LogLinear,
        }
    }
}

#[cfg(test)]
mod tests;
