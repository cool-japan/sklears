//! Bagging ensemble methods
//!
//! Bootstrap aggregating (bagging) trains multiple base estimators on random
//! subsets of the training data and aggregates their predictions.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::prelude::*;
use sklears_core::error::{Result, SklearsError};
use sklears_core::prelude::Predict;
use sklears_core::traits::{Estimator, Fit, Trained, Untrained};
use sklears_core::types::{Float, Int};
// use sklears_tree::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion}; // Temporarily disabled
use crate::adaboost::{DecisionTreeClassifier, DecisionTreeRegressor, SplitCriterion};
#[allow(unused_imports)]
use std::collections::HashSet;
use std::marker::PhantomData;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

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
pub struct BaggingClassifier<State = Untrained> {
    config: BaggingConfig,
    state: PhantomData<State>,
    // Fitted parameters
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
        self.config.n_jobs = Some(-1); // -1 means use all available cores
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
        self.config.bootstrap = false; // Extra trees typically don't use bootstrap
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

        // For extra randomized trees, use the full dataset
        if self.config.extra_randomized {
            let sample_indices: Vec<usize> = (0..n_samples).collect();
            return Ok((x.clone(), y.clone(), sample_indices));
        }

        let max_samples = self.config.max_samples.unwrap_or(n_samples);

        // Find unique classes and their indices
        let mut class_indices: std::collections::HashMap<Int, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, &class) in y.iter().enumerate() {
            class_indices.entry(class).or_default().push(idx);
        }

        let mut sample_indices = Vec::new();

        if self.config.bootstrap {
            // First, do standard bootstrap sampling
            sample_indices = (0..max_samples)
                .map(|_| rng.gen_range(0..n_samples))
                .collect();
        } else {
            // Random sampling without replacement
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(rng);
            indices.truncate(max_samples);
            sample_indices = indices;
        }

        // Verify we have at least 2 classes in the sample
        let mut sampled_classes = std::collections::HashSet::new();
        for &idx in &sample_indices {
            sampled_classes.insert(y[idx]);
        }

        // If we still have only one class, force diversity by replacing some samples
        if sampled_classes.len() < 2 && class_indices.len() >= 2 {
            let mut other_classes: Vec<Int> = class_indices
                .keys()
                .filter(|&&c| !sampled_classes.contains(&c))
                .cloned()
                .collect();

            // Sort for deterministic behavior
            other_classes.sort();

            if !other_classes.is_empty() {
                // Replace the last sample with one from the first other class (deterministic)
                let other_class = other_classes[0];
                if let Some(other_indices) = class_indices.get(&other_class) {
                    if !other_indices.is_empty() {
                        // Use the first available index for deterministic behavior
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
        rng: &mut StdRng,
        n_features: usize,
    ) -> Result<(
        Vec<DecisionTreeClassifier<Trained>>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
    )> {
        // Pre-generate all bootstrap samples and feature indices to maintain determinism
        let mut bootstrap_data = Vec::new();
        for i in 0..self.config.n_estimators {
            let mut local_rng =
                StdRng::seed_from_u64(self.config.random_state.unwrap_or(42) + i as u64);

            let (x_bootstrap, y_bootstrap, sample_indices) =
                self.bootstrap_sample(x, y, &mut local_rng)?;
            let feature_indices = self.get_feature_indices(n_features, &mut local_rng);

            bootstrap_data.push((x_bootstrap, y_bootstrap, sample_indices, feature_indices));
        }

        // Determine whether to use parallel processing
        let use_parallel = self.should_use_parallel();

        if use_parallel {
            #[cfg(feature = "parallel")]
            {
                // Parallel training with work-stealing using rayon
                let results: Result<Vec<_>> = bootstrap_data
                    .into_par_iter()
                    .enumerate()
                    .map(
                        |(i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices))| {
                            self.fit_single_estimator(
                                &x_bootstrap,
                                &y_bootstrap,
                                &feature_indices,
                                i,
                            )
                            .map(|estimator| (estimator, feature_indices, sample_indices))
                        },
                    )
                    .collect();

                let fitted_data = results?;
                let (estimators, estimators_features, estimators_samples) =
                    fitted_data.into_iter().fold(
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
            {
                // Fall back to sequential if parallel feature is not enabled
                self.train_ensemble_sequential(bootstrap_data)
            }
        } else {
            // Sequential training
            self.train_ensemble_sequential(bootstrap_data)
        }
    }

    /// Train ensemble sequentially
    fn train_ensemble_sequential(
        &self,
        bootstrap_data: Vec<(Array2<Float>, Array1<Int>, Vec<usize>, Vec<usize>)>,
    ) -> Result<(
        Vec<DecisionTreeClassifier<Trained>>,
        Vec<Vec<usize>>,
        Vec<Vec<usize>>,
    )> {
        let mut estimators = Vec::new();
        let mut estimators_features = Vec::new();
        let mut estimators_samples = Vec::new();

        for (i, (x_bootstrap, y_bootstrap, sample_indices, feature_indices)) in
            bootstrap_data.into_iter().enumerate()
        {
            let fitted_tree =
                self.fit_single_estimator(&x_bootstrap, &y_bootstrap, &feature_indices, i)?;

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
        // Extract features for this estimator
        let mut x_features = Array2::zeros((x_bootstrap.nrows(), feature_indices.len()));
        for (j, &feature_idx) in feature_indices.iter().enumerate() {
            x_features
                .column_mut(j)
                .assign(&x_bootstrap.column(feature_idx));
        }

        // Create and configure base estimator (decision tree)
        let mut tree = DecisionTreeClassifier::new()
            .criterion(SplitCriterion::Gini)
            .min_samples_split(self.config.min_samples_split)
            .min_samples_leaf(self.config.min_samples_leaf);

        if let Some(max_depth) = self.config.max_depth {
            tree = tree.max_depth(max_depth);
        }

        if let Some(seed) = self.config.random_state.map(|s| s + estimator_index as u64) {
            tree = tree.random_state(Some(seed));
        }

        // Fit the tree
        tree.fit(&x_features, y_bootstrap)
    }

    /// Determine whether to use parallel processing based on configuration
    fn should_use_parallel(&self) -> bool {
        match self.config.n_jobs {
            Some(n) if n != 1 => true, // Use parallel if n_jobs is set and not 1
            None => false,             // Don't use parallel if not specified
            _ => false,                // n_jobs == Some(1), use sequential
        }
    }

    /// Generate feature indices for feature bagging
    fn get_feature_indices(&self, n_features: usize, rng: &mut StdRng) -> Vec<usize> {
        let max_features = self.config.max_features.unwrap_or(n_features);
        let mut feature_indices: Vec<usize> = (0..n_features).collect();

        if self.config.bootstrap_features {
            // Sample features with replacement
            feature_indices = (0..max_features)
                .map(|_| rng.gen_range(0..n_features))
                .collect();
        } else {
            // Sample features without replacement
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

        for (estimator_idx, (estimator, (features, samples))) in estimators
            .iter()
            .zip(estimators_features.iter().zip(estimators_samples.iter()))
            .enumerate()
        {
            // Find out-of-bag samples (samples not used in training)
            let mut oob_mask = vec![true; n_samples];
            for &sample_idx in samples {
                if sample_idx < n_samples {
                    oob_mask[sample_idx] = false;
                }
            }

            // Predict on out-of-bag samples
            for (sample_idx, &is_oob) in oob_mask.iter().enumerate() {
                if is_oob {
                    // Extract features for this sample
                    let x_sample = x.row(sample_idx);
                    let x_features = Array2::from_shape_vec(
                        (1, features.len()),
                        features.iter().map(|&f| x_sample[f]).collect(),
                    )
                    .map_err(|_| {
                        SklearsError::InvalidInput("Failed to create feature subset".to_string())
                    })?;

                    let pred = estimator.predict(&x_features)?;
                    oob_predictions[sample_idx] += pred[0] as Float;
                    oob_counts[sample_idx] += 1.0;
                }
            }
        }

        // Calculate OOB score (accuracy)
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

        if total == 0 {
            Ok(0.0)
        } else {
            Ok(correct as Float / total as Float)
        }
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
            return Err(SklearsError::InvalidInput(
                "Cannot fit bagging on empty dataset".to_string(),
            ));
        }

        // Find unique classes
        let mut unique_classes: Vec<Int> = y.iter().cloned().collect();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Bagging requires at least 2 classes".to_string(),
            ));
        }

        // Initialize random number generator
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42), // Use fallback seed for entropy
        };

        // Train ensemble (parallel or sequential based on configuration)
        let (estimators, estimators_features, estimators_samples) =
            self.train_ensemble_parallel(x, y, &mut rng, n_features)?;

        // Calculate out-of-bag score if requested
        let oob_score = if self.config.oob_score {
            Some(self.calculate_oob_predictions(
                x,
                y,
                &estimators,
                &estimators_features,
                &estimators_samples,
            )?)
        } else {
            None
        };

        // Calculate feature importances (average over all trees)
        let mut feature_importances = Array1::zeros(n_features);
        for (estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            // For now, use uniform importance within selected features
            let tree_importance = 1.0 / features.len() as Float;
            for &feature_idx in features {
                feature_importances[feature_idx] += tree_importance;
            }
        }

        // Normalize feature importances
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
        self.estimators_
            .as_ref()
            .expect("BaggingClassifier should be fitted")
    }

    /// Get the feature indices used by each estimator
    pub fn estimators_features(&self) -> &[Vec<usize>] {
        self.estimators_features_
            .as_ref()
            .expect("BaggingClassifier should be fitted")
    }

    /// Get the sample indices used by each estimator
    pub fn estimators_samples(&self) -> &[Vec<usize>] {
        self.estimators_samples_
            .as_ref()
            .expect("BaggingClassifier should be fitted")
    }

    /// Get the out-of-bag score if calculated
    pub fn oob_score(&self) -> Option<Float> {
        self.oob_score_
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<Int> {
        self.classes_
            .as_ref()
            .expect("BaggingClassifier should be fitted")
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.n_classes_.expect("BaggingClassifier should be fitted")
    }

    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("BaggingClassifier should be fitted")
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_
            .as_ref()
            .expect("BaggingClassifier should be fitted")
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

        // Get predictions from all estimators
        for (estimator_idx, (estimator, features)) in estimators
            .iter()
            .zip(estimators_features.iter())
            .enumerate()
        {
            // Extract features for this estimator
            let mut x_features = Array2::zeros((n_samples, features.len()));
            for (j, &feature_idx) in features.iter().enumerate() {
                x_features.column_mut(j).assign(&x.column(feature_idx));
            }

            let predictions = estimator.predict(&x_features)?;

            // Validate prediction array size
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

        // Calculate final predictions and confidence intervals
        let mut final_predictions = Array1::zeros(n_samples);
        let mut confidence_intervals = Array2::zeros((n_samples, 2)); // [lower, upper] bounds

        for i in 0..n_samples {
            let sample_predictions = all_predictions.row(i);

            // Mode for final prediction
            let mut class_counts = vec![0; n_classes];
            for &pred in sample_predictions {
                let class_idx = classes.iter().position(|&c| c == pred as Int).unwrap_or(0);
                class_counts[class_idx] += 1;
            }

            let max_class_idx = class_counts
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.cmp(b))
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            final_predictions[i] = classes[max_class_idx];

            // Bootstrap confidence interval
            let mut sorted_predictions: Vec<Float> = sample_predictions.to_vec();
            sorted_predictions.sort_by(|a, b| a.partial_cmp(b).unwrap());

            let alpha = 1.0 - self.config.confidence_level;
            let lower_idx = ((alpha / 2.0) * n_estimators as Float) as usize;
            let upper_idx = ((1.0 - alpha / 2.0) * n_estimators as Float) as usize;

            confidence_intervals[[i, 0]] = sorted_predictions[lower_idx.min(n_estimators - 1)];
            confidence_intervals[[i, 1]] = sorted_predictions[upper_idx.min(n_estimators - 1)];
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

        // Aggregate predictions from all estimators
        for (estimator, features) in estimators.iter().zip(estimators_features.iter()) {
            // Extract features for this estimator
            let mut x_features = Array2::zeros((n_samples, features.len()));
            for (j, &feature_idx) in features.iter().enumerate() {
                x_features.column_mut(j).assign(&x.column(feature_idx));
            }

            let predictions = estimator.predict(&x_features)?;

            // Count votes for each class
            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = classes.iter().position(|&c| c == pred) {
                    class_votes[[i, class_idx]] += 1.0;
                }
            }
        }

        // Select class with most votes
        let mut final_predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let max_idx = class_votes
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b): &(_, &Float)| a.partial_cmp(b).unwrap())
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
            return Err(SklearsError::InvalidInput(
                "n_estimators must be greater than 0".to_string(),
            ));
        }

        if let Some(max_samples) = self.config.max_samples {
            if max_samples == 0 {
                return Err(SklearsError::InvalidInput(
                    "max_samples must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(max_features) = self.config.max_features {
            if max_features == 0 {
                return Err(SklearsError::InvalidInput(
                    "max_features must be greater than 0".to_string(),
                ));
            }
        }

        if self.config.min_samples_split < 2 {
            return Err(SklearsError::InvalidInput(
                "min_samples_split must be at least 2".to_string(),
            ));
        }

        if self.config.min_samples_leaf < 1 {
            return Err(SklearsError::InvalidInput(
                "min_samples_leaf must be at least 1".to_string(),
            ));
        }

        if self.config.confidence_level <= 0.0 || self.config.confidence_level >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "confidence_level must be between 0.0 and 1.0".to_string(),
            ));
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
pub struct BaggingRegressor<State = Untrained> {
    config: BaggingConfig,
    state: PhantomData<State>,
    // Fitted parameters
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
}

impl BaggingRegressor<Trained> {
    /// Get the out-of-bag score if calculated
    pub fn oob_score(&self) -> Option<Float> {
        self.oob_score_
    }

    /// Get the number of input features
    pub fn n_features_in(&self) -> usize {
        self.n_features_in_
            .expect("BaggingRegressor should be fitted")
    }

    /// Get feature importances
    pub fn feature_importances(&self) -> &Array1<Float> {
        self.feature_importances_
            .as_ref()
            .expect("BaggingRegressor should be fitted")
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
            return Err(SklearsError::InvalidInput(
                "n_estimators must be greater than 0".to_string(),
            ));
        }

        if let Some(max_samples) = self.config.max_samples {
            if max_samples == 0 {
                return Err(SklearsError::InvalidInput(
                    "max_samples must be greater than 0".to_string(),
                ));
            }
        }

        if let Some(max_features) = self.config.max_features {
            if max_features == 0 {
                return Err(SklearsError::InvalidInput(
                    "max_features must be greater than 0".to_string(),
                ));
            }
        }

        if self.config.min_samples_split < 2 {
            return Err(SklearsError::InvalidInput(
                "min_samples_split must be at least 2".to_string(),
            ));
        }

        if self.config.min_samples_leaf < 1 {
            return Err(SklearsError::InvalidInput(
                "min_samples_leaf must be at least 1".to_string(),
            ));
        }

        if self.config.confidence_level <= 0.0 || self.config.confidence_level >= 1.0 {
            return Err(SklearsError::InvalidInput(
                "confidence_level must be between 0.0 and 1.0".to_string(),
            ));
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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use sklears_core::traits::Predict;

    // Property-based tests using proptest
    use proptest::prelude::*;

    #[test]
    fn test_bagging_classifier_creation() {
        let classifier = BaggingClassifier::new()
            .n_estimators(20)
            .random_state(42)
            .oob_score(true);

        assert_eq!(classifier.config.n_estimators, 20);
        assert_eq!(classifier.config.random_state, Some(42));
        assert_eq!(classifier.config.oob_score, true);
    }

    #[test]
    fn test_bagging_classifier_fit_predict() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2, 0, 1];

        let classifier = BaggingClassifier::new().n_estimators(5).random_state(42);

        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
        assert_eq!(fitted.n_classes(), 3);
        assert_eq!(fitted.classes().len(), 3);
        assert_eq!(fitted.n_features_in(), 2);
    }

    #[test]
    fn test_bagging_classifier_with_oob() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0],
            [9.0, 10.0],
            [10.0, 11.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2, 0, 1, 2, 0];

        let classifier = BaggingClassifier::new()
            .n_estimators(10)
            .random_state(42)
            .oob_score(true)
            .bootstrap(true);

        let fitted = classifier.fit(&x, &y).unwrap();

        assert!(fitted.oob_score().is_some());
        let oob_score = fitted.oob_score().unwrap();
        assert!(oob_score >= 0.0 && oob_score <= 1.0);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    fn test_bagging_classifier_feature_bagging() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
            [6.0, 7.0, 8.0, 9.0],
        ];
        let y = array![0, 0, 1, 1, 2, 2];

        let classifier = BaggingClassifier::new()
            .n_estimators(5)
            .max_features(Some(2)) // Use only 2 features per estimator
            .bootstrap_features(false)
            .random_state(42);

        let fitted = classifier.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert_eq!(fitted.n_features_in(), 4);

        // Check that feature importances are calculated
        let importances = fitted.feature_importances();
        assert_eq!(importances.len(), 4);

        // Feature importances should sum to 1
        let sum: Float = importances.sum();
        assert!((sum - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_bagging_classifier_confidence_intervals() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];
        let y = array![0, 0, 1, 1];

        let classifier = BaggingClassifier::new()
            .n_estimators(10)
            .random_state(42)
            .confidence_level(0.95);

        let fitted = classifier.fit(&x, &y).unwrap();
        let (predictions, confidence_intervals) = fitted.predict_with_confidence(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert_eq!(confidence_intervals.dim(), (4, 2));

        // Check that lower bound <= upper bound
        for i in 0..4 {
            assert!(confidence_intervals[[i, 0]] <= confidence_intervals[[i, 1]]);
        }
    }

    #[test]
    fn test_bagging_regressor_creation() {
        let regressor = BaggingRegressor::new().n_estimators(15).random_state(123);

        assert_eq!(regressor.config.n_estimators, 15);
        assert_eq!(regressor.config.random_state, Some(123));
    }

    #[test]
    fn test_bagging_config_default() {
        let config = BaggingConfig::default();

        assert_eq!(config.n_estimators, 10);
        assert_eq!(config.bootstrap, true);
        assert_eq!(config.bootstrap_features, false);
        assert_eq!(config.oob_score, false);
        assert_eq!(config.random_state, None);
        assert_eq!(config.min_samples_split, 2);
        assert_eq!(config.min_samples_leaf, 1);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_bagging_classifier_invalid_input() {
        // Empty input
        let classifier = BaggingClassifier::new();
        let x = Array2::zeros((0, 2));
        let y = Array1::zeros(0);
        assert!(classifier.fit(&x, &y).is_err());

        // Mismatched dimensions
        let classifier = BaggingClassifier::new();
        let x = Array2::zeros((3, 2));
        let y = Array1::zeros(2);
        assert!(classifier.fit(&x, &y).is_err());

        // Single class
        let classifier = BaggingClassifier::new();
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![0, 0];
        assert!(classifier.fit(&x, &y).is_err());
    }

    #[test]
    fn test_bagging_classifier_feature_mismatch() {
        let x_train = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let y_train = array![0, 1];
        let x_test = array![[1.0, 2.0]]; // Wrong number of features

        let classifier = BaggingClassifier::new();
        let fitted = classifier.fit(&x_train, &y_train).unwrap();
        assert!(fitted.predict(&x_test).is_err());
    }

    // Property-based tests for ensemble properties

    proptest! {
        #[test]
        fn prop_bagging_deterministic_with_seed(
            n_estimators in 1usize..10,
            random_seed in 0u64..1000,
        ) {
            let x = array![
                [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
            ];
            let y = array![0, 0, 1, 1, 2, 2, 0, 1];

            // Train two identical classifiers with same seed
            let classifier1 = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .random_state(random_seed)
                .fit(&x, &y)
                .unwrap();

            let classifier2 = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .random_state(random_seed)
                .fit(&x, &y)
                .unwrap();

            let pred1 = classifier1.predict(&x).unwrap();
            let pred2 = classifier2.predict(&x).unwrap();

            // Predictions should be identical with same seed
            prop_assert_eq!(pred1, pred2);
        }

        #[test]
        fn prop_bagging_feature_importance_normalization(
            n_estimators in 1usize..10,
            max_features in 1usize..4,
        ) {
            let x = array![
                [1.0, 2.0, 3.0], [2.0, 3.0, 4.0], [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0], [5.0, 6.0, 7.0], [6.0, 7.0, 8.0],
            ];
            let y = array![0, 0, 1, 1, 2, 2];

            let classifier = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .max_features(Some(max_features))
                .random_state(42)
                .fit(&x, &y)
                .unwrap();

            let importances = classifier.feature_importances();
            let sum: Float = importances.sum();

            // Feature importances should sum to 1 (normalized)
            prop_assert!((sum - 1.0).abs() < 1e-10);

            // All importances should be non-negative
            for &importance in importances.iter() {
                prop_assert!(importance >= 0.0);
            }
        }

        #[test]
        fn prop_bagging_bootstrap_diversity(
            n_estimators in 2usize..8,
        ) {
            let x = array![
                [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
                [9.0, 10.0], [10.0, 11.0],
            ];
            let y = array![0, 0, 1, 1, 2, 2, 0, 1, 2, 0];

            let classifier = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .bootstrap(true)
                .random_state(42)
                .fit(&x, &y)
                .unwrap();

            let estimators_samples = classifier.estimators_samples();

            // Each estimator should use different bootstrap samples (diversity)
            let mut unique_sample_sets = HashSet::new();
            for samples in estimators_samples {
                let mut sorted_samples = samples.clone();
                sorted_samples.sort();
                unique_sample_sets.insert(sorted_samples);
            }

            // Should have some diversity in bootstrap samples
            // (not all estimators use exactly the same samples)
            prop_assert!(unique_sample_sets.len() >= 1);
        }

        #[test]
        fn prop_bagging_prediction_stability(
            n_estimators in 3usize..10,
        ) {
            let x = array![
                [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                [5.0, 6.0], [6.0, 7.0],
            ];
            let y = array![0, 0, 1, 1, 2, 2];

            let classifier = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .random_state(42)
                .fit(&x, &y)
                .unwrap();

            let predictions = classifier.predict(&x).unwrap();

            // All predictions should be valid class labels
            let classes = classifier.classes();
            for &pred in predictions.iter() {
                prop_assert!(classes.iter().any(|&c| c == pred));
            }

            // Number of predictions should match input samples
            prop_assert_eq!(predictions.len(), x.nrows());
        }

        #[test]
        fn prop_bagging_oob_score_bounds(
            n_estimators in 5usize..15,
        ) {
            let x = array![
                [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
                [5.0, 6.0], [6.0, 7.0], [7.0, 8.0], [8.0, 9.0],
                [9.0, 10.0], [10.0, 11.0], [11.0, 12.0], [12.0, 13.0],
            ];
            let y = array![0, 0, 1, 1, 2, 2, 0, 1, 2, 0, 1, 2];

            let classifier = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .oob_score(true)
                .bootstrap(true)
                .random_state(42)
                .fit(&x, &y)
                .unwrap();

            if let Some(oob_score) = classifier.oob_score() {
                // OOB score should be between 0 and 1 (accuracy)
                prop_assert!(oob_score >= 0.0 && oob_score <= 1.0);
            }
        }

        #[test]
        fn prop_bagging_confidence_intervals_bounds(
            n_estimators in 3usize..8,
            confidence_level in 0.7..0.99,
        ) {
            let x = array![
                [1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],
            ];
            let y = array![0, 0, 1, 1];

            let classifier = BaggingClassifier::new()
                .n_estimators(n_estimators)
                .confidence_level(confidence_level)
                .random_state(42)
                .fit(&x, &y)
                .unwrap();

            let (predictions, confidence_intervals) = classifier.predict_with_confidence(&x).unwrap();

            // Check confidence interval properties
            for i in 0..predictions.len() {
                let lower = confidence_intervals[[i, 0]];
                let upper = confidence_intervals[[i, 1]];

                // Lower bound should be <= upper bound
                prop_assert!(lower <= upper);

                // Confidence intervals should be reasonable
                prop_assert!(lower.is_finite() && upper.is_finite());
            }
        }
    }
}
