//! Ensemble dummy estimators that combine multiple baseline strategies
//!
//! This module provides ensemble dummy estimators that combine predictions from multiple
//! simple baseline strategies to create more robust and diverse baseline models.

use crate::dummy_classifier::{DummyClassifier, Strategy as ClassifierStrategy};
use crate::dummy_regressor::{DummyRegressor, Strategy as RegressorStrategy};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{prelude::*, Rng};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict, PredictProba};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Ensemble strategy for combining multiple dummy estimators
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum EnsembleStrategy {
    /// Simple average of all estimator predictions
    Average,
    /// Weighted average based on estimated performance
    WeightedAverage,
    /// Majority voting for classification
    MajorityVoting,
    /// Select the best performing strategy on a validation split
    BestStrategy,
    /// Stacking approach using a meta-learner (another dummy estimator)
    Stacking(Box<RegressorStrategy>), // Meta-learner strategy
    /// Random selection among strategies for each prediction
    RandomSelection,
    /// Adaptive selection based on input characteristics
    AdaptiveSelection,
}

/// Ensemble dummy classifier combining multiple classification strategies
#[derive(Debug, Clone)]
pub struct EnsembleDummyClassifier<State = sklears_core::traits::Untrained> {
    /// List of strategies to combine
    pub strategies: Vec<ClassifierStrategy>,
    /// Ensemble combination strategy
    pub ensemble_strategy: EnsembleStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,
    /// Validation split ratio for strategy selection
    pub validation_split: Float,
    /// Fitted base estimators
    pub(crate) base_estimators_: Option<Vec<DummyClassifier<sklears_core::traits::Trained>>>,
    /// Strategy weights for weighted averaging
    pub(crate) strategy_weights_: Option<Array1<Float>>,
    /// Best strategy index for BestStrategy approach
    pub(crate) best_strategy_index_: Option<usize>,
    /// Meta-learner for stacking
    pub(crate) meta_learner_: Option<DummyRegressor<sklears_core::traits::Trained>>,
    /// Classes for classification
    pub(crate) classes_: Option<Array1<Int>>,
    /// Number of classes
    pub(crate) n_classes_: Option<usize>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

/// Ensemble dummy regressor combining multiple regression strategies
#[derive(Debug, Clone)]
pub struct EnsembleDummyRegressor<State = sklears_core::traits::Untrained> {
    /// List of strategies to combine
    pub strategies: Vec<RegressorStrategy>,
    /// Ensemble combination strategy
    pub ensemble_strategy: EnsembleStrategy,
    /// Random state for reproducible output
    pub random_state: Option<u64>,
    /// Validation split ratio for strategy selection
    pub validation_split: Float,
    /// Fitted base estimators
    pub(crate) base_estimators_: Option<Vec<DummyRegressor<sklears_core::traits::Trained>>>,
    /// Strategy weights for weighted averaging
    pub(crate) strategy_weights_: Option<Array1<Float>>,
    /// Best strategy index for BestStrategy approach
    pub(crate) best_strategy_index_: Option<usize>,
    /// Meta-learner for stacking
    pub(crate) meta_learner_: Option<DummyRegressor<sklears_core::traits::Trained>>,
    /// Phantom data for state
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl EnsembleDummyClassifier {
    /// Create a new ensemble dummy classifier
    pub fn new(strategies: Vec<ClassifierStrategy>, ensemble_strategy: EnsembleStrategy) -> Self {
        Self {
            strategies,
            ensemble_strategy,
            random_state: None,
            validation_split: 0.2,
            base_estimators_: None,
            strategy_weights_: None,
            best_strategy_index_: None,
            meta_learner_: None,
            classes_: None,
            n_classes_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the validation split ratio
    pub fn with_validation_split(mut self, validation_split: Float) -> Self {
        self.validation_split = validation_split;
        self
    }

    /// Get the number of strategies
    pub fn n_strategies(&self) -> usize {
        self.strategies.len()
    }
}

impl EnsembleDummyRegressor {
    /// Create a new ensemble dummy regressor
    pub fn new(strategies: Vec<RegressorStrategy>, ensemble_strategy: EnsembleStrategy) -> Self {
        Self {
            strategies,
            ensemble_strategy,
            random_state: None,
            validation_split: 0.2,
            base_estimators_: None,
            strategy_weights_: None,
            best_strategy_index_: None,
            meta_learner_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the validation split ratio
    pub fn with_validation_split(mut self, validation_split: Float) -> Self {
        self.validation_split = validation_split;
        self
    }

    /// Get the number of strategies
    pub fn n_strategies(&self) -> usize {
        self.strategies.len()
    }
}

impl Default for EnsembleDummyClassifier {
    fn default() -> Self {
        Self::new(
            vec![
                ClassifierStrategy::MostFrequent,
                ClassifierStrategy::Stratified,
                ClassifierStrategy::Uniform,
            ],
            EnsembleStrategy::Average,
        )
    }
}

impl Default for EnsembleDummyRegressor {
    fn default() -> Self {
        Self::new(
            vec![
                RegressorStrategy::Mean,
                RegressorStrategy::Median,
                RegressorStrategy::Normal {
                    mean: None,
                    std: None,
                },
            ],
            EnsembleStrategy::Average,
        )
    }
}

impl Estimator for EnsembleDummyClassifier {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Estimator for EnsembleDummyRegressor {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for EnsembleDummyClassifier {
    type Fitted = EnsembleDummyClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                "X has {} samples but y has {} samples",
                x.nrows(),
                y.len()
            )));
        }

        // Get unique classes
        let mut unique_classes = y.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        let classes = Array1::from_vec(unique_classes);
        let n_classes = classes.len();

        // Create train/validation split for strategy evaluation
        let n_samples = x.nrows();
        let n_val = (n_samples as Float * self.validation_split).round() as usize;
        let n_train = n_samples - n_val;

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Create random indices for train/validation split
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Manual shuffle implementation
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        // Fit all base estimators
        let mut base_estimators = Vec::new();
        let mut val_scores = Vec::new();

        for strategy in &self.strategies {
            let estimator = DummyClassifier::new(strategy.clone())
                .with_random_state(self.random_state.unwrap_or(42));

            // Create training subset
            let x_train = x.select(Axis(0), train_indices);
            let y_train = Array1::from_iter(train_indices.iter().map(|&i| y[i]));

            // Fit on training data
            let fitted = estimator.fit(&x_train, &y_train)?;

            // Evaluate on validation data if available
            let val_score = if !val_indices.is_empty() {
                let x_val = x.select(Axis(0), val_indices);
                let y_val = Array1::from_iter(val_indices.iter().map(|&i| y[i]));
                let predictions = fitted.predict(&x_val)?;

                // Calculate accuracy
                let correct = predictions
                    .iter()
                    .zip(y_val.iter())
                    .filter(|(&pred, &true_val)| pred == true_val)
                    .count();
                correct as Float / val_indices.len() as Float
            } else {
                0.5 // Default score if no validation data
            };

            val_scores.push(val_score);
            base_estimators.push(fitted);
        }

        // Now fit on full dataset for final models
        let mut final_estimators = Vec::new();
        for strategy in &self.strategies {
            let estimator = DummyClassifier::new(strategy.clone())
                .with_random_state(self.random_state.unwrap_or(42));
            let fitted = estimator.fit(x, y)?;
            final_estimators.push(fitted);
        }

        // Calculate strategy weights and best strategy based on ensemble strategy
        let (strategy_weights, best_strategy_index, meta_learner) = match &self.ensemble_strategy {
            EnsembleStrategy::Average => (None, None, None),
            EnsembleStrategy::WeightedAverage => {
                // Weight by validation performance
                let total_score: Float = val_scores.iter().sum();
                let weights = if total_score > 0.0 {
                    Array1::from_iter(val_scores.iter().map(|&score| score / total_score))
                } else {
                    Array1::from_elem(self.strategies.len(), 1.0 / self.strategies.len() as Float)
                };
                (Some(weights), None, None)
            }
            EnsembleStrategy::MajorityVoting => (None, None, None),
            EnsembleStrategy::BestStrategy => {
                let best_idx = val_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _): (usize, &Float)| idx);
                (None, best_idx, None)
            }
            EnsembleStrategy::Stacking(meta_strategy) => {
                // Create meta-features from base estimator predictions
                if !val_indices.is_empty() {
                    let x_val = x.select(Axis(0), val_indices);
                    let y_val = Array1::from_iter(val_indices.iter().map(|&i| y[i] as Float));

                    let mut meta_features =
                        Array2::zeros((val_indices.len(), self.strategies.len()));
                    for (i, estimator) in base_estimators.iter().enumerate() {
                        let predictions = estimator.predict(&x_val)?;
                        for (j, &pred) in predictions.iter().enumerate() {
                            meta_features[[j, i]] = pred as Float;
                        }
                    }

                    // Fit meta-learner
                    let meta_learner_estimator = DummyRegressor::new(*meta_strategy.as_ref())
                        .with_random_state(self.random_state.unwrap_or(42));
                    let meta_fitted = meta_learner_estimator.fit(&meta_features, &y_val)?;
                    (None, None, Some(meta_fitted))
                } else {
                    (None, None, None)
                }
            }
            EnsembleStrategy::RandomSelection => (None, None, None),
            EnsembleStrategy::AdaptiveSelection => {
                // Simple adaptive: weight by diversity and accuracy
                let diversity_bonus = 0.1;
                let weights =
                    Array1::from_iter(val_scores.iter().enumerate().map(|(i, &score)| {
                        score + diversity_bonus * (i as Float / self.strategies.len() as Float)
                    }));
                let total: Float = weights.sum();
                let normalized_weights = if total > 0.0 {
                    weights / total
                } else {
                    Array1::from_elem(self.strategies.len(), 1.0 / self.strategies.len() as Float)
                };
                (Some(normalized_weights), None, None)
            }
        };

        Ok(EnsembleDummyClassifier {
            strategies: self.strategies,
            ensemble_strategy: self.ensemble_strategy,
            random_state: self.random_state,
            validation_split: self.validation_split,
            base_estimators_: Some(final_estimators),
            strategy_weights_: strategy_weights,
            best_strategy_index_: best_strategy_index,
            meta_learner_: meta_learner,
            classes_: Some(classes),
            n_classes_: Some(n_classes),
            _state: std::marker::PhantomData,
        })
    }
}

impl Fit<Features, Array1<Float>> for EnsembleDummyRegressor {
    type Fitted = EnsembleDummyRegressor<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Float>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(format!(
                "X has {} samples but y has {} samples",
                x.nrows(),
                y.len()
            )));
        }

        // Create train/validation split for strategy evaluation
        let n_samples = x.nrows();
        let n_val = (n_samples as Float * self.validation_split).round() as usize;
        let n_train = n_samples - n_val;

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(0)
        };

        // Create random indices for train/validation split
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Manual shuffle implementation
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let train_indices = &indices[..n_train];
        let val_indices = &indices[n_train..];

        // Fit all base estimators on training subset and evaluate on validation
        let mut base_estimators = Vec::new();
        let mut val_scores = Vec::new();

        for strategy in &self.strategies {
            let estimator =
                DummyRegressor::new(*strategy).with_random_state(self.random_state.unwrap_or(42));

            // Create training subset
            let x_train = x.select(Axis(0), train_indices);
            let y_train = Array1::from_iter(train_indices.iter().map(|&i| y[i]));

            // Fit on training data
            let fitted = estimator.fit(&x_train, &y_train)?;

            // Evaluate on validation data if available
            let val_score = if !val_indices.is_empty() {
                let x_val = x.select(Axis(0), val_indices);
                let y_val = Array1::from_iter(val_indices.iter().map(|&i| y[i]));
                let predictions = fitted.predict(&x_val)?;

                // Calculate negative MSE (higher is better)
                let mse = predictions
                    .iter()
                    .zip(y_val.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                    .sum::<Float>()
                    / val_indices.len() as Float;
                -mse
            } else {
                0.0 // Default score if no validation data
            };

            val_scores.push(val_score);
            base_estimators.push(fitted);
        }

        // Now fit on full dataset for final models
        let mut final_estimators = Vec::new();
        for strategy in &self.strategies {
            let estimator =
                DummyRegressor::new(*strategy).with_random_state(self.random_state.unwrap_or(42));
            let fitted = estimator.fit(x, y)?;
            final_estimators.push(fitted);
        }

        // Calculate strategy weights and best strategy based on ensemble strategy
        let (strategy_weights, best_strategy_index, meta_learner) = match &self.ensemble_strategy {
            EnsembleStrategy::Average => (None, None, None),
            EnsembleStrategy::WeightedAverage => {
                // Weight by validation performance (convert negative MSE to positive weights)
                let max_score = val_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let shifted_scores: Vec<Float> = val_scores
                    .iter()
                    .map(|&score| score - max_score + 1.0) // Shift to make positive
                    .collect();
                let total_score: Float = shifted_scores.iter().sum();
                let weights = if total_score > 0.0 {
                    Array1::from_iter(shifted_scores.iter().map(|&score| score / total_score))
                } else {
                    Array1::from_elem(self.strategies.len(), 1.0 / self.strategies.len() as Float)
                };
                (Some(weights), None, None)
            }
            EnsembleStrategy::BestStrategy => {
                let best_idx = val_scores
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .map(|(idx, _): (usize, &Float)| idx);
                (None, best_idx, None)
            }
            EnsembleStrategy::Stacking(meta_strategy) => {
                // Create meta-features from base estimator predictions
                if !val_indices.is_empty() {
                    let x_val = x.select(Axis(0), val_indices);
                    let y_val = Array1::from_iter(val_indices.iter().map(|&i| y[i]));

                    let mut meta_features =
                        Array2::zeros((val_indices.len(), self.strategies.len()));
                    for (i, estimator) in base_estimators.iter().enumerate() {
                        let predictions = estimator.predict(&x_val)?;
                        for (j, &pred) in predictions.iter().enumerate() {
                            meta_features[[j, i]] = pred;
                        }
                    }

                    // Fit meta-learner
                    let meta_learner_estimator = DummyRegressor::new(*meta_strategy.as_ref())
                        .with_random_state(self.random_state.unwrap_or(42));
                    let meta_fitted = meta_learner_estimator.fit(&meta_features, &y_val)?;
                    (None, None, Some(meta_fitted))
                } else {
                    (None, None, None)
                }
            }
            EnsembleStrategy::RandomSelection => (None, None, None),
            EnsembleStrategy::AdaptiveSelection => {
                // Simple adaptive: weight by performance and diversity
                let diversity_bonus = 0.1;
                let max_score = val_scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let weights =
                    Array1::from_iter(val_scores.iter().enumerate().map(|(i, &score)| {
                        let normalized_score = score - max_score + 1.0;
                        normalized_score
                            + diversity_bonus * (i as Float / self.strategies.len() as Float)
                    }));
                let total: Float = weights.sum();
                let normalized_weights = if total > 0.0 {
                    weights / total
                } else {
                    Array1::from_elem(self.strategies.len(), 1.0 / self.strategies.len() as Float)
                };
                (Some(normalized_weights), None, None)
            }
            _ => (None, None, None),
        };

        Ok(EnsembleDummyRegressor {
            strategies: self.strategies,
            ensemble_strategy: self.ensemble_strategy,
            random_state: self.random_state,
            validation_split: self.validation_split,
            base_estimators_: Some(final_estimators),
            strategy_weights_: strategy_weights,
            best_strategy_index_: best_strategy_index,
            meta_learner_: meta_learner,
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>> for EnsembleDummyClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        let base_estimators = self.base_estimators_.as_ref().unwrap();
        let n_samples = x.nrows();

        match &self.ensemble_strategy {
            EnsembleStrategy::BestStrategy => {
                let best_idx = self.best_strategy_index_.unwrap();
                base_estimators[best_idx].predict(x)
            }
            EnsembleStrategy::RandomSelection => {
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                let mut predictions = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let strategy_idx = rng.gen_range(0..base_estimators.len());
                    let single_sample = x.slice(scirs2_core::ndarray::s![i..i + 1, ..]).to_owned();
                    let pred = base_estimators[strategy_idx].predict(&single_sample)?;
                    predictions[i] = pred[0];
                }
                Ok(predictions)
            }
            EnsembleStrategy::MajorityVoting => {
                // Get predictions from all estimators
                let mut all_predictions = Vec::new();
                for estimator in base_estimators {
                    let preds = estimator.predict(x)?;
                    all_predictions.push(preds);
                }

                // Perform majority voting
                let mut final_predictions = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let mut vote_counts: HashMap<Int, usize> = HashMap::new();
                    for preds in &all_predictions {
                        *vote_counts.entry(preds[i]).or_insert(0) += 1;
                    }
                    final_predictions[i] = *vote_counts
                        .iter()
                        .max_by_key(|(_, &count)| count)
                        .map(|(class, _)| class)
                        .unwrap();
                }
                Ok(final_predictions)
            }
            EnsembleStrategy::Stacking(_) => {
                if let Some(meta_learner) = &self.meta_learner_ {
                    // Get predictions from all base estimators
                    let mut meta_features = Array2::zeros((n_samples, base_estimators.len()));
                    for (i, estimator) in base_estimators.iter().enumerate() {
                        let preds = estimator.predict(x)?;
                        for (j, &pred) in preds.iter().enumerate() {
                            meta_features[[j, i]] = pred as Float;
                        }
                    }

                    // Get meta-learner predictions and convert to integers
                    let meta_preds = meta_learner.predict(&meta_features)?;
                    Ok(meta_preds.mapv(|x| x.round() as Int))
                } else {
                    // Fallback to average if meta-learner not available
                    self.predict_average(x)
                }
            }
            _ => {
                // Average, WeightedAverage, AdaptiveSelection
                self.predict_average(x)
            }
        }
    }
}

impl EnsembleDummyClassifier<sklears_core::traits::Trained> {
    fn predict_average(&self, x: &Features) -> Result<Array1<Int>> {
        let base_estimators = self.base_estimators_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = self.n_classes_.unwrap();
        let classes = self.classes_.as_ref().unwrap();

        // Get probability predictions and average them
        let mut avg_probas = Array2::zeros((n_samples, n_classes));

        let weights = self.strategy_weights_.as_ref();

        for (i, estimator) in base_estimators.iter().enumerate() {
            let probas = estimator.predict_proba(x)?;
            let weight = weights
                .map(|w| w[i])
                .unwrap_or(1.0 / base_estimators.len() as Float);

            for j in 0..n_samples {
                for k in 0..n_classes {
                    avg_probas[[j, k]] += weight * probas[[j, k]];
                }
            }
        }

        // Convert averaged probabilities to class predictions
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let class_idx = avg_probas
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a): &(usize, &Float), (_, b): &(usize, &Float)| {
                    a.partial_cmp(b).unwrap()
                })
                .map(|(idx, _)| idx)
                .unwrap();
            predictions[i] = classes[class_idx];
        }

        Ok(predictions)
    }
}

impl Predict<Features, Array1<Float>> for EnsembleDummyRegressor<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Float>> {
        let base_estimators = self.base_estimators_.as_ref().unwrap();
        let n_samples = x.nrows();

        match &self.ensemble_strategy {
            EnsembleStrategy::BestStrategy => {
                let best_idx = self.best_strategy_index_.unwrap();
                base_estimators[best_idx].predict(x)
            }
            EnsembleStrategy::RandomSelection => {
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                let mut predictions = Array1::zeros(n_samples);
                for i in 0..n_samples {
                    let strategy_idx = rng.gen_range(0..base_estimators.len());
                    let single_sample = x.slice(scirs2_core::ndarray::s![i..i + 1, ..]).to_owned();
                    let pred = base_estimators[strategy_idx].predict(&single_sample)?;
                    predictions[i] = pred[0];
                }
                Ok(predictions)
            }
            EnsembleStrategy::Stacking(_) => {
                if let Some(meta_learner) = &self.meta_learner_ {
                    // Get predictions from all base estimators
                    let mut meta_features = Array2::zeros((n_samples, base_estimators.len()));
                    for (i, estimator) in base_estimators.iter().enumerate() {
                        let preds = estimator.predict(x)?;
                        for (j, &pred) in preds.iter().enumerate() {
                            meta_features[[j, i]] = pred;
                        }
                    }

                    // Get meta-learner predictions
                    meta_learner.predict(&meta_features)
                } else {
                    // Fallback to average if meta-learner not available
                    self.predict_average(x)
                }
            }
            _ => {
                // Average, WeightedAverage, AdaptiveSelection
                self.predict_average(x)
            }
        }
    }
}

impl EnsembleDummyRegressor<sklears_core::traits::Trained> {
    fn predict_average(&self, x: &Features) -> Result<Array1<Float>> {
        let base_estimators = self.base_estimators_.as_ref().unwrap();
        let n_samples = x.nrows();

        let mut predictions = Array1::zeros(n_samples);
        let weights = self.strategy_weights_.as_ref();

        for (i, estimator) in base_estimators.iter().enumerate() {
            let preds = estimator.predict(x)?;
            let weight = weights
                .map(|w| w[i])
                .unwrap_or(1.0 / base_estimators.len() as Float);

            for j in 0..n_samples {
                predictions[j] += weight * preds[j];
            }
        }

        Ok(predictions)
    }

    /// Get strategy weights used in ensemble
    pub fn strategy_weights(&self) -> Option<&Array1<Float>> {
        self.strategy_weights_.as_ref()
    }

    /// Get the index of the best strategy (for BestStrategy ensemble)
    pub fn best_strategy_index(&self) -> Option<usize> {
        self.best_strategy_index_
    }
}

impl PredictProba<Features, Array2<Float>>
    for EnsembleDummyClassifier<sklears_core::traits::Trained>
{
    fn predict_proba(&self, x: &Features) -> Result<Array2<Float>> {
        let base_estimators = self.base_estimators_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = self.n_classes_.unwrap();

        match &self.ensemble_strategy {
            EnsembleStrategy::BestStrategy => {
                let best_idx = self.best_strategy_index_.unwrap();
                base_estimators[best_idx].predict_proba(x)
            }
            _ => {
                // Average probabilities from all estimators
                let mut avg_probas = Array2::zeros((n_samples, n_classes));
                let weights = self.strategy_weights_.as_ref();

                for (i, estimator) in base_estimators.iter().enumerate() {
                    let probas = estimator.predict_proba(x)?;
                    let weight = weights
                        .map(|w| w[i])
                        .unwrap_or(1.0 / base_estimators.len() as Float);

                    for j in 0..n_samples {
                        for k in 0..n_classes {
                            avg_probas[[j, k]] += weight * probas[[j, k]];
                        }
                    }
                }

                Ok(avg_probas)
            }
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ensemble_dummy_classifier() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 0, 1];

        let strategies = vec![
            ClassifierStrategy::MostFrequent,
            ClassifierStrategy::Stratified,
            ClassifierStrategy::Uniform,
        ];

        let ensemble = EnsembleDummyClassifier::new(strategies, EnsembleStrategy::Average)
            .with_random_state(42);
        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_ensemble_dummy_regressor() {
        let x = Array2::from_shape_vec(
            (5, 2),
            vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let strategies = vec![
            RegressorStrategy::Mean,
            RegressorStrategy::Median,
            RegressorStrategy::Normal {
                mean: None,
                std: None,
            },
        ];

        let ensemble = EnsembleDummyRegressor::new(strategies, EnsembleStrategy::Average)
            .with_random_state(42);
        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 5);
        for &pred in predictions.iter() {
            assert!(pred.is_finite());
        }
    }

    #[test]
    fn test_ensemble_weighted_average() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![RegressorStrategy::Mean, RegressorStrategy::Median];

        let ensemble = EnsembleDummyRegressor::new(strategies, EnsembleStrategy::WeightedAverage)
            .with_random_state(42)
            .with_validation_split(0.5);
        let fitted = ensemble.fit(&x, &y).unwrap();

        // Check that weights were computed
        assert!(fitted.strategy_weights().is_some());

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_ensemble_best_strategy() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let strategies = vec![
            RegressorStrategy::Mean,
            RegressorStrategy::Median,
            RegressorStrategy::Normal {
                mean: None,
                std: None,
            },
        ];

        let ensemble = EnsembleDummyRegressor::new(strategies, EnsembleStrategy::BestStrategy)
            .with_random_state(42)
            .with_validation_split(0.3);
        let fitted = ensemble.fit(&x, &y).unwrap();

        // Check that best strategy was selected
        assert!(fitted.best_strategy_index().is_some());

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_ensemble_stacking() {
        let x = Array2::from_shape_vec((8, 2), (0..16).map(|x| x as f64).collect()).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let strategies = vec![RegressorStrategy::Mean, RegressorStrategy::Median];

        let ensemble = EnsembleDummyRegressor::new(
            strategies,
            EnsembleStrategy::Stacking(Box::new(RegressorStrategy::Mean)),
        )
        .with_random_state(42)
        .with_validation_split(0.5);

        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 8);
    }

    #[test]
    fn test_ensemble_majority_voting() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 0, 1];

        let strategies = vec![
            ClassifierStrategy::MostFrequent,
            ClassifierStrategy::Stratified,
        ];

        let ensemble = EnsembleDummyClassifier::new(strategies, EnsembleStrategy::MajorityVoting)
            .with_random_state(42);
        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
    }

    #[test]
    fn test_ensemble_random_selection() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![RegressorStrategy::Mean, RegressorStrategy::Median];

        let ensemble = EnsembleDummyRegressor::new(strategies, EnsembleStrategy::RandomSelection)
            .with_random_state(42);
        let fitted = ensemble.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_ensemble_predict_proba() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1];

        let strategies = vec![
            ClassifierStrategy::MostFrequent,
            ClassifierStrategy::Stratified,
        ];

        let ensemble = EnsembleDummyClassifier::new(strategies, EnsembleStrategy::Average)
            .with_random_state(42);
        let fitted = ensemble.fit(&x, &y).unwrap();
        let probas = fitted.predict_proba(&x).unwrap();

        assert_eq!(probas.shape(), &[4, 2]); // 4 samples, 2 classes

        // Check probabilities sum to 1
        for i in 0..4 {
            let row_sum: Float = probas.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }
}
