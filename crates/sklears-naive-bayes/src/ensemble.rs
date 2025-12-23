//! Ensemble methods for Naive Bayes classifiers
//!
//! This module provides ensemble methods including bagging, voting,
//! and model averaging for Naive Bayes classifiers.

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
// SciRS2 Policy Compliance - Use scirs2-core for random functionality
use rayon::prelude::*;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict, PredictProba, Trained},
};
use std::collections::HashMap;

use crate::{BernoulliNB, GaussianNB, MultinomialNB};

/// Voting strategy for ensemble methods
#[derive(Debug, Clone)]
pub enum VotingStrategy {
    /// Hard voting (majority vote)
    Hard,
    /// Soft voting (average probabilities)
    Soft,
}

/// Bootstrap sampling strategy for bagging
#[derive(Debug, Clone)]
pub struct BootstrapConfig {
    /// Whether to sample with replacement
    pub with_replacement: bool,
    /// Fraction of samples to use for each bootstrap
    pub sample_fraction: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for BootstrapConfig {
    fn default() -> Self {
        Self {
            with_replacement: true,
            sample_fraction: 1.0,
            random_state: None,
        }
    }
}

/// Bagged Naive Bayes ensemble
#[derive(Debug)]
pub struct BaggedNaiveBayes<BaseClassifier = GaussianNB<Trained>>
where
    BaseClassifier: Clone,
{
    /// Base classifier type
    pub base_classifier: BaseClassifier,
    /// Number of estimators in the ensemble
    pub n_estimators: usize,
    /// Bootstrap configuration
    pub bootstrap_config: BootstrapConfig,
    /// Trained base classifiers
    estimators_: Option<Vec<BaseClassifier>>,
    /// Classes seen during training
    classes_: Option<Array1<i32>>,
}

impl<BaseClassifier> BaggedNaiveBayes<BaseClassifier>
where
    BaseClassifier: Clone,
{
    pub fn new(base_classifier: BaseClassifier) -> Self {
        Self {
            base_classifier,
            n_estimators: 10,
            bootstrap_config: BootstrapConfig::default(),
            estimators_: None,
            classes_: None,
        }
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn bootstrap_config(mut self, config: BootstrapConfig) -> Self {
        self.bootstrap_config = config;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.bootstrap_config.random_state = Some(random_state);
        self
    }

    /// Generate bootstrap sample indices
    fn bootstrap_sample(
        &self,
        n_samples: usize,
        rng: &mut scirs2_core::random::CoreRandom,
    ) -> Vec<usize> {
        let sample_size = (n_samples as f64 * self.bootstrap_config.sample_fraction) as usize;

        if self.bootstrap_config.with_replacement {
            (0..sample_size)
                .map(|_| rng.gen_range(0..n_samples))
                .collect()
        } else {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);
            indices
        }
    }

    /// Extract bootstrap sample
    fn extract_bootstrap_sample(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        indices: &[usize],
    ) -> (Array2<f64>, Array1<i32>) {
        let x_bootstrap = x.select(Axis(0), indices);
        let y_bootstrap = Array1::from_iter(indices.iter().map(|&i| y[i]));
        (x_bootstrap, y_bootstrap)
    }
}

/// Voting ensemble for Naive Bayes classifiers
pub struct VotingNaiveBayes {
    /// List of base classifiers
    pub estimators: Vec<Box<dyn NaiveBayesEstimator>>,
    /// Voting strategy
    pub voting: VotingStrategy,
    /// Weights for each estimator (optional)
    pub weights: Option<Array1<f64>>,
    /// Classes seen during training
    classes_: Option<Array1<i32>>,
}

/// Trait for Naive Bayes estimators that can be used in ensembles
pub trait NaiveBayesEstimator: Send + Sync {
    fn fit_clone(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Box<dyn NaiveBayesEstimator>>;
    fn ensemble_predict(&self, x: &Array2<f64>) -> Result<Array1<i32>>;
    fn ensemble_predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>>;
    fn ensemble_classes(&self) -> &Array1<i32>;
}

impl VotingNaiveBayes {
    pub fn new(voting: VotingStrategy) -> Self {
        Self {
            estimators: Vec::new(),
            voting,
            weights: None,
            classes_: None,
        }
    }

    pub fn add_estimator(&mut self, estimator: Box<dyn NaiveBayesEstimator>) {
        self.estimators.push(estimator);
    }

    pub fn weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(weights);
        self
    }

    /// Fit all estimators
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        if self.estimators.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No estimators provided".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        self.classes_ = Some(Array1::from_vec(classes));

        // Fit all estimators
        let fitted_estimators: Result<Vec<_>> = self
            .estimators
            .iter()
            .map(|estimator| estimator.fit_clone(x, y))
            .collect();

        self.estimators = fitted_estimators?;
        Ok(())
    }

    /// Predict using ensemble
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        match self.voting {
            VotingStrategy::Hard => self.predict_hard_voting(x),
            VotingStrategy::Soft => {
                let probabilities = self.predict_proba(x)?;
                Ok(probabilities.map_axis(Axis(1), |row| {
                    let max_idx = row
                        .iter()
                        .enumerate()
                        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                        .map(|(idx, _)| idx)
                        .unwrap_or(0);
                    classes[max_idx]
                }))
            }
        }
    }

    /// Hard voting prediction
    fn predict_hard_voting(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let classes = self.classes_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Collect predictions from all estimators
        let predictions: Result<Vec<_>> = self
            .estimators
            .iter()
            .map(|estimator| estimator.ensemble_predict(x))
            .collect();
        let predictions = predictions?;

        let mut final_predictions = Array1::zeros(n_samples);

        for sample_idx in 0..n_samples {
            let mut class_votes = HashMap::new();

            // Count votes for each class
            for (est_idx, pred) in predictions.iter().enumerate() {
                let weight = self.weights.as_ref().map(|w| w[est_idx]).unwrap_or(1.0);

                *class_votes.entry(pred[sample_idx]).or_insert(0.0) += weight;
            }

            // Find class with most votes
            let best_class = class_votes
                .iter()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(&class, _)| class)
                .unwrap_or(classes[0]);

            final_predictions[sample_idx] = best_class;
        }

        Ok(final_predictions)
    }

    /// Predict probabilities using soft voting
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if !matches!(self.voting, VotingStrategy::Soft) {
            return Err(SklearsError::InvalidOperation(
                "predict_proba requires soft voting".to_string(),
            ));
        }

        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Collect probabilities from all estimators
        let probabilities: Result<Vec<_>> = self
            .estimators
            .iter()
            .map(|estimator| estimator.ensemble_predict_proba(x))
            .collect();
        let probabilities = probabilities?;

        let mut ensemble_proba = Array2::zeros((n_samples, n_classes));
        let total_weight = self
            .weights
            .as_ref()
            .map(|w| w.sum())
            .unwrap_or(self.estimators.len() as f64);

        for (est_idx, proba) in probabilities.iter().enumerate() {
            let weight = self.weights.as_ref().map(|w| w[est_idx]).unwrap_or(1.0);

            ensemble_proba += &(proba * weight);
        }

        ensemble_proba /= total_weight;
        Ok(ensemble_proba)
    }
}

/// Model averaging ensemble
pub struct AveragingNaiveBayes {
    /// List of fitted estimators
    estimators: Vec<Box<dyn NaiveBayesEstimator>>,
    /// Weights for each estimator
    weights: Array1<f64>,
    /// Classes seen during training
    classes_: Option<Array1<i32>>,
}

impl AveragingNaiveBayes {
    pub fn new(estimators: Vec<Box<dyn NaiveBayesEstimator>>) -> Self {
        let n_estimators = estimators.len();
        let weights = Array1::from_elem(n_estimators, 1.0 / n_estimators as f64);

        Self {
            estimators,
            weights,
            classes_: None,
        }
    }

    pub fn with_weights(mut self, weights: Array1<f64>) -> Result<Self> {
        if weights.len() != self.estimators.len() {
            return Err(SklearsError::InvalidInput(
                "Number of weights must match number of estimators".to_string(),
            ));
        }

        // Normalize weights
        let weight_sum = weights.sum();
        self.weights = weights / weight_sum;
        Ok(self)
    }

    /// Fit all estimators
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        self.classes_ = Some(Array1::from_vec(classes));

        // Fit all estimators
        let fitted_estimators: Result<Vec<_>> = self
            .estimators
            .iter()
            .map(|estimator| estimator.fit_clone(x, y))
            .collect();

        self.estimators = fitted_estimators?;
        Ok(())
    }

    /// Predict using weighted average of probabilities
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(x)?;
        let classes = self.classes_.as_ref().unwrap();

        Ok(probabilities.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            classes[max_idx]
        }))
    }

    /// Predict probabilities using weighted average
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        let n_samples = x.nrows();
        let n_classes = classes.len();

        // Collect probabilities from all estimators
        let probabilities: Result<Vec<_>> = self
            .estimators
            .iter()
            .map(|estimator| estimator.ensemble_predict_proba(x))
            .collect();
        let probabilities = probabilities?;

        let mut ensemble_proba = Array2::zeros((n_samples, n_classes));

        for (est_idx, proba) in probabilities.iter().enumerate() {
            let weight = self.weights[est_idx];
            ensemble_proba += &(proba * weight);
        }

        Ok(ensemble_proba)
    }
}

/// Boosting strategy for Naive Bayes ensembles
#[derive(Debug, Clone)]
pub enum BoostingStrategy {
    /// AdaBoost with Naive Bayes as weak learners
    AdaBoost,
    /// LogitBoost adaptation for probabilistic classifiers
    LogitBoost,
    /// Gentle AdaBoost with smoother updates
    GentleAdaBoost,
}

/// Adaptive Boosting (AdaBoost) for Naive Bayes classifiers
pub struct AdaBoostNaiveBayes {
    /// Base classifier type
    pub base_classifier: String,
    /// Number of boosting iterations
    pub n_estimators: usize,
    /// Learning rate (shrinks contribution of each classifier)
    pub learning_rate: f64,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Boosting strategy
    pub strategy: BoostingStrategy,
    /// Trained weak learners
    estimators_: Option<Vec<Box<dyn NaiveBayesEstimator>>>,
    /// Weights for each weak learner
    estimator_weights_: Option<Array1<f64>>,
    /// Training errors for each weak learner
    estimator_errors_: Option<Array1<f64>>,
    /// Classes seen during training
    classes_: Option<Array1<i32>>,
}

impl AdaBoostNaiveBayes {
    pub fn new(base_classifier: &str) -> Self {
        Self {
            base_classifier: base_classifier.to_string(),
            n_estimators: 50,
            learning_rate: 1.0,
            random_state: None,
            strategy: BoostingStrategy::AdaBoost,
            estimators_: None,
            estimator_weights_: None,
            estimator_errors_: None,
            classes_: None,
        }
    }

    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    pub fn strategy(mut self, strategy: BoostingStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Create a base classifier instance
    fn create_base_classifier(&self) -> Result<Box<dyn NaiveBayesEstimator>> {
        match self.base_classifier.as_str() {
            "GaussianNB" => Ok(Box::new(GaussianNBEstimator::new())),
            "MultinomialNB" => Ok(Box::new(MultinomialNBEstimator::new())),
            "BernoulliNB" => Ok(Box::new(BernoulliNBEstimator::new())),
            _ => Err(SklearsError::InvalidInput(format!(
                "Unknown classifier: {}",
                self.base_classifier
            ))),
        }
    }

    /// Fit the AdaBoost ensemble
    pub fn fit(&mut self, x: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        let n_samples = x.nrows();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();

        if classes.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "AdaBoost currently supports only binary classification".to_string(),
            ));
        }

        self.classes_ = Some(Array1::from_vec(classes.clone()));

        // Initialize sample weights uniformly
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut estimator_errors = Vec::new();

        match self.strategy {
            BoostingStrategy::AdaBoost => {
                self.fit_adaboost(
                    x,
                    y,
                    &mut sample_weights,
                    &mut estimators,
                    &mut estimator_weights,
                    &mut estimator_errors,
                )?;
            }
            BoostingStrategy::LogitBoost => {
                self.fit_logitboost(
                    x,
                    y,
                    &mut sample_weights,
                    &mut estimators,
                    &mut estimator_weights,
                    &mut estimator_errors,
                )?;
            }
            BoostingStrategy::GentleAdaBoost => {
                self.fit_gentle_adaboost(
                    x,
                    y,
                    &mut sample_weights,
                    &mut estimators,
                    &mut estimator_weights,
                    &mut estimator_errors,
                )?;
            }
        }

        self.estimators_ = Some(estimators);
        self.estimator_weights_ = Some(Array1::from_vec(estimator_weights));
        self.estimator_errors_ = Some(Array1::from_vec(estimator_errors));

        Ok(())
    }

    /// Fit using standard AdaBoost algorithm
    fn fit_adaboost(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_weights: &mut Array1<f64>,
        estimators: &mut Vec<Box<dyn NaiveBayesEstimator>>,
        estimator_weights: &mut Vec<f64>,
        estimator_errors: &mut Vec<f64>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let classes = self.classes_.as_ref().unwrap();

        for iteration in 0..self.n_estimators {
            // Create and fit weak learner with weighted samples
            let mut base_estimator = self.create_base_classifier()?;

            // For Naive Bayes, we simulate weighted sampling by resampling
            let (x_weighted, y_weighted) = self.resample_with_weights(x, y, sample_weights)?;
            base_estimator = base_estimator.fit_clone(&x_weighted, &y_weighted)?;

            // Get predictions
            let predictions = base_estimator.ensemble_predict(x)?;

            // Calculate weighted error
            let mut weighted_error = 0.0;
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    weighted_error += sample_weights[i];
                }
            }

            // Avoid division by zero and perfect classifiers
            weighted_error = weighted_error.clamp(1e-10, 1.0 - 1e-10);

            // Calculate estimator weight (alpha)
            let alpha = self.learning_rate * 0.5 * ((1.0 - weighted_error) / weighted_error).ln();

            // Update sample weights
            for i in 0..n_samples {
                let prediction_correct = predictions[i] == y[i];
                let update_factor = if prediction_correct {
                    (-alpha).exp()
                } else {
                    alpha.exp()
                };
                sample_weights[i] *= update_factor;
            }

            // Normalize sample weights
            let weight_sum = sample_weights.sum();
            if weight_sum > 0.0 {
                *sample_weights /= weight_sum;
            }

            estimators.push(base_estimator);
            estimator_weights.push(alpha);
            estimator_errors.push(weighted_error);

            // Early stopping if error is too low or too high
            if weighted_error < 1e-10 {
                break;
            }
        }

        Ok(())
    }

    /// Fit using LogitBoost algorithm
    fn fit_logitboost(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_weights: &mut Array1<f64>,
        estimators: &mut Vec<Box<dyn NaiveBayesEstimator>>,
        estimator_weights: &mut Vec<f64>,
        estimator_errors: &mut Vec<f64>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let classes = self.classes_.as_ref().unwrap();

        // Convert labels to {-1, +1}
        let y_signed: Array1<f64> = y.mapv(|label| if label == classes[0] { -1.0 } else { 1.0 });

        // Initialize predictions to zero (log-odds)
        let mut f_values = Array1::zeros(n_samples);

        for iteration in 0..self.n_estimators {
            // Compute probabilities and working response
            let probabilities: Array1<f64> = f_values.mapv(|f: f64| 1.0 / (1.0 + (-f).exp()));
            let working_response: Array1<f64> = y_signed
                .iter()
                .zip(probabilities.iter())
                .map(|(&y_val, &p)| (y_val - 2.0 * p + 1.0) / (2.0 * p * (1.0 - p) + 1e-10))
                .collect::<Vec<_>>()
                .into();

            // Create weighted targets for Naive Bayes
            // This is a simplified approach - LogitBoost typically uses regression trees
            let binary_targets: Array1<i32> =
                working_response.mapv(|w| if w > 0.0 { 1 } else { 0 });

            let mut base_estimator = self.create_base_classifier()?;
            base_estimator = base_estimator.fit_clone(x, &binary_targets)?;

            let predictions = base_estimator.ensemble_predict(x)?;
            let prediction_values: Array1<f64> =
                predictions.mapv(|p| if p == 1 { 1.0 } else { -1.0 });

            // Update f_values (log-odds)
            f_values = &f_values + &(&prediction_values * self.learning_rate);

            estimators.push(base_estimator);
            estimator_weights.push(self.learning_rate);
            estimator_errors.push(0.0); // LogitBoost doesn't have traditional error
        }

        Ok(())
    }

    /// Fit using Gentle AdaBoost algorithm
    fn fit_gentle_adaboost(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_weights: &mut Array1<f64>,
        estimators: &mut Vec<Box<dyn NaiveBayesEstimator>>,
        estimator_weights: &mut Vec<f64>,
        estimator_errors: &mut Vec<f64>,
    ) -> Result<()> {
        let n_samples = x.nrows();
        let classes = self.classes_.as_ref().unwrap();

        // Convert labels to {-1, +1}
        let y_signed: Array1<f64> = y.mapv(|label| if label == classes[0] { -1.0 } else { 1.0 });

        for iteration in 0..self.n_estimators {
            // Create and fit weak learner
            let (x_weighted, y_weighted) = self.resample_with_weights(x, y, sample_weights)?;
            let mut base_estimator = self.create_base_classifier()?;
            base_estimator = base_estimator.fit_clone(&x_weighted, &y_weighted)?;

            // Get class probabilities
            let probabilities = base_estimator.ensemble_predict_proba(x)?;

            // For binary classification, get probability of positive class
            let positive_probs: Array1<f64> = if probabilities.ncols() == 2 {
                probabilities.column(1).to_owned()
            } else {
                // If only one column, assume it's the probability of the positive class
                probabilities.column(0).to_owned()
            };

            // Gentle update: h(x) = log(p / (1-p))
            let h_values: Array1<f64> = positive_probs.mapv(|p| {
                let p_safe = p.clamp(1e-10, 1.0 - 1e-10);
                (p_safe / (1.0 - p_safe)).ln()
            });

            // Update sample weights gently
            for i in 0..n_samples {
                sample_weights[i] *= (-y_signed[i] * h_values[i] * self.learning_rate).exp();
            }

            // Normalize sample weights
            let weight_sum = sample_weights.sum();
            if weight_sum > 0.0 {
                *sample_weights /= weight_sum;
            }

            estimators.push(base_estimator);
            estimator_weights.push(self.learning_rate);
            estimator_errors.push(0.0); // Gentle AdaBoost doesn't compute traditional error
        }

        Ok(())
    }

    /// Resample data according to sample weights
    fn resample_with_weights(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        weights: &Array1<f64>,
    ) -> Result<(Array2<f64>, Array1<i32>)> {
        let n_samples = x.nrows();
        let mut rng = match self.random_state {
            Some(seed) => scirs2_core::random::CoreRandom::seed_from_u64(seed),
            None => {
                scirs2_core::random::CoreRandom::from_rng(&mut scirs2_core::random::thread_rng())
            }
        };

        // Convert weights to cumulative distribution
        let mut cumulative_weights = weights.clone();
        for i in 1..n_samples {
            cumulative_weights[i] += cumulative_weights[i - 1];
        }

        // Sample with replacement according to weights
        let mut resampled_indices = Vec::new();
        for _ in 0..n_samples {
            let rand_val: f64 = rng.gen();
            let selected_idx = cumulative_weights
                .iter()
                .position(|&cum_weight| rand_val <= cum_weight)
                .unwrap_or(n_samples - 1);
            resampled_indices.push(selected_idx);
        }

        let x_resampled = x.select(Axis(0), &resampled_indices);
        let y_resampled = Array1::from_iter(resampled_indices.iter().map(|&i| y[i]));

        Ok((x_resampled, y_resampled))
    }

    /// Predict using the boosted ensemble
    pub fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let estimators = self
            .estimators_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;
        let weights = self.estimator_weights_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_samples = x.nrows();
        let mut decision_scores: Array1<f64> = Array1::zeros(n_samples);

        // Accumulate weighted predictions
        for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
            let predictions = estimator.ensemble_predict(x)?;

            for i in 0..n_samples {
                let prediction_score = if predictions[i] == classes[1] {
                    1.0
                } else {
                    -1.0
                };
                decision_scores[i] += weight * prediction_score;
            }
        }

        // Convert decision scores to class predictions
        let final_predictions =
            decision_scores.mapv(
                |score: f64| {
                    if score > 0.0 {
                        classes[1]
                    } else {
                        classes[0]
                    }
                },
            );

        Ok(final_predictions)
    }

    /// Predict class probabilities using the boosted ensemble
    pub fn predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let estimators = self
            .estimators_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;
        let weights = self.estimator_weights_.as_ref().unwrap();
        let classes = self.classes_.as_ref().unwrap();

        let n_samples = x.nrows();
        let mut decision_scores: Array1<f64> = Array1::zeros(n_samples);

        // Accumulate weighted predictions
        for (estimator, &weight) in estimators.iter().zip(weights.iter()) {
            let predictions = estimator.ensemble_predict(x)?;

            for i in 0..n_samples {
                let prediction_score = if predictions[i] == classes[1] {
                    1.0
                } else {
                    -1.0
                };
                decision_scores[i] += weight * prediction_score;
            }
        }

        // Convert decision scores to probabilities using sigmoid
        let mut probabilities = Array2::zeros((n_samples, 2));
        for i in 0..n_samples {
            let prob_positive = 1.0 / (1.0 + (-decision_scores[i]).exp());
            probabilities[[i, 0]] = 1.0 - prob_positive;
            probabilities[[i, 1]] = prob_positive;
        }

        Ok(probabilities)
    }

    /// Get feature importance based on estimator weights
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        // For Naive Bayes, feature importance is not straightforward
        // This would require access to the internal parameters of each estimator
        // For now, return None - could be implemented with specific NB implementations
        None
    }

    /// Get the number of estimators actually used
    pub fn n_estimators_(&self) -> usize {
        self.estimators_.as_ref().map(|e| e.len()).unwrap_or(0)
    }
}

/// Wrapper structs for making Naive Bayes classifiers compatible with ensemble trait
#[derive(Debug, Clone)]
struct GaussianNBEstimator {
    model: Option<GaussianNB<Trained>>,
    classes: Option<Array1<i32>>,
}

impl GaussianNBEstimator {
    fn new() -> Self {
        Self {
            model: None,
            classes: None,
        }
    }
}

impl NaiveBayesEstimator for GaussianNBEstimator {
    fn fit_clone(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Box<dyn NaiveBayesEstimator>> {
        let model = GaussianNB::new();
        let fitted_model = model.fit(x, y)?;

        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();

        Ok(Box::new(GaussianNBEstimator {
            model: Some(fitted_model),
            classes: Some(Array1::from_vec(classes)),
        }))
    }

    fn ensemble_predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        model.predict(x)
    }

    fn ensemble_predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict_proba".to_string(),
        })?;
        model.predict_proba(x)
    }

    fn ensemble_classes(&self) -> &Array1<i32> {
        self.classes.as_ref().unwrap()
    }
}

/// MultinomialNB wrapper for ensemble compatibility
#[derive(Debug, Clone)]
struct MultinomialNBEstimator {
    model: Option<MultinomialNB<Trained>>,
    classes: Option<Array1<i32>>,
}

impl MultinomialNBEstimator {
    fn new() -> Self {
        Self {
            model: None,
            classes: None,
        }
    }
}

impl NaiveBayesEstimator for MultinomialNBEstimator {
    fn fit_clone(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Box<dyn NaiveBayesEstimator>> {
        let model = MultinomialNB::new();
        let fitted_model = model.fit(x, y)?;

        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();

        Ok(Box::new(MultinomialNBEstimator {
            model: Some(fitted_model),
            classes: Some(Array1::from_vec(classes)),
        }))
    }

    fn ensemble_predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        model.predict(x)
    }

    fn ensemble_predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict_proba".to_string(),
        })?;
        model.predict_proba(x)
    }

    fn ensemble_classes(&self) -> &Array1<i32> {
        self.classes.as_ref().unwrap()
    }
}

/// BernoulliNB wrapper for ensemble compatibility
#[derive(Debug, Clone)]
struct BernoulliNBEstimator {
    model: Option<BernoulliNB<Trained>>,
    classes: Option<Array1<i32>>,
}

impl BernoulliNBEstimator {
    fn new() -> Self {
        Self {
            model: None,
            classes: None,
        }
    }
}

impl NaiveBayesEstimator for BernoulliNBEstimator {
    fn fit_clone(&self, x: &Array2<f64>, y: &Array1<i32>) -> Result<Box<dyn NaiveBayesEstimator>> {
        let model = BernoulliNB::new();
        let fitted_model = model.fit(x, y)?;

        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();

        Ok(Box::new(BernoulliNBEstimator {
            model: Some(fitted_model),
            classes: Some(Array1::from_vec(classes)),
        }))
    }

    fn ensemble_predict(&self, x: &Array2<f64>) -> Result<Array1<i32>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict".to_string(),
        })?;
        model.predict(x)
    }

    fn ensemble_predict_proba(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let model = self.model.as_ref().ok_or_else(|| SklearsError::NotFitted {
            operation: "predict_proba".to_string(),
        })?;
        model.predict_proba(x)
    }

    fn ensemble_classes(&self) -> &Array1<i32> {
        self.classes.as_ref().unwrap()
    }
}

/// Stacking ensemble for Naive Bayes classifiers
pub struct StackingNaiveBayes {
    /// Base level estimators
    base_estimators: Vec<(String, Box<dyn NaiveBayesEstimator>)>,
    /// Meta-learner (final estimator)
    meta_learner: Box<dyn NaiveBayesEstimator>,
    /// Cross-validation strategy for generating meta-features
    cv_folds: usize,
    /// Whether to use the original features along with predictions
    passthrough: bool,
    /// Fitted base estimators
    fitted_base_estimators: Option<Vec<Box<dyn NaiveBayesEstimator>>>,
    /// Fitted meta-learner
    fitted_meta_learner: Option<Box<dyn NaiveBayesEstimator>>,
    /// Classes
    classes_: Option<Array1<i32>>,
    /// Random state for cross-validation
    random_state: Option<u64>,
}

impl StackingNaiveBayes {
    pub fn new() -> Self {
        Self {
            base_estimators: Vec::new(),
            meta_learner: Box::new(GaussianNBEstimator::new()),
            cv_folds: 5,
            passthrough: false,
            fitted_base_estimators: None,
            fitted_meta_learner: None,
            classes_: None,
            random_state: None,
        }
    }

    /// Add a base estimator to the ensemble
    pub fn add_estimator(mut self, name: String, estimator: Box<dyn NaiveBayesEstimator>) -> Self {
        self.base_estimators.push((name, estimator));
        self
    }

    /// Add a Gaussian Naive Bayes base estimator
    pub fn add_gaussian_nb(mut self, name: String) -> Self {
        self.base_estimators
            .push((name, Box::new(GaussianNBEstimator::new())));
        self
    }

    /// Add a Multinomial Naive Bayes base estimator
    pub fn add_multinomial_nb(mut self, name: String) -> Self {
        self.base_estimators
            .push((name, Box::new(MultinomialNBEstimator::new())));
        self
    }

    /// Add a Bernoulli Naive Bayes base estimator
    pub fn add_bernoulli_nb(mut self, name: String) -> Self {
        self.base_estimators
            .push((name, Box::new(BernoulliNBEstimator::new())));
        self
    }

    /// Set the meta-learner
    pub fn meta_learner(mut self, meta_learner: Box<dyn NaiveBayesEstimator>) -> Self {
        self.meta_learner = meta_learner;
        self
    }

    /// Set meta-learner to Gaussian Naive Bayes
    pub fn gaussian_meta_learner(mut self) -> Self {
        self.meta_learner = Box::new(GaussianNBEstimator::new());
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv_folds(mut self, cv_folds: usize) -> Self {
        self.cv_folds = cv_folds;
        self
    }

    /// Enable/disable passthrough of original features
    pub fn passthrough(mut self, passthrough: bool) -> Self {
        self.passthrough = passthrough;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit the stacking ensemble
    pub fn fit(&mut self, X: &Array2<f64>, y: &Array1<i32>) -> Result<()> {
        if self.base_estimators.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "base_estimators".to_string(),
                reason: "At least one base estimator must be added".to_string(),
            });
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        self.classes_ = Some(Array1::from_vec(classes));

        // Generate meta-features using cross-validation
        let meta_features = self.generate_meta_features(X, y)?;

        // Fit base estimators on full training data
        let mut fitted_base_estimators = Vec::new();
        for (_, base_estimator) in &self.base_estimators {
            let fitted_estimator = base_estimator.fit_clone(X, y)?;
            fitted_base_estimators.push(fitted_estimator);
        }
        self.fitted_base_estimators = Some(fitted_base_estimators);

        // Fit meta-learner on meta-features
        let fitted_meta_learner = self.meta_learner.fit_clone(&meta_features, y)?;
        self.fitted_meta_learner = Some(fitted_meta_learner);

        Ok(())
    }

    /// Generate meta-features using cross-validation
    #[allow(non_snake_case)]
    fn generate_meta_features(&self, X: &Array2<f64>, y: &Array1<i32>) -> Result<Array2<f64>> {
        let n_samples = X.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_base_estimators = self.base_estimators.len();

        // Determine feature dimensions
        let meta_feature_dim = if self.passthrough {
            X.ncols() + n_base_estimators * n_classes
        } else {
            n_base_estimators * n_classes
        };

        let mut meta_features = Array2::zeros((n_samples, meta_feature_dim));

        // Generate cross-validation folds
        let folds = self.generate_cv_folds(n_samples)?;

        // For each fold, train base estimators and generate predictions
        for (train_indices, test_indices) in folds {
            // Create training and test sets for this fold
            let X_train = self.extract_rows(X, &train_indices);
            let y_train = self.extract_elements(y, &train_indices);
            let X_test = self.extract_rows(X, &test_indices);

            // Train base estimators on fold training data
            for (estimator_idx, (_, base_estimator)) in self.base_estimators.iter().enumerate() {
                let fitted_estimator = base_estimator.fit_clone(&X_train, &y_train)?;
                let predictions = fitted_estimator.ensemble_predict_proba(&X_test)?;

                // Store predictions as meta-features
                for (test_sample_idx, &original_idx) in test_indices.iter().enumerate() {
                    let start_col = estimator_idx * n_classes;
                    for class_idx in 0..n_classes {
                        meta_features[[original_idx, start_col + class_idx]] =
                            predictions[[test_sample_idx, class_idx]];
                    }
                }
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            let passthrough_start = n_base_estimators * n_classes;
            for i in 0..n_samples {
                for j in 0..X.ncols() {
                    meta_features[[i, passthrough_start + j]] = X[[i, j]];
                }
            }
        }

        Ok(meta_features)
    }

    /// Generate cross-validation folds
    fn generate_cv_folds(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut folds = Vec::new();
        let fold_size = n_samples / self.cv_folds;
        let remainder = n_samples % self.cv_folds;

        for i in 0..self.cv_folds {
            let start = i * fold_size + (i.min(remainder));
            let end = start + fold_size + if i < remainder { 1 } else { 0 };

            let test_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..n_samples)
                .filter(|&x| !test_indices.contains(&x))
                .collect();

            folds.push((train_indices, test_indices));
        }

        Ok(folds)
    }

    /// Extract rows from a matrix by indices
    fn extract_rows(&self, matrix: &Array2<f64>, indices: &[usize]) -> Array2<f64> {
        let n_rows = indices.len();
        let n_cols = matrix.ncols();
        let mut result = Array2::zeros((n_rows, n_cols));

        for (new_row, &old_row) in indices.iter().enumerate() {
            result.row_mut(new_row).assign(&matrix.row(old_row));
        }

        result
    }

    /// Extract elements from an array by indices
    fn extract_elements(&self, array: &Array1<i32>, indices: &[usize]) -> Array1<i32> {
        let mut result = Array1::zeros(indices.len());
        for (new_idx, &old_idx) in indices.iter().enumerate() {
            result[new_idx] = array[old_idx];
        }
        result
    }

    /// Predict using the stacking ensemble
    pub fn predict(&self, X: &Array2<f64>) -> Result<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let classes = self
            .classes_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict".to_string(),
            })?;

        let predictions = probabilities
            .outer_iter()
            .map(|row| {
                let max_idx = row
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                classes[max_idx]
            })
            .collect();

        Ok(Array1::from_vec(predictions))
    }

    /// Predict probabilities using the stacking ensemble
    pub fn predict_proba(&self, X: &Array2<f64>) -> Result<Array2<f64>> {
        let fitted_base_estimators =
            self.fitted_base_estimators
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba".to_string(),
                })?;
        let fitted_meta_learner =
            self.fitted_meta_learner
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba".to_string(),
                })?;

        let n_samples = X.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_base_estimators = fitted_base_estimators.len();

        // Generate meta-features from base estimator predictions
        let meta_feature_dim = if self.passthrough {
            X.ncols() + n_base_estimators * n_classes
        } else {
            n_base_estimators * n_classes
        };

        let mut meta_features = Array2::zeros((n_samples, meta_feature_dim));

        // Get predictions from base estimators
        for (estimator_idx, estimator) in fitted_base_estimators.iter().enumerate() {
            let predictions = estimator.ensemble_predict_proba(X)?;
            let start_col = estimator_idx * n_classes;

            for sample_idx in 0..n_samples {
                for class_idx in 0..n_classes {
                    meta_features[[sample_idx, start_col + class_idx]] =
                        predictions[[sample_idx, class_idx]];
                }
            }
        }

        // Add original features if passthrough is enabled
        if self.passthrough {
            let passthrough_start = n_base_estimators * n_classes;
            for i in 0..n_samples {
                for j in 0..X.ncols() {
                    meta_features[[i, passthrough_start + j]] = X[[i, j]];
                }
            }
        }

        // Get final predictions from meta-learner
        fitted_meta_learner.ensemble_predict_proba(&meta_features)
    }

    /// Get the base estimator names
    pub fn base_estimator_names(&self) -> Vec<&String> {
        self.base_estimators.iter().map(|(name, _)| name).collect()
    }

    /// Get the number of base estimators
    pub fn n_base_estimators(&self) -> usize {
        self.base_estimators.len()
    }

    /// Get feature importance (simplified - based on meta-learner if applicable)
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        // For stacking with Naive Bayes, feature importance is not straightforward
        // This would require analyzing the meta-learner's parameters
        None
    }

    /// Check if the ensemble is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted_base_estimators.is_some() && self.fitted_meta_learner.is_some()
    }
}

impl Default for StackingNaiveBayes {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_voting_naive_bayes() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut ensemble = VotingNaiveBayes::new(VotingStrategy::Soft);

        // For now, just test that the structure works
        // In a full implementation, we'd add actual estimators
        assert_eq!(ensemble.estimators.len(), 0);
    }

    #[test]
    fn test_bootstrap_config() {
        let config = BootstrapConfig::default();
        assert_eq!(config.with_replacement, true);
        assert_eq!(config.sample_fraction, 1.0);
        assert!(config.random_state.is_none());

        let custom_config = BootstrapConfig {
            with_replacement: false,
            sample_fraction: 0.8,
            random_state: Some(42),
        };
        assert_eq!(custom_config.with_replacement, false);
        assert_eq!(custom_config.sample_fraction, 0.8);
        assert_eq!(custom_config.random_state, Some(42));
    }

    #[test]
    fn test_adaboost_naive_bayes() {
        let x = array![
            [1.0, 2.0],
            [2.0, 1.0],
            [3.0, 4.0],
            [4.0, 3.0],
            [-1.0, -2.0],
            [-2.0, -1.0],
            [-3.0, -4.0],
            [-4.0, -3.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let mut adaboost = AdaBoostNaiveBayes::new("GaussianNB")
            .n_estimators(5)
            .learning_rate(0.5)
            .random_state(42);

        let result = adaboost.fit(&x, &y);
        assert!(result.is_ok());

        let predictions = adaboost.predict(&x);
        assert!(predictions.is_ok());

        let probs = adaboost.predict_proba(&x);
        assert!(probs.is_ok());

        let probs = probs.unwrap();
        assert_eq!(probs.nrows(), 8);
        assert_eq!(probs.ncols(), 2);

        // Check that probabilities sum to approximately 1
        for i in 0..probs.nrows() {
            let row_sum: f64 = probs.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 0.1); // Allow some tolerance
        }
    }

    #[test]
    fn test_boosting_strategies() {
        let x = array![
            [2.0, 3.0],
            [3.0, 2.0],
            [1.0, 4.0],
            [4.0, 1.0],
            [-2.0, -3.0],
            [-3.0, -2.0],
            [-1.0, -4.0],
            [-4.0, -1.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        // Test different boosting strategies
        let strategies = vec![
            BoostingStrategy::AdaBoost,
            BoostingStrategy::LogitBoost,
            BoostingStrategy::GentleAdaBoost,
        ];

        for strategy in strategies {
            let mut adaboost = AdaBoostNaiveBayes::new("GaussianNB")
                .n_estimators(3)
                .strategy(strategy)
                .random_state(42);

            let result = adaboost.fit(&x, &y);
            assert!(
                result.is_ok(),
                "Failed to fit with strategy: {:?}",
                adaboost.strategy
            );

            let predictions = adaboost.predict(&x);
            assert!(
                predictions.is_ok(),
                "Failed to predict with strategy: {:?}",
                adaboost.strategy
            );
        }
    }

    #[test]
    fn test_adaboost_with_different_base_classifiers() {
        let x = array![
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let base_classifiers = vec!["GaussianNB", "BernoulliNB"];

        for base_classifier in base_classifiers {
            let mut adaboost = AdaBoostNaiveBayes::new(base_classifier)
                .n_estimators(3)
                .random_state(42);

            let result = adaboost.fit(&x, &y);
            assert!(
                result.is_ok(),
                "Failed to fit with base classifier: {}",
                base_classifier
            );

            assert!(adaboost.n_estimators_() > 0);
        }
    }

    #[test]
    fn test_adaboost_invalid_multiclass() {
        let x = array![[1.0, 2.0], [2.0, 1.0], [3.0, 4.0]];
        let y = array![0, 1, 2]; // More than 2 classes

        let mut adaboost = AdaBoostNaiveBayes::new("GaussianNB");
        let result = adaboost.fit(&x, &y);
        assert!(result.is_err()); // Should fail for multiclass
    }

    #[test]
    fn test_adaboost_invalid_classifier() {
        let mut adaboost = AdaBoostNaiveBayes::new("InvalidClassifier");
        let result = adaboost.create_base_classifier();
        assert!(result.is_err());
    }
}
