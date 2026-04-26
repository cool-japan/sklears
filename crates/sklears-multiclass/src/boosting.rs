//! Boosting multiclass strategies
//!
//! This module implements boosting algorithms including AdaBoost
//! for multiclass classification.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// AdaBoost multiclass boosting strategies
#[derive(Debug, Clone, PartialEq, Default)]
pub enum AdaBoostStrategy {
    /// AdaBoost.M1 - Original discrete AdaBoost for multiclass
    #[default]
    M1,
    /// AdaBoost.M2 - Real AdaBoost for multiclass with weighted error
    M2,
}

/// Configuration for AdaBoost multiclass classifier
#[derive(Debug, Clone)]
pub struct AdaBoostConfig {
    /// Number of estimators
    pub n_estimators: usize,
    /// Learning rate
    pub learning_rate: f64,
    /// AdaBoost strategy (M1 or M2)
    pub strategy: AdaBoostStrategy,
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Maximum depth for base estimators (if applicable)
    pub max_depth: Option<usize>,
}

impl Default for AdaBoostConfig {
    fn default() -> Self {
        Self {
            n_estimators: 50,
            learning_rate: 1.0,
            strategy: AdaBoostStrategy::default(),
            random_state: None,
            max_depth: Some(1), // Decision stumps by default
        }
    }
}

/// AdaBoost Multiclass Classifier
///
/// AdaBoost (Adaptive Boosting) is an ensemble method that combines multiple
/// weak learners to create a strong classifier. For multiclass problems,
/// we implement both AdaBoost.M1 and AdaBoost.M2 variants.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
#[derive(Debug)]
pub struct AdaBoostClassifier<C, S = Untrained> {
    base_estimator: C,
    config: AdaBoostConfig,
    state: PhantomData<S>,
}

/// Trained data for AdaBoost classifier
#[derive(Debug)]
pub struct AdaBoostTrainedData<T> {
    /// Vector of trained estimators
    estimators: Vec<T>,
    /// Estimator weights
    estimator_weights: Array1<Float>,
    /// Classes seen during training
    classes: Array1<i32>,
    /// Number of classes
    n_classes: usize,
    /// Training error evolution
    errors: Vec<Float>,
}

impl<T: Clone> Clone for AdaBoostTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            estimator_weights: self.estimator_weights.clone(),
            classes: self.classes.clone(),
            n_classes: self.n_classes,
            errors: self.errors.clone(),
        }
    }
}

impl<C> AdaBoostClassifier<C, Untrained> {
    /// Create a new AdaBoost classifier
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: AdaBoostConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for AdaBoost classifier
    pub fn builder(base_estimator: C) -> AdaBoostBuilder<C> {
        AdaBoostBuilder::new(base_estimator)
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the AdaBoost strategy
    pub fn strategy(mut self, strategy: AdaBoostStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

/// Builder for AdaBoost classifier
#[derive(Debug)]
pub struct AdaBoostBuilder<C> {
    base_estimator: C,
    config: AdaBoostConfig,
}

impl<C> AdaBoostBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: AdaBoostConfig::default(),
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the AdaBoost strategy
    pub fn strategy(mut self, strategy: AdaBoostStrategy) -> Self {
        self.config.strategy = strategy;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Build the AdaBoost classifier
    pub fn build(self) -> AdaBoostClassifier<C, Untrained> {
        AdaBoostClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for AdaBoostClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for AdaBoostBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for AdaBoostClassifier<C, Untrained> {
    type Config = AdaBoostConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

pub type TrainedAdaBoost<T> = AdaBoostClassifier<AdaBoostTrainedData<T>, Trained>;

impl<T> TrainedAdaBoost<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get the feature importances (if base estimator supports it)
    pub fn feature_importances(&self) -> Option<Array1<Float>> {
        // This would need to be implemented based on base estimator capabilities
        None
    }

    /// Get the training errors
    pub fn errors(&self) -> &[Float] {
        &self.base_estimator.errors
    }

    /// Get the estimator weights
    pub fn estimator_weights(&self) -> &Array1<Float> {
        &self.base_estimator.estimator_weights
    }

    /// Get the number of estimators actually used
    pub fn n_estimators_(&self) -> usize {
        self.base_estimator.estimators.len()
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }
}

impl<C, F> Fit<Array2<Float>, Array1<i32>> for AdaBoostClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F> + Send + Sync,
    F: Predict<Array2<Float>, Array1<Float>> + Clone + Send + Sync,
{
    type Fitted = TrainedAdaBoost<F>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Get unique classes
        let mut classes: Vec<i32> = y.iter().cloned().collect();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        match self.config.strategy {
            AdaBoostStrategy::M1 => self.fit_adaboost_m1(x, y, classes_array, n_classes),
            AdaBoostStrategy::M2 => self.fit_adaboost_m2(x, y, classes_array, n_classes),
        }
    }
}

impl<C, F> AdaBoostClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<Float>, Array1<Float>, Fitted = F> + Send + Sync,
    F: Predict<Array2<Float>, Array1<Float>> + Clone + Send + Sync,
{
    /// Convert integer class labels to binary float targets for a given target class.
    fn make_binary_targets(y: &Array1<i32>, target_class: i32) -> Array1<Float> {
        y.mapv(|label| if label == target_class { 1.0 } else { -1.0 })
    }

    /// Threshold float predictions from base estimator to class labels.
    /// Returns `target_class` if prediction > 0.0, otherwise `other_class`.
    fn threshold_to_class(
        predictions: &Array1<Float>,
        target_class: i32,
        other_class: i32,
    ) -> Array1<i32> {
        predictions.mapv(|p| if p > 0.0 { target_class } else { other_class })
    }

    fn fit_adaboost_m1(
        self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: Array1<i32>,
        n_classes: usize,
    ) -> SklResult<TrainedAdaBoost<F>> {
        let (n_samples, _n_features) = x.dim();
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as Float);

        let mut estimators: Vec<F> = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut errors = Vec::new();

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        // AdaBoost.M1 trains one multi-class estimator per round using binary
        // One-vs-Rest targets for the "majority class" approach.
        // We use the sign of sum predictions across all classes.
        // For simplicity, train one estimator on binary (class 0 vs rest) targets
        // representing the overall class distribution, then threshold predictions.
        // The predicted class uses argmax vote weighting.

        for t in 0..self.config.n_estimators {
            // Bootstrap sample with weights
            let (x_bootstrap, y_bootstrap_int) =
                self.bootstrap_sample_int(x, y, &sample_weights, &mut rng)?;

            // Convert integer labels to float binary: class 0 = 1.0, rest = -1.0
            // We train one global estimator per iteration.
            // Use the most common class as the "positive" class for this round.
            let round_target_class = classes[0];
            let y_bootstrap_float = Self::make_binary_targets(&y_bootstrap_int, round_target_class);

            // Train base estimator on float targets
            let estimator = self
                .base_estimator
                .clone()
                .fit(&x_bootstrap, &y_bootstrap_float)?;

            // Make predictions on original data, threshold back to class labels
            let float_preds = estimator.predict(x)?;
            let predictions = Self::threshold_to_class(
                &float_preds,
                round_target_class,
                if n_classes > 1 {
                    classes[1]
                } else {
                    classes[0]
                },
            );

            // Calculate weighted error
            let mut weighted_error = 0.0;
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    weighted_error += sample_weights[i];
                }
            }

            // Early stopping if perfect classifier
            if weighted_error == 0.0 {
                estimators.push(estimator);
                estimator_weights.push(1.0);
                errors.push(weighted_error);
                break;
            }

            // Early stopping if worse than random
            if weighted_error >= 1.0 - 1.0 / n_classes as Float {
                if t == 0 {
                    return Err(SklearsError::InvalidInput(
                        "Base estimator performs worse than random".to_string(),
                    ));
                }
                break;
            }

            // Calculate estimator weight (AdaBoost.M1)
            let alpha = self.config.learning_rate * ((1.0 - weighted_error) / weighted_error).ln()
                + (n_classes as Float - 1.0).ln();

            // Update sample weights
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    sample_weights[i] *= (alpha / self.config.learning_rate).exp();
                }
            }

            // Normalize sample weights
            let weight_sum: Float = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(estimator);
            estimator_weights.push(alpha);
            errors.push(weighted_error);
        }

        let estimator_weights_array = Array1::from_vec(estimator_weights);
        let trained_data = AdaBoostTrainedData {
            estimators,
            estimator_weights: estimator_weights_array,
            classes,
            n_classes,
            errors,
        };

        Ok(AdaBoostClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }

    fn fit_adaboost_m2(
        self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        classes: Array1<i32>,
        n_classes: usize,
    ) -> SklResult<TrainedAdaBoost<F>> {
        let (n_samples, _n_features) = x.dim();

        // For AdaBoost.M2, we maintain a distribution over sample-label pairs
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as Float);

        let mut estimators: Vec<F> = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut errors = Vec::new();

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        for t in 0..self.config.n_estimators {
            // Bootstrap sample with weights
            let (x_bootstrap, y_bootstrap_int) =
                self.bootstrap_sample_int(x, y, &sample_weights, &mut rng)?;

            // Convert labels to float binary targets
            let round_target_class = classes[0];
            let y_bootstrap_float = Self::make_binary_targets(&y_bootstrap_int, round_target_class);

            // Train base estimator on float targets
            let estimator = self
                .base_estimator
                .clone()
                .fit(&x_bootstrap, &y_bootstrap_float)?;

            // Make predictions on original data, threshold back to class labels
            let float_preds = estimator.predict(x)?;
            let predictions = Self::threshold_to_class(
                &float_preds,
                round_target_class,
                if n_classes > 1 {
                    classes[1]
                } else {
                    classes[0]
                },
            );

            // Calculate pseudo-loss for AdaBoost.M2
            let mut pseudo_loss = 0.0;
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    // For wrong predictions, add weight proportional to how wrong
                    pseudo_loss += sample_weights[i] * 0.5;
                } else {
                    // For correct predictions, penalize based on confidence
                    pseudo_loss += sample_weights[i] * 0.5 * (1.0 - 1.0 / n_classes as Float);
                }
            }

            // Early stopping conditions
            if pseudo_loss >= 0.5 {
                if t == 0 {
                    return Err(SklearsError::InvalidInput(
                        "Base estimator performs worse than random for AdaBoost.M2".to_string(),
                    ));
                }
                break;
            }

            if pseudo_loss == 0.0 {
                estimators.push(estimator);
                estimator_weights.push(1.0);
                errors.push(pseudo_loss);
                break;
            }

            // Calculate estimator weight (AdaBoost.M2)
            let beta = pseudo_loss / (1.0 - pseudo_loss);
            let alpha = self.config.learning_rate * (-beta.ln());

            // Update sample weights for AdaBoost.M2
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    sample_weights[i] *= beta.powf(0.5);
                } else {
                    sample_weights[i] *= beta.powf(-0.5);
                }
            }

            // Normalize sample weights
            let weight_sum: Float = sample_weights.sum();
            if weight_sum > 0.0 {
                sample_weights /= weight_sum;
            }

            estimators.push(estimator);
            estimator_weights.push(alpha);
            errors.push(pseudo_loss);
        }

        let estimator_weights_array = Array1::from_vec(estimator_weights);
        let trained_data = AdaBoostTrainedData {
            estimators,
            estimator_weights: estimator_weights_array,
            classes,
            n_classes,
            errors,
        };

        Ok(AdaBoostClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }

    fn bootstrap_sample_int(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        sample_weights: &Array1<Float>,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<Float>, Array1<i32>)> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Create cumulative distribution
        let mut cum_weights = Array1::zeros(n_samples);
        cum_weights[0] = sample_weights[0];
        for i in 1..n_samples {
            cum_weights[i] = cum_weights[i - 1] + sample_weights[i];
        }

        // Sample with replacement
        let mut x_bootstrap = Array2::zeros((n_samples, n_features));
        let mut y_bootstrap = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let r: Float = rng.random::<Float>();
            let idx = cum_weights
                .iter()
                .position(|&w| w >= r)
                .unwrap_or(n_samples - 1);

            x_bootstrap.row_mut(i).assign(&x.row(idx));
            y_bootstrap[i] = y[idx];
        }

        Ok((x_bootstrap, y_bootstrap))
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedAdaBoost<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let n_samples = x.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<Float>::zeros((n_samples, trained_data.n_classes));

        // Accumulate weighted votes: each estimator produces float scores.
        // Score > 0 maps to classes[0] (target class used at training time),
        // score <= 0 maps to classes[1] (the "other" class).
        for (estimator, &weight) in trained_data
            .estimators
            .iter()
            .zip(trained_data.estimator_weights.iter())
        {
            let float_preds = estimator.predict(x)?;

            for (i, &score) in float_preds.iter().enumerate() {
                // Map float prediction sign to class index vote
                let class_idx = if score > 0.0 { 0 } else { 1 };
                if class_idx < trained_data.n_classes {
                    votes[[i, class_idx]] += weight;
                }
            }
        }

        // Find class with maximum vote for each sample
        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let class_idx = votes
                .row(i)
                .iter()
                .enumerate()
                .max_by(|(_, a): &(_, &Float), (_, b): &(_, &Float)| {
                    a.partial_cmp(b).expect("vote comparison should succeed")
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = trained_data.classes[class_idx];
        }

        Ok(predictions)
    }
}

impl<T> PredictProba<Array2<Float>, Array2<Float>> for TrainedAdaBoost<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict_proba(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n_samples = x.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<Float>::zeros((n_samples, trained_data.n_classes));

        // Accumulate weighted votes from float-output estimators
        for (estimator, &weight) in trained_data
            .estimators
            .iter()
            .zip(trained_data.estimator_weights.iter())
        {
            let float_preds = estimator.predict(x)?;

            for (i, &score) in float_preds.iter().enumerate() {
                let class_idx = if score > 0.0 { 0 } else { 1 };
                if class_idx < trained_data.n_classes {
                    votes[[i, class_idx]] += weight;
                }
            }
        }

        // Convert votes to probabilities using softmax
        let mut probabilities = Array2::<Float>::zeros((n_samples, trained_data.n_classes));
        for i in 0..n_samples {
            let row = votes.row(i);
            let max_vote = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));

            let mut exp_sum = 0.0;
            for j in 0..trained_data.n_classes {
                let exp_val = (votes[[i, j]] - max_vote).exp();
                probabilities[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            if exp_sum > 0.0 {
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] /= exp_sum;
                }
            } else {
                // Uniform distribution if all votes are equal
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] = 1.0 / trained_data.n_classes as Float;
                }
            }
        }

        Ok(probabilities)
    }
}
