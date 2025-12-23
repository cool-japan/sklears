//! Boosting multiclass strategies
//!
//! This module implements boosting algorithms including AdaBoost and Gradient Boosting
//! for multiclass classification.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom, Rng};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

/// AdaBoost multiclass boosting strategies
#[derive(Debug, Clone, PartialEq)]
pub enum AdaBoostStrategy {
    /// AdaBoost.M1 - Original discrete AdaBoost for multiclass
    M1,
    /// AdaBoost.M2 - Real AdaBoost for multiclass with weighted error
    M2,
}

impl Default for AdaBoostStrategy {
    fn default() -> Self {
        Self::M1
    }
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
    estimator_weights: Array1<f64>,
    /// Classes seen during training
    classes: Array1<i32>,
    /// Number of classes
    n_classes: usize,
    /// Training error evolution
    errors: Vec<f64>,
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
    T: Predict<Array2<f64>, Array1<i32>>,
{
    /// Get the feature importances (if base estimator supports it)
    pub fn feature_importances(&self) -> Option<Array1<f64>> {
        // This would need to be implemented based on base estimator capabilities
        None
    }

    /// Get the training errors
    pub fn errors(&self) -> &[f64] {
        &self.base_estimator.errors
    }

    /// Get the estimator weights
    pub fn estimator_weights(&self) -> &Array1<f64> {
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

impl<C> Fit<Array2<f64>, Array1<i32>> for AdaBoostClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted = TrainedAdaBoost<C::Fitted>;

    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(X, y)?;
        let (_n_samples, _n_features) = X.dim();

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
            AdaBoostStrategy::M1 => self.fit_adaboost_m1(X, y, classes_array, n_classes),
            AdaBoostStrategy::M2 => self.fit_adaboost_m2(X, y, classes_array, n_classes),
        }
    }
}

impl<C> AdaBoostClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn fit_adaboost_m1(
        self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: Array1<i32>,
        n_classes: usize,
    ) -> SklResult<TrainedAdaBoost<C::Fitted>> {
        let (n_samples, _n_features) = X.dim();
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut errors = Vec::new();

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        for t in 0..self.config.n_estimators {
            // Bootstrap sample with weights
            let (X_bootstrap, y_bootstrap) =
                self.bootstrap_sample(X, y, &sample_weights, &mut rng)?;

            // Train base estimator
            let estimator = self
                .base_estimator
                .clone()
                .fit(&X_bootstrap, &y_bootstrap)?;

            // Make predictions on original data
            let predictions = estimator.predict(X)?;

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
            if weighted_error >= 1.0 - 1.0 / n_classes as f64 {
                if t == 0 {
                    return Err(SklearsError::InvalidInput(
                        "Base estimator performs worse than random".to_string(),
                    ));
                }
                break;
            }

            // Calculate estimator weight (AdaBoost.M1)
            let alpha = self.config.learning_rate * ((1.0 - weighted_error) / weighted_error).ln()
                + (n_classes as f64 - 1.0).ln();

            // Update sample weights
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    sample_weights[i] *= (alpha / self.config.learning_rate).exp();
                }
            }

            // Normalize sample weights
            let weight_sum: f64 = sample_weights.sum();
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
        X: &Array2<f64>,
        y: &Array1<i32>,
        classes: Array1<i32>,
        n_classes: usize,
    ) -> SklResult<TrainedAdaBoost<C::Fitted>> {
        let (n_samples, _n_features) = X.dim();

        // For AdaBoost.M2, we maintain a distribution over sample-label pairs
        let mut sample_weights = Array1::from_elem(n_samples, 1.0 / n_samples as f64);

        let mut estimators = Vec::new();
        let mut estimator_weights = Vec::new();
        let mut errors = Vec::new();

        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        for t in 0..self.config.n_estimators {
            // Bootstrap sample with weights
            let (X_bootstrap, y_bootstrap) =
                self.bootstrap_sample(X, y, &sample_weights, &mut rng)?;

            // Train base estimator
            let estimator = self
                .base_estimator
                .clone()
                .fit(&X_bootstrap, &y_bootstrap)?;

            // Make predictions on original data
            let predictions = estimator.predict(X)?;

            // Calculate pseudo-loss for AdaBoost.M2
            let mut pseudo_loss = 0.0;
            for i in 0..n_samples {
                if predictions[i] != y[i] {
                    // For wrong predictions, add weight proportional to how wrong
                    pseudo_loss += sample_weights[i] * 0.5;
                } else {
                    // For correct predictions, penalize based on confidence
                    pseudo_loss += sample_weights[i] * 0.5 * (1.0 - 1.0 / n_classes as f64);
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
            let weight_sum: f64 = sample_weights.sum();
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

    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        sample_weights: &Array1<f64>,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<i32>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();

        // Create cumulative distribution
        let mut cum_weights = Array1::zeros(n_samples);
        cum_weights[0] = sample_weights[0];
        for i in 1..n_samples {
            cum_weights[i] = cum_weights[i - 1] + sample_weights[i];
        }

        // Sample with replacement
        let mut X_bootstrap = Array2::zeros((n_samples, n_features));
        let mut y_bootstrap = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let r: f64 = rng.gen();
            let idx = cum_weights
                .iter()
                .position(|&w| w >= r)
                .unwrap_or(n_samples - 1);

            X_bootstrap.row_mut(i).assign(&X.row(idx));
            y_bootstrap[i] = y[idx];
        }

        Ok((X_bootstrap, y_bootstrap))
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>> for TrainedAdaBoost<T>
where
    T: Predict<Array2<f64>, Array1<i32>>,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let n_samples = X.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        // Accumulate weighted votes
        for (estimator, &weight) in trained_data
            .estimators
            .iter()
            .zip(trained_data.estimator_weights.iter())
        {
            let predictions = estimator.predict(X)?;

            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred) {
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
                .max_by(|(_, a): &(_, &f64), (_, b): &(_, &f64)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = trained_data.classes[class_idx];
        }

        Ok(predictions)
    }
}

impl<T> PredictProba<Array2<f64>, Array2<f64>> for TrainedAdaBoost<T>
where
    T: Predict<Array2<f64>, Array1<i32>>,
{
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        // Accumulate weighted votes
        for (estimator, &weight) in trained_data
            .estimators
            .iter()
            .zip(trained_data.estimator_weights.iter())
        {
            let predictions = estimator.predict(X)?;

            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred) {
                    votes[[i, class_idx]] += weight;
                }
            }
        }

        // Convert votes to probabilities using softmax
        let mut probabilities = Array2::<f64>::zeros((n_samples, trained_data.n_classes));
        for i in 0..n_samples {
            let row = votes.row(i);
            let max_vote = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

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
                    probabilities[[i, j]] = 1.0 / trained_data.n_classes as f64;
                }
            }
        }

        Ok(probabilities)
    }
}

/// Loss functions for gradient boosting
#[derive(Debug, Clone, PartialEq)]
pub enum GradientBoostingLoss {
    /// Multinomial deviance loss for multiclass classification
    Deviance,
    /// Exponential loss (equivalent to AdaBoost)
    Exponential,
}

impl Default for GradientBoostingLoss {
    fn default() -> Self {
        Self::Deviance
    }
}

/// Gradient Boosting configuration
#[derive(Debug, Clone)]
pub struct GradientBoostingConfig {
    /// Number of boosting stages to perform
    pub n_estimators: usize,
    /// Learning rate shrinks the contribution of each classifier
    pub learning_rate: f64,
    /// Maximum depth of the individual regression estimators
    pub max_depth: Option<usize>,
    /// Minimum number of samples required to split an internal node
    pub min_samples_split: usize,
    /// Minimum number of samples required to be at a leaf node
    pub min_samples_leaf: usize,
    /// Fraction of samples used for fitting the individual base learners
    pub subsample: f64,
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Loss function to be optimized
    pub loss: GradientBoostingLoss,
    /// Maximum number of leaf nodes in each tree
    pub max_leaf_nodes: Option<usize>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
}

impl Default for GradientBoostingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 100,
            learning_rate: 0.1,
            max_depth: Some(3),
            min_samples_split: 2,
            min_samples_leaf: 1,
            subsample: 1.0,
            random_state: None,
            loss: GradientBoostingLoss::default(),
            max_leaf_nodes: None,
            n_jobs: None,
        }
    }
}

/// Gradient Boosting Multiclass Classifier
///
/// Gradient Boosting for classification builds an additive model in a forward stage-wise
/// fashion; it allows for the optimization of arbitrary differentiable loss functions.
/// In each stage a regression tree is fit on the negative gradient of the binomial or
/// multinomial deviance loss function.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::GradientBoostingClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let gb = GradientBoostingClassifier::new(base_classifier);
/// ```
#[derive(Debug)]
pub struct GradientBoostingClassifier<C, S = Untrained> {
    base_estimator: C,
    config: GradientBoostingConfig,
    state: PhantomData<S>,
}

/// Trained data for Gradient Boosting classifier
#[derive(Debug, Clone)]
pub struct GradientBoostingTrainedData<T> {
    /// Base estimators at each boosting stage
    pub estimators: Vec<Vec<T>>, // One set of estimators per class
    /// Class labels
    pub classes: Array1<i32>,
    /// Number of classes
    pub n_classes: usize,
    /// Initial class priors
    pub init_priors: Array1<f64>,
    /// Feature importances
    pub feature_importances: Option<Array1<f64>>,
    /// Training loss at each iteration
    pub train_score: Vec<f64>,
}

impl<C> GradientBoostingClassifier<C, Untrained> {
    /// Create a new GradientBoostingClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: GradientBoostingConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for GradientBoostingClassifier
    pub fn builder(base_estimator: C) -> GradientBoostingBuilder<C> {
        GradientBoostingBuilder::new(base_estimator)
    }

    /// Set the number of boosting stages
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: GradientBoostingLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

/// Builder for GradientBoostingClassifier
#[derive(Debug)]
pub struct GradientBoostingBuilder<C> {
    base_estimator: C,
    config: GradientBoostingConfig,
}

impl<C> GradientBoostingBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: GradientBoostingConfig::default(),
        }
    }

    /// Set the number of boosting stages
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the loss function
    pub fn loss(mut self, loss: GradientBoostingLoss) -> Self {
        self.config.loss = loss;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Build the GradientBoostingClassifier
    pub fn build(self) -> GradientBoostingClassifier<C, Untrained> {
        GradientBoostingClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for GradientBoostingClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C> Estimator for GradientBoostingClassifier<C, Untrained> {
    type Config = GradientBoostingConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

pub type TrainedGradientBoosting<T> =
    GradientBoostingClassifier<GradientBoostingTrainedData<T>, Trained>;

impl<T> TrainedGradientBoosting<T> {
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.n_classes
    }

    /// Get the feature importances
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.base_estimator.feature_importances.as_ref()
    }

    /// Get the training scores
    pub fn train_score(&self) -> &[f64] {
        &self.base_estimator.train_score
    }
}
