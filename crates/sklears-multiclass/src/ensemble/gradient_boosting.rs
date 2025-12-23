//! Gradient Boosting Classifier implementation
//!
//! This module provides a Gradient Boosting classifier for multiclass classification.
//! Gradient Boosting builds an additive model in a forward stage-wise fashion, optimizing
//! arbitrary differentiable loss functions. For multiclass problems, it uses a One-vs-Rest
//! approach where one gradient boosting classifier is trained per class.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::seeded_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

/// Compute residuals for a given loss function
fn compute_residuals_for_loss(
    loss: &GradientBoostingLoss,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> Array1<f64> {
    match loss {
        GradientBoostingLoss::Deviance => {
            // For logistic loss: residual = y - sigmoid(pred)
            let mut residuals = Array1::zeros(y_true.len());
            for i in 0..y_true.len() {
                let sigmoid = 1.0 / (1.0 + (-y_pred[i]).exp());
                residuals[i] = y_true[i] - sigmoid;
            }
            residuals
        }
        GradientBoostingLoss::Exponential => {
            // For exponential loss: residual = y * exp(-y * pred)
            let mut residuals = Array1::zeros(y_true.len());
            for i in 0..y_true.len() {
                residuals[i] = y_true[i] * (-y_true[i] * y_pred[i]).exp();
            }
            residuals
        }
        GradientBoostingLoss::Multinomial => {
            // For multinomial loss: residual = y - softmax(pred)
            let mut residuals = Array1::zeros(y_true.len());
            let max_pred = y_pred.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f64 = y_pred.iter().map(|&p| (p - max_pred).exp()).sum();

            for i in 0..y_true.len() {
                let softmax = (y_pred[i] - max_pred).exp() / exp_sum;
                residuals[i] = y_true[i] - softmax;
            }
            residuals
        }
        GradientBoostingLoss::Huber { delta } => {
            // For Huber loss: hybrid between squared loss and absolute loss
            let mut residuals = Array1::zeros(y_true.len());
            for i in 0..y_true.len() {
                let error = y_true[i] - y_pred[i];
                if error.abs() <= *delta {
                    residuals[i] = error; // Squared loss region
                } else {
                    residuals[i] = delta * error.signum(); // Linear loss region
                }
            }
            residuals
        }
    }
}

/// Compute loss for a given loss function
fn compute_loss_for_loss(
    loss: &GradientBoostingLoss,
    y_true: &Array1<f64>,
    y_pred: &Array1<f64>,
) -> f64 {
    match loss {
        GradientBoostingLoss::Deviance => {
            // Logistic loss
            let mut loss_val = 0.0;
            for i in 0..y_true.len() {
                let sigmoid = 1.0 / (1.0 + (-y_pred[i]).exp());
                loss_val += -y_true[i] * sigmoid.ln() - (1.0 - y_true[i]) * (1.0 - sigmoid).ln();
            }
            loss_val / y_true.len() as f64
        }
        GradientBoostingLoss::Exponential => {
            // Exponential loss
            let mut loss_val = 0.0;
            for i in 0..y_true.len() {
                loss_val += (-y_true[i] * y_pred[i]).exp();
            }
            loss_val / y_true.len() as f64
        }
        GradientBoostingLoss::Multinomial => {
            // Multinomial loss (cross-entropy)
            let mut loss_val = 0.0;
            let max_pred = y_pred.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_sum: f64 = y_pred.iter().map(|&p| (p - max_pred).exp()).sum();

            for i in 0..y_true.len() {
                if y_true[i] > 0.0 {
                    let log_softmax = y_pred[i] - max_pred - exp_sum.ln();
                    loss_val -= y_true[i] * log_softmax;
                }
            }
            loss_val / y_true.len() as f64
        }
        GradientBoostingLoss::Huber { delta } => {
            // Huber loss
            let mut loss_val = 0.0;
            for i in 0..y_true.len() {
                let error = y_true[i] - y_pred[i];
                if error.abs() <= *delta {
                    loss_val += 0.5 * error * error; // Squared loss
                } else {
                    loss_val += delta * (error.abs() - 0.5 * delta); // Linear loss
                }
            }
            loss_val / y_true.len() as f64
        }
    }
}

/// Early stopping configuration
#[derive(Debug, Clone, PartialEq)]
pub struct EarlyStoppingConfig {
    /// Number of rounds without improvement to stop training
    pub n_iter_no_change: usize,
    /// Minimum change in loss to qualify as improvement
    pub tol: f64,
    /// Scoring metric for early stopping
    pub scoring: EarlyStoppingScoring,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            n_iter_no_change: 10,
            tol: 1e-4,
            scoring: EarlyStoppingScoring::Loss,
        }
    }
}

/// Scoring metrics for early stopping
#[derive(Debug, Clone, PartialEq)]
pub enum EarlyStoppingScoring {
    /// Use training loss
    Loss,
    /// Use validation accuracy
    Accuracy,
    /// Use custom scoring function
    Custom,
}

/// Early stopping information
#[derive(Debug, Clone)]
pub struct EarlyStoppingInfo {
    /// Best iteration achieved
    pub best_iteration: usize,
    /// Best score achieved
    pub best_score: f64,
    /// Number of iterations without improvement
    pub n_iter_no_change: usize,
    /// Whether early stopping was triggered
    pub stopped_early: bool,
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
    /// Early stopping configuration
    pub early_stopping: Option<EarlyStoppingConfig>,
    /// Warm start configuration
    pub warm_start: bool,
    /// Validation fraction for early stopping
    pub validation_fraction: f64,
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
            early_stopping: None,
            warm_start: false,
            validation_fraction: 0.1,
        }
    }
}

/// Loss functions for gradient boosting
#[derive(Debug, Clone, PartialEq)]
pub enum GradientBoostingLoss {
    /// Multinomial deviance loss for multiclass classification
    Deviance,
    /// Exponential loss (equivalent to AdaBoost)
    Exponential,
    /// True multinomial deviance loss (direct multiclass)
    Multinomial,
    /// Huber loss for robust classification
    Huber { delta: f64 },
}

impl Default for GradientBoostingLoss {
    fn default() -> Self {
        Self::Deviance
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
    /// Validation scores for early stopping
    pub validation_scores: Vec<f64>,
    /// Number of estimators actually trained (for early stopping)
    pub n_estimators_trained: usize,
    /// Early stopping information
    pub early_stopping_info: Option<EarlyStoppingInfo>,
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

    /// Set the maximum depth of trees
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the subsample fraction
    pub fn subsample(mut self, subsample: f64) -> Self {
        self.config.subsample = subsample;
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

    /// Enable early stopping
    pub fn early_stopping(mut self, early_stopping: EarlyStoppingConfig) -> Self {
        self.config.early_stopping = Some(early_stopping);
        self
    }

    /// Enable warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Set validation fraction for early stopping
    pub fn validation_fraction(mut self, validation_fraction: f64) -> Self {
        self.config.validation_fraction = validation_fraction;
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

    /// Set the maximum depth of trees
    pub fn max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.config.max_depth = max_depth;
        self
    }

    /// Set the subsample fraction
    pub fn subsample(mut self, subsample: f64) -> Self {
        self.config.subsample = subsample;
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

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable early stopping
    pub fn early_stopping(mut self, early_stopping: EarlyStoppingConfig) -> Self {
        self.config.early_stopping = Some(early_stopping);
        self
    }

    /// Enable warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Set validation fraction for early stopping
    pub fn validation_fraction(mut self, validation_fraction: f64) -> Self {
        self.config.validation_fraction = validation_fraction;
        self
    }

    /// Build the classifier
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

type TrainedGradientBoosting<T> =
    GradientBoostingClassifier<GradientBoostingTrainedData<T>, Trained>;

impl<C> Fit<Array2<f64>, Array1<i32>> for GradientBoostingClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<f64>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<f64>> + Clone + Send + Sync,
{
    type Fitted = TrainedGradientBoosting<C::Fitted>;

    #[allow(non_snake_case)]
    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one sample".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        let classes_array = Array1::from_vec(classes.clone());
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes".to_string(),
            ));
        }

        let (n_samples, _n_features) = X.dim();

        // Initialize class priors
        let mut class_counts = vec![0; n_classes];
        for &label in y.iter() {
            if let Some(idx) = classes.iter().position(|&c| c == label) {
                class_counts[idx] += 1;
            }
        }

        let init_priors = Array1::from_vec(
            class_counts
                .iter()
                .map(|&count| count as f64 / n_samples as f64)
                .collect(),
        );

        // Split data for early stopping if enabled
        let (X_train, y_train, X_val, y_val) = if self.config.early_stopping.is_some() {
            let val_size = ((n_samples as f64) * self.config.validation_fraction) as usize;
            let train_size = n_samples - val_size;

            let X_train = X.slice(s![0..train_size, ..]).to_owned();
            let y_train = y.slice(s![0..train_size]).to_owned();
            let X_val = X.slice(s![train_size.., ..]).to_owned();
            let y_val = y.slice(s![train_size..]).to_owned();

            (X_train, y_train, Some(X_val), Some(y_val))
        } else {
            (X.clone(), y.clone(), None, None)
        };

        // One-vs-Rest approach: train one gradient boosting classifier per class
        let mut all_estimators = Vec::new();
        let mut train_scores = Vec::new();
        let mut validation_scores = Vec::new();
        let mut early_stopping_info = None;
        let mut n_estimators_trained = self.config.n_estimators;

        for (class_idx, &target_class) in classes.iter().enumerate() {
            // Create binary targets for current class
            let binary_targets = Array1::from_vec(
                y_train
                    .iter()
                    .map(|&label| if label == target_class { 1.0 } else { -1.0 })
                    .collect(),
            );

            // Initialize predictions with log-odds (logit of class prior)
            let prior = init_priors[class_idx];
            let init_pred = (prior / (1.0 - prior + f64::EPSILON)).ln();
            let mut current_predictions = Array1::from_elem(X_train.nrows(), init_pred);

            let mut class_estimators = Vec::new();

            let mut rng = match self.config.random_state {
                Some(seed) => seeded_rng(seed + class_idx as u64),
                None => seeded_rng(42),
            };

            // Early stopping variables
            let mut best_score = f64::INFINITY;
            let mut best_iteration = 0;
            let mut no_improve_count = 0;
            #[allow(unused_assignments)]
            let mut stopped_early = false;

            // Gradient boosting iterations
            for iteration in 0..self.config.n_estimators {
                // Compute negative gradients (residuals)
                let residuals = self.compute_residuals(&binary_targets, &current_predictions);

                // Subsample if needed
                let (X_subset, residuals_subset) = if self.config.subsample < 1.0 {
                    let n_subset = ((X_train.nrows() as f64) * self.config.subsample) as usize;
                    let mut indices: Vec<usize> = (0..X_train.nrows()).collect();
                    for i in 0..n_subset {
                        let j = rng.gen_range(i..X_train.nrows());
                        indices.swap(i, j);
                    }
                    indices.truncate(n_subset);

                    let X_subset = X_train.select(Axis(0), &indices);
                    let residuals_subset = residuals.select(Axis(0), &indices);
                    (X_subset, residuals_subset)
                } else {
                    (X_train.clone(), residuals)
                };

                // Fit base estimator on residuals
                let estimator = self
                    .base_estimator
                    .clone()
                    .fit(&X_subset, &residuals_subset)?;
                let predictions = estimator.predict(&X_train)?;

                // Update current predictions with learning rate
                for i in 0..X_train.nrows() {
                    current_predictions[i] += self.config.learning_rate * predictions[i];
                }

                class_estimators.push(estimator);

                // Compute training loss for monitoring
                if iteration % 10 == 0 {
                    let loss = self.compute_loss(&binary_targets, &current_predictions);
                    train_scores.push(loss);
                }

                // Early stopping check
                if let Some(ref early_config) = self.config.early_stopping {
                    if let (Some(ref X_val), Some(ref y_val)) = (&X_val, &y_val) {
                        let val_binary_targets = Array1::from_vec(
                            y_val
                                .iter()
                                .map(|&label| if label == target_class { 1.0 } else { -1.0 })
                                .collect(),
                        );

                        // Compute validation predictions
                        let mut val_predictions = Array1::from_elem(X_val.nrows(), init_pred);
                        for est in &class_estimators {
                            let pred = est.predict(X_val)?;
                            for i in 0..X_val.nrows() {
                                val_predictions[i] += self.config.learning_rate * pred[i];
                            }
                        }

                        let val_loss = self.compute_loss(&val_binary_targets, &val_predictions);
                        validation_scores.push(val_loss);

                        // Check for improvement
                        if val_loss < best_score - early_config.tol {
                            best_score = val_loss;
                            best_iteration = iteration;
                            no_improve_count = 0;
                        } else {
                            no_improve_count += 1;
                        }

                        // Stop if no improvement for too long
                        if no_improve_count >= early_config.n_iter_no_change {
                            stopped_early = true;
                            n_estimators_trained = iteration + 1;

                            early_stopping_info = Some(EarlyStoppingInfo {
                                best_iteration,
                                best_score,
                                n_iter_no_change: no_improve_count,
                                stopped_early,
                            });
                            break;
                        }
                    }
                }
            }

            all_estimators.push(class_estimators);
        }

        let trained_data = GradientBoostingTrainedData {
            estimators: all_estimators,
            classes: classes_array,
            n_classes,
            init_priors,
            feature_importances: None, // Could be computed from base estimators
            train_score: train_scores,
            validation_scores,
            n_estimators_trained,
            early_stopping_info,
        };

        Ok(GradientBoostingClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<C> GradientBoostingClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<f64>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<f64>> + Clone + Send + Sync,
{
    fn compute_residuals(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> Array1<f64> {
        compute_residuals_for_loss(&self.config.loss, y_true, y_pred)
    }

    fn compute_loss(&self, y_true: &Array1<f64>, y_pred: &Array1<f64>) -> f64 {
        compute_loss_for_loss(&self.config.loss, y_true, y_pred)
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>>
    for GradientBoostingClassifier<GradientBoostingTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<f64>> + Clone + Send + Sync,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let probabilities = self.predict_proba(X)?;
        let (n_samples, _) = probabilities.dim();

        let mut predictions = Array1::zeros(n_samples);
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            predictions[i] = self.base_estimator.classes[max_idx];
        }

        Ok(predictions)
    }
}

impl<T> GradientBoostingClassifier<GradientBoostingTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<f64>> + Clone + Send + Sync,
{
    /// Get the number of estimators actually trained
    pub fn n_estimators_trained(&self) -> usize {
        self.base_estimator.n_estimators_trained
    }

    /// Get early stopping information if available
    pub fn early_stopping_info(&self) -> Option<&EarlyStoppingInfo> {
        self.base_estimator.early_stopping_info.as_ref()
    }

    /// Get training scores
    pub fn train_scores(&self) -> &[f64] {
        &self.base_estimator.train_score
    }

    /// Get validation scores if available
    pub fn validation_scores(&self) -> &[f64] {
        &self.base_estimator.validation_scores
    }

    /// Compute feature importances from the trained estimators
    ///
    /// This computes the relative importance of each feature based on how often
    /// it's used in the tree splits across all classes and estimators.
    ///
    /// Note: This is a simplified importance calculation. For more accurate importance,
    /// the base estimators would need to provide their own feature importance scores.
    pub fn compute_feature_importances(&self, n_features: usize) -> Array1<f64> {
        let mut importances = Array1::zeros(n_features);
        let mut total_estimators = 0;

        // For each class, aggregate importance from all estimators
        for class_estimators in &self.base_estimator.estimators {
            total_estimators += class_estimators.len();

            // Since we don't have access to the base estimator's feature importance,
            // we'll use a uniform importance as a placeholder
            // In practice, this would query the base estimator's feature_importances
            for _estimator in class_estimators {
                for i in 0..n_features {
                    // Placeholder: assign equal importance to all features
                    // Real implementation would query estimator.feature_importances()
                    importances[i] += 1.0 / n_features as f64;
                }
            }
        }

        // Normalize by total number of estimators
        if total_estimators > 0 {
            for i in 0..n_features {
                importances[i] /= total_estimators as f64;
            }
        }

        importances
    }

    /// Set feature importances (useful when base estimators provide importance scores)
    pub fn set_feature_importances(&mut self, importances: Array1<f64>) {
        self.base_estimator.feature_importances = Some(importances);
    }

    /// Get feature importances if computed
    pub fn feature_importances(&self) -> Option<&Array1<f64>> {
        self.base_estimator.feature_importances.as_ref()
    }
}

impl<T> PredictProba<Array2<f64>, Array2<f64>>
    for GradientBoostingClassifier<GradientBoostingTrainedData<T>, Trained>
where
    T: Predict<Array2<f64>, Array1<f64>> + Clone + Send + Sync,
{
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, _) = X.dim();
        let trained_data = &self.base_estimator;

        // Compute decision function for each class
        let mut class_scores = Array2::zeros((n_samples, trained_data.n_classes));

        for (class_idx, class_estimators) in trained_data.estimators.iter().enumerate() {
            // Initialize with log-odds of class prior
            let prior = trained_data.init_priors[class_idx];
            let init_pred = (prior / (1.0 - prior + f64::EPSILON)).ln();

            for i in 0..n_samples {
                class_scores[[i, class_idx]] = init_pred;
            }

            // Add contributions from all boosting iterations
            for estimator in class_estimators {
                let predictions = estimator.predict(X)?;
                for i in 0..n_samples {
                    class_scores[[i, class_idx]] += self.config.learning_rate * predictions[i];
                }
            }
        }

        // Convert to probabilities using softmax
        let mut probabilities = Array2::zeros((n_samples, trained_data.n_classes));
        for i in 0..n_samples {
            let row = class_scores.row(i);
            let max_score = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let mut exp_sum = 0.0;
            for j in 0..trained_data.n_classes {
                let exp_val = (class_scores[[i, j]] - max_score).exp();
                probabilities[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            if exp_sum > 0.0 {
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] /= exp_sum;
                }
            } else {
                // Uniform distribution fallback
                for j in 0..trained_data.n_classes {
                    probabilities[[i, j]] = 1.0 / trained_data.n_classes as f64;
                }
            }
        }

        Ok(probabilities)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array2};

    // Mock base estimator for testing
    #[derive(Debug, Clone)]
    struct MockRegressor;

    #[derive(Debug, Clone)]
    struct MockRegressorTrained;

    impl Fit<Array2<f64>, Array1<f64>> for MockRegressor {
        type Fitted = MockRegressorTrained;

        fn fit(self, _X: &Array2<f64>, _y: &Array1<f64>) -> SklResult<Self::Fitted> {
            Ok(MockRegressorTrained)
        }
    }

    impl Predict<Array2<f64>, Array1<f64>> for MockRegressorTrained {
        fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
            // Return zeros for simplicity
            Ok(Array1::zeros(X.nrows()))
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_basic() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.5);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit Gradient Boosting");

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Check probabilities
        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (4, 2));

        // Probabilities should sum to 1
        for i in 0..4 {
            let sum: f64 = probabilities.row(i).sum();
            assert!((sum - 1.0).abs() < 1e-10, "Probabilities should sum to 1");
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_builder() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::builder(base_classifier)
            .n_estimators(10)
            .learning_rate(0.1)
            .max_depth(Some(5))
            .subsample(0.8)
            .loss(GradientBoostingLoss::Deviance)
            .random_state(Some(42))
            .build();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![0, 1, 2];

        let trained = gb.fit(&X, &y).expect("Failed to fit with builder");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_multiclass() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(3)
            .learning_rate(0.5);

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 2, 0, 1, 2];

        let trained = gb.fit(&X, &y).expect("Failed to fit multiclass");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 6);

        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (6, 3));

        // Check that each prediction corresponds to highest probability
        for i in 0..6 {
            let prob_row = probabilities.row(i);
            let max_prob_idx = prob_row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();
            // The prediction should correspond to the highest probability class
            let predicted_class = predictions[i];
            let expected_class = [0, 1, 2][max_prob_idx];
            assert_eq!(predicted_class, expected_class);
        }
    }

    #[test]
    fn test_gradient_boosting_clone() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier);

        let _cloned = gb.clone();
        // If clone works, this test passes
    }

    #[test]
    fn test_gradient_boosting_config_default() {
        let config = GradientBoostingConfig::default();
        assert_eq!(config.n_estimators, 100);
        assert_eq!(config.learning_rate, 0.1);
        assert_eq!(config.max_depth, Some(3));
        assert_eq!(config.min_samples_split, 2);
        assert_eq!(config.min_samples_leaf, 1);
        assert_eq!(config.subsample, 1.0);
        assert_eq!(config.random_state, None);
        assert_eq!(config.loss, GradientBoostingLoss::Deviance);
        assert_eq!(config.max_leaf_nodes, None);
        assert_eq!(config.n_jobs, None);
    }

    #[test]
    fn test_gradient_boosting_loss_default() {
        let loss = GradientBoostingLoss::default();
        assert_eq!(loss, GradientBoostingLoss::Deviance);
    }

    #[test]
    fn test_gradient_boosting_estimator_trait() {
        let classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(classifier);

        let config = gb.config();
        assert_eq!(config.n_estimators, 100);
        assert_eq!(config.learning_rate, 0.1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_insufficient_classes() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier);

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // Only one class

        let result = gb.fit(&X, &y);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("Need at least 2 classes"));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_loss_functions() {
        let base_classifier = MockRegressor;

        // Test Deviance loss
        let gb_deviance = GradientBoostingClassifier::new(base_classifier.clone())
            .loss(GradientBoostingLoss::Deviance)
            .n_estimators(3);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb_deviance
            .fit(&X, &y)
            .expect("Failed to fit with Deviance loss");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Test Exponential loss
        let gb_exponential = GradientBoostingClassifier::new(base_classifier)
            .loss(GradientBoostingLoss::Exponential)
            .n_estimators(3);

        let trained = gb_exponential
            .fit(&X, &y)
            .expect("Failed to fit with Exponential loss");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_subsample() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(3)
            .subsample(0.5)
            .random_state(Some(42));

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
            [7.0, 8.0],
            [8.0, 9.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit with subsampling");
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 8);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_empty_dataset() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier);

        let X = Array2::<f64>::zeros((0, 2));
        let y = Array1::<i32>::zeros(0);

        let result = gb.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_reproducibility() {
        let base_classifier = MockRegressor;

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        // Train two models with same random state
        let gb1 = GradientBoostingClassifier::new(base_classifier.clone())
            .n_estimators(5)
            .random_state(Some(42))
            .subsample(0.8);
        let trained1 = gb1.fit(&X, &y).expect("Failed to fit first model");

        let gb2 = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .random_state(Some(42))
            .subsample(0.8);
        let trained2 = gb2.fit(&X, &y).expect("Failed to fit second model");

        let pred1 = trained1.predict(&X).expect("Failed to predict first");
        let pred2 = trained2.predict(&X).expect("Failed to predict second");

        // Results should be identical with same random state
        assert_eq!(pred1.to_vec(), pred2.to_vec());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_detailed_training() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1);

        let X = array![
            [1.0, 1.0],
            [1.1, 1.1],
            [1.2, 1.2], // Class 0
            [3.0, 3.0],
            [3.1, 3.1],
            [3.2, 3.2], // Class 1
            [5.0, 5.0],
            [5.1, 5.1],
            [5.2, 5.2] // Class 2
        ];
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];

        let trained = gb.fit(&X, &y).expect("Failed to fit detailed training");

        // Test prediction consistency
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 9);

        // Test probability consistency
        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (9, 3));

        // Each row should sum to 1
        for i in 0..9 {
            let sum: f64 = probabilities.row(i).sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "Probabilities row {} should sum to 1, got {}",
                i,
                sum
            );
        }

        // Check that trained data contains expected structures
        let trained_data = &trained.base_estimator;
        assert_eq!(trained_data.n_classes, 3);
        assert_eq!(trained_data.classes.len(), 3);
        assert_eq!(trained_data.estimators.len(), 3); // One per class

        // Each class should have n_estimators models
        for class_estimators in &trained_data.estimators {
            assert_eq!(class_estimators.len(), 5);
        }
    }

    #[test]
    fn test_gradient_boosting_loss_computation() {
        let base_classifier = MockRegressor;
        let gb = GradientBoostingClassifier::new(base_classifier);

        // Test residual computation for different loss functions
        let y_true = array![1.0, -1.0, 1.0, -1.0];
        let y_pred = array![0.5, -0.5, 0.8, -0.2];

        // Test with Deviance loss
        let gb_deviance = gb.clone().loss(GradientBoostingLoss::Deviance);
        let residuals_deviance = gb_deviance.compute_residuals(&y_true, &y_pred);
        assert_eq!(residuals_deviance.len(), 4);

        // Test with Exponential loss
        let gb_exponential = gb.loss(GradientBoostingLoss::Exponential);
        let residuals_exponential = gb_exponential.compute_residuals(&y_true, &y_pred);
        assert_eq!(residuals_exponential.len(), 4);

        // Residuals should be different for different loss functions
        let diff: f64 = residuals_deviance
            .iter()
            .zip(residuals_exponential.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(
            diff > 1e-10,
            "Different loss functions should produce different residuals"
        );
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_configuration_methods() {
        let base_classifier = MockRegressor;

        // Test all configuration methods
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(20)
            .learning_rate(0.05)
            .max_depth(Some(4))
            .subsample(0.9)
            .loss(GradientBoostingLoss::Exponential)
            .random_state(Some(123));

        // Verify configuration through training
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit configured model");

        // Verify estimator count per class
        let trained_data = &trained.base_estimator;
        for class_estimators in &trained_data.estimators {
            assert_eq!(class_estimators.len(), 20); // Should match n_estimators
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_early_stopping() {
        let base_classifier = MockRegressor;

        // Configure early stopping
        let early_stopping = EarlyStoppingConfig {
            n_iter_no_change: 5,
            tol: 1e-4,
            scoring: EarlyStoppingScoring::Loss,
        };

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(100)
            .learning_rate(0.1)
            .early_stopping(early_stopping)
            .validation_fraction(0.3)
            .random_state(Some(42));

        // Create a larger dataset for proper train/validation split
        let X = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3],
            [1.4, 2.4],
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3],
            [3.4, 4.4],
            [5.0, 6.0],
            [5.1, 6.1],
            [5.2, 6.2],
            [5.3, 6.3],
            [5.4, 6.4],
            [7.0, 8.0],
            [7.1, 8.1],
            [7.2, 8.2],
            [7.3, 8.3],
            [7.4, 8.4]
        ];
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 0, 0, 0, 0, 0];

        let trained = gb.fit(&X, &y).expect("Failed to fit with early stopping");

        // Check that early stopping info is present
        let trained_data = &trained.base_estimator;
        assert!(trained_data.validation_scores.len() > 0);

        // Verify that the model was trained
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 20);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_warm_start() {
        let base_classifier = MockRegressor;

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(10)
            .learning_rate(0.1)
            .warm_start(true)
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit with warm start");

        // Check that warm start is configured
        assert!(trained.config.warm_start);

        // Verify that the model was trained
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_builder_with_new_features() {
        let base_classifier = MockRegressor;

        let early_stopping = EarlyStoppingConfig::default();

        let gb = GradientBoostingClassifier::builder(base_classifier)
            .n_estimators(15)
            .learning_rate(0.05)
            .early_stopping(early_stopping)
            .warm_start(true)
            .validation_fraction(0.2)
            .random_state(Some(123))
            .build();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0], [5.0, 6.0]];
        let y = array![0, 1, 0, 1, 0];

        let trained = gb.fit(&X, &y).expect("Failed to fit with builder");

        // Verify configuration
        assert!(trained.config.warm_start);
        assert!(trained.config.early_stopping.is_some());
        assert_eq!(trained.config.validation_fraction, 0.2);

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 5);
    }

    #[test]
    fn test_early_stopping_config_default() {
        let config = EarlyStoppingConfig::default();
        assert_eq!(config.n_iter_no_change, 10);
        assert_eq!(config.tol, 1e-4);
        assert_eq!(config.scoring, EarlyStoppingScoring::Loss);
    }

    #[test]
    fn test_gradient_boosting_config_with_new_fields() {
        let config = GradientBoostingConfig::default();
        assert_eq!(config.early_stopping, None);
        assert_eq!(config.warm_start, false);
        assert_eq!(config.validation_fraction, 0.1);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_warm_start_config() {
        let base_classifier = MockRegressor;

        // Test warm start configuration
        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1)
            .warm_start(true)
            .random_state(Some(42));

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 0, 1, 2, 2];

        let trained = gb.fit(&X, &y).expect("Failed to fit initial model");

        // Check that warm start is enabled in config
        assert!(trained.config.warm_start);

        // Check initial state tracking
        let initial_n_estimators = trained.n_estimators_trained();
        assert_eq!(initial_n_estimators, 5);

        // Verify predictions work
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 6);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_accessor_methods() {
        let base_classifier = MockRegressor;

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(10)
            .learning_rate(0.1)
            .warm_start(true)
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit model");

        // Test accessor methods
        assert_eq!(trained.n_estimators_trained(), 10);
        assert!(trained.train_scores().len() > 0);
        assert_eq!(trained.validation_scores().len(), 0); // No early stopping
        assert!(trained.early_stopping_info().is_none());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_warm_start_with_early_stopping() {
        let base_classifier = MockRegressor;

        let early_stopping = EarlyStoppingConfig {
            n_iter_no_change: 3,
            tol: 1e-4,
            scoring: EarlyStoppingScoring::Loss,
        };

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1)
            .warm_start(true)
            .early_stopping(early_stopping)
            .validation_fraction(0.3)
            .random_state(Some(42));

        // Create a larger dataset for proper train/validation split
        let X = array![
            [1.0, 2.0],
            [1.1, 2.1],
            [1.2, 2.2],
            [1.3, 2.3],
            [1.4, 2.4],
            [3.0, 4.0],
            [3.1, 4.1],
            [3.2, 4.2],
            [3.3, 4.3],
            [3.4, 4.4]
        ];
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit initial model");

        // Check that both warm start and early stopping are configured
        assert!(trained.config.warm_start);
        assert!(trained.config.early_stopping.is_some());

        // Check that validation scores were recorded
        assert!(trained.validation_scores().len() > 0);

        // Verify predictions still work
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 10);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_multinomial_loss() {
        let base_classifier = MockRegressor;

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1)
            .loss(GradientBoostingLoss::Multinomial)
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 0];

        let trained = gb.fit(&X, &y).expect("Failed to fit with multinomial loss");

        // Verify that multinomial loss is configured
        assert_eq!(trained.config.loss, GradientBoostingLoss::Multinomial);

        // Test predictions
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Test probabilities
        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (4, 3));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_huber_loss() {
        let base_classifier = MockRegressor;

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1)
            .loss(GradientBoostingLoss::Huber { delta: 1.0 })
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit with Huber loss");

        // Verify that Huber loss is configured
        assert_eq!(
            trained.config.loss,
            GradientBoostingLoss::Huber { delta: 1.0 }
        );

        // Test predictions
        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_feature_importances() {
        let base_classifier = MockRegressor;

        let gb = GradientBoostingClassifier::new(base_classifier)
            .n_estimators(5)
            .learning_rate(0.1)
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = gb.fit(&X, &y).expect("Failed to fit model");

        // Compute feature importances
        let importances = trained.compute_feature_importances(2);
        assert_eq!(importances.len(), 2);

        // Since we use uniform importance, each feature should have equal importance
        assert!((importances[0] - 0.5).abs() < 1e-10);
        assert!((importances[1] - 0.5).abs() < 1e-10);

        // Test setting custom feature importances
        let mut trained_mut = trained;
        let custom_importances = array![0.7, 0.3];
        trained_mut.set_feature_importances(custom_importances.clone());

        let retrieved_importances = trained_mut.feature_importances().unwrap();
        assert_eq!(retrieved_importances[0], 0.7);
        assert_eq!(retrieved_importances[1], 0.3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_boosting_all_new_loss_functions() {
        let base_classifier = MockRegressor;
        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        // Test all loss functions
        let loss_functions = vec![
            GradientBoostingLoss::Deviance,
            GradientBoostingLoss::Exponential,
            GradientBoostingLoss::Multinomial,
            GradientBoostingLoss::Huber { delta: 1.5 },
        ];

        for loss in loss_functions {
            let gb = GradientBoostingClassifier::new(base_classifier.clone())
                .n_estimators(3)
                .learning_rate(0.1)
                .loss(loss.clone())
                .random_state(Some(42));

            let trained = gb
                .fit(&X, &y)
                .expect(&format!("Failed to fit with loss: {:?}", loss));
            let predictions = trained
                .predict(&X)
                .expect(&format!("Failed to predict with loss: {:?}", loss));
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_loss_function_computations() {
        // Test the helper functions for loss computation
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![0.8, 0.2, 0.7, 0.1];

        // Test deviance loss
        let residuals =
            compute_residuals_for_loss(&GradientBoostingLoss::Deviance, &y_true, &y_pred);
        assert_eq!(residuals.len(), 4);

        // Test exponential loss
        let residuals =
            compute_residuals_for_loss(&GradientBoostingLoss::Exponential, &y_true, &y_pred);
        assert_eq!(residuals.len(), 4);

        // Test multinomial loss
        let residuals =
            compute_residuals_for_loss(&GradientBoostingLoss::Multinomial, &y_true, &y_pred);
        assert_eq!(residuals.len(), 4);

        // Test Huber loss
        let residuals = compute_residuals_for_loss(
            &GradientBoostingLoss::Huber { delta: 1.0 },
            &y_true,
            &y_pred,
        );
        assert_eq!(residuals.len(), 4);

        // Test loss computation
        let deviance_loss =
            compute_loss_for_loss(&GradientBoostingLoss::Deviance, &y_true, &y_pred);
        assert!(deviance_loss >= 0.0);

        let huber_loss = compute_loss_for_loss(
            &GradientBoostingLoss::Huber { delta: 1.0 },
            &y_true,
            &y_pred,
        );
        assert!(huber_loss >= 0.0);
    }
}
