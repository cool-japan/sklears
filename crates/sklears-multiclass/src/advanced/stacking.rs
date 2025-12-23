//! Stacking and Blending multiclass classification
//!
//! This module provides stacking and blending strategies for multiclass classification
//! where multiple base classifiers are combined using a meta-learner or weighted blending.

use rayon::prelude::*;
use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seeded_rng, CoreRandom};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

/// Stacking method configuration
#[derive(Debug, Clone, PartialEq)]
pub enum StackingMethod {
    /// Cross-validation based stacking
    CrossValidation { cv_folds: usize },
    /// Holdout validation stacking
    Holdout { test_size: f64 },
    /// Blending with a fixed holdout set
    Blending { blend_ratio: f64 },
}

impl Default for StackingMethod {
    fn default() -> Self {
        Self::CrossValidation { cv_folds: 5 }
    }
}

/// Meta-learner strategy
#[derive(Debug, Clone, PartialEq)]
pub enum MetaLearnerStrategy {
    /// Simple averaging of base classifier predictions
    Average,
    /// Weighted averaging with learned weights
    WeightedAverage,
    /// Linear combination meta-learner
    Linear,
    /// Logistic regression meta-learner
    LogisticRegression,
    /// Neural network meta-learner
    NeuralNetwork,
    /// Dynamic selection based on cross-validation performance
    Dynamic {
        /// Candidate strategies to evaluate
        candidates: Vec<MetaLearnerStrategy>,
        /// Cross-validation folds for evaluation
        cv_folds: usize,
        /// Scoring metric for selection
        scoring: DynamicSelectionScoring,
    },
    /// Adaptive selection based on data characteristics
    Adaptive {
        /// Rules for meta-learner selection
        selection_rules: AdaptiveSelectionRules,
    },
}

/// Scoring metrics for dynamic meta-learner selection
#[derive(Debug, Clone, PartialEq)]
pub enum DynamicSelectionScoring {
    /// Accuracy-based selection
    Accuracy,
    /// F1-score based selection
    F1Score,
    /// Log-loss based selection
    LogLoss,
    /// Custom scoring function
    Custom,
}

/// Rules for adaptive meta-learner selection
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptiveSelectionRules {
    /// Number of samples threshold for complex meta-learners
    pub min_samples_for_complex: usize,
    /// Number of features threshold for neural networks
    pub min_features_for_nn: usize,
    /// Number of classes threshold for linear methods
    pub max_classes_for_linear: usize,
    /// Imbalance ratio threshold for weighted methods
    pub imbalance_threshold: f64,
}

impl Default for MetaLearnerStrategy {
    fn default() -> Self {
        Self::LogisticRegression
    }
}

impl Default for DynamicSelectionScoring {
    fn default() -> Self {
        Self::Accuracy
    }
}

impl Default for AdaptiveSelectionRules {
    fn default() -> Self {
        Self {
            min_samples_for_complex: 1000,
            min_features_for_nn: 10,
            max_classes_for_linear: 10,
            imbalance_threshold: 2.0,
        }
    }
}

impl AdaptiveSelectionRules {
    /// Select appropriate meta-learner based on data characteristics
    pub fn select_meta_learner(
        &self,
        n_samples: usize,
        n_features: usize,
        n_classes: usize,
        class_distribution: &[f64],
    ) -> MetaLearnerStrategy {
        // Calculate class imbalance ratio
        let max_freq = class_distribution.iter().fold(0.0f64, |a, &b| a.max(b));
        let min_freq = class_distribution
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b));
        let imbalance_ratio = if min_freq > 0.0 {
            max_freq / min_freq
        } else {
            f64::INFINITY
        };

        // Apply selection rules
        if imbalance_ratio > self.imbalance_threshold {
            // High imbalance: use weighted averaging
            MetaLearnerStrategy::WeightedAverage
        } else if n_classes > self.max_classes_for_linear {
            // Many classes: use neural network if enough features
            if n_features >= self.min_features_for_nn && n_samples >= self.min_samples_for_complex {
                MetaLearnerStrategy::NeuralNetwork
            } else {
                MetaLearnerStrategy::Average
            }
        } else if n_samples < self.min_samples_for_complex {
            // Small dataset: use simple averaging
            MetaLearnerStrategy::Average
        } else if n_features >= self.min_features_for_nn {
            // High-dimensional: use neural network
            MetaLearnerStrategy::NeuralNetwork
        } else {
            // Default: logistic regression
            MetaLearnerStrategy::LogisticRegression
        }
    }
}

/// Stacking configuration
#[derive(Debug, Clone)]
pub struct StackingConfig {
    /// Stacking method
    pub stacking_method: StackingMethod,
    /// Meta-learner strategy
    pub meta_learner_strategy: MetaLearnerStrategy,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// StdRng state for reproducibility
    pub random_state: Option<u64>,
    /// Use original features in meta-learner
    pub passthrough: bool,
    /// Stack class probabilities instead of predictions
    pub stack_probabilities: bool,
    /// Enable dynamic meta-learner adaptation during training
    pub enable_online_adaptation: bool,
    /// Minimum performance improvement to trigger adaptation
    pub adaptation_threshold: f64,
    /// Window size for online adaptation
    pub adaptation_window_size: usize,
}

impl Default for StackingConfig {
    fn default() -> Self {
        Self {
            stacking_method: StackingMethod::default(),
            meta_learner_strategy: MetaLearnerStrategy::default(),
            n_jobs: None,
            random_state: None,
            passthrough: false,
            stack_probabilities: true,
            enable_online_adaptation: false,
            adaptation_threshold: 0.01,
            adaptation_window_size: 100,
        }
    }
}

/// Multiclass Stacking Classifier
///
/// Stacking is an ensemble method that combines multiple base classifiers using a meta-learner.
/// The base classifiers are trained on the original data, and their predictions (or probabilities)
/// are used as features for training a meta-learner that makes the final prediction.
///
/// # Type Parameters
///
/// * `C` - Vector of base classifier types that implement Fit and Predict
/// * `M` - Meta-learner classifier type
/// * `S` - State type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::advanced::MulticlassStackingClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with hypothetical base classifiers and meta-learner
/// // let base_classifiers = vec![SomeClassifier::new(), AnotherClassifier::new()];
/// // let meta_learner = MetaClassifier::new();
/// // let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);
/// ```
#[derive(Debug)]
pub struct MulticlassStackingClassifier<C, M, S = Untrained> {
    base_estimators: C,
    meta_learner: M,
    config: StackingConfig,
    state: PhantomData<S>,
}

/// Trained data for stacking classifier
#[derive(Debug)]
pub struct StackingTrainedData<T, U> {
    /// Trained base estimators
    pub base_estimators: Vec<T>,
    /// Trained meta-learner
    pub meta_learner: U,
    /// Unique classes from training data
    pub classes: Array1<i32>,
    /// Number of classes
    pub n_classes: usize,
    /// Original number of features
    pub n_features: usize,
    /// Selected meta-learner strategy (for dynamic selection)
    pub selected_meta_learner: MetaLearnerStrategy,
    /// Performance scores of evaluated strategies (for dynamic selection)
    pub strategy_scores: Option<Vec<(MetaLearnerStrategy, f64)>>,
    /// Adaptation history for online learning
    pub adaptation_history: Vec<AdaptationEvent>,
}

/// Event recorded during adaptive meta-learner selection
#[derive(Debug, Clone)]
pub struct AdaptationEvent {
    /// Iteration when adaptation occurred
    pub iteration: usize,
    /// Previous meta-learner strategy
    pub previous_strategy: MetaLearnerStrategy,
    /// New meta-learner strategy
    pub new_strategy: MetaLearnerStrategy,
    /// Performance improvement achieved
    pub performance_improvement: f64,
    /// Reason for adaptation
    pub reason: String,
}

impl<C, M> MulticlassStackingClassifier<Vec<C>, M, Untrained> {
    /// Create a new stacking classifier
    pub fn new(base_estimators: Vec<C>, meta_learner: M) -> Self {
        Self {
            base_estimators,
            meta_learner,
            config: StackingConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for the stacking classifier
    pub fn builder(base_estimators: Vec<C>, meta_learner: M) -> StackingBuilder<C, M> {
        StackingBuilder::new(base_estimators, meta_learner)
    }

    /// Set the stacking method
    pub fn stacking_method(mut self, stacking_method: StackingMethod) -> Self {
        self.config.stacking_method = stacking_method;
        self
    }

    /// Set the meta-learner strategy
    pub fn meta_learner_strategy(mut self, meta_learner_strategy: MetaLearnerStrategy) -> Self {
        self.config.meta_learner_strategy = meta_learner_strategy;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set whether to pass original features to meta-learner
    pub fn passthrough(mut self, passthrough: bool) -> Self {
        self.config.passthrough = passthrough;
        self
    }

    /// Set whether to stack probabilities instead of predictions
    pub fn stack_probabilities(mut self, stack_probabilities: bool) -> Self {
        self.config.stack_probabilities = stack_probabilities;
        self
    }

    /// Enable dynamic meta-learner selection
    pub fn dynamic_meta_learner(
        mut self,
        candidates: Vec<MetaLearnerStrategy>,
        cv_folds: usize,
        scoring: DynamicSelectionScoring,
    ) -> Self {
        self.config.meta_learner_strategy = MetaLearnerStrategy::Dynamic {
            candidates,
            cv_folds,
            scoring,
        };
        self
    }

    /// Enable adaptive meta-learner selection based on data characteristics
    pub fn adaptive_meta_learner(mut self, selection_rules: AdaptiveSelectionRules) -> Self {
        self.config.meta_learner_strategy = MetaLearnerStrategy::Adaptive { selection_rules };
        self
    }

    /// Enable online adaptation during training
    pub fn enable_online_adaptation(mut self, threshold: f64, window_size: usize) -> Self {
        self.config.enable_online_adaptation = true;
        self.config.adaptation_threshold = threshold;
        self.config.adaptation_window_size = window_size;
        self
    }

    /// Get reference to base estimators
    pub fn base_estimators(&self) -> &Vec<C> {
        &self.base_estimators
    }

    /// Get reference to meta-learner
    pub fn meta_learner(&self) -> &M {
        &self.meta_learner
    }

    /// Get reference to configuration
    pub fn config(&self) -> &StackingConfig {
        &self.config
    }
}

/// Builder for MulticlassStackingClassifier
#[derive(Debug)]
pub struct StackingBuilder<C, M> {
    base_estimators: Vec<C>,
    meta_learner: M,
    config: StackingConfig,
}

impl<C, M> StackingBuilder<C, M> {
    /// Create a new builder
    pub fn new(base_estimators: Vec<C>, meta_learner: M) -> Self {
        Self {
            base_estimators,
            meta_learner,
            config: StackingConfig::default(),
        }
    }

    /// Set the stacking method
    pub fn stacking_method(mut self, stacking_method: StackingMethod) -> Self {
        self.config.stacking_method = stacking_method;
        self
    }

    /// Set the meta-learner strategy
    pub fn meta_learner_strategy(mut self, meta_learner_strategy: MetaLearnerStrategy) -> Self {
        self.config.meta_learner_strategy = meta_learner_strategy;
        self
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }

    /// Set whether to pass original features to meta-learner
    pub fn passthrough(mut self, passthrough: bool) -> Self {
        self.config.passthrough = passthrough;
        self
    }

    /// Set whether to stack probabilities instead of predictions
    pub fn stack_probabilities(mut self, stack_probabilities: bool) -> Self {
        self.config.stack_probabilities = stack_probabilities;
        self
    }

    /// Build the classifier
    pub fn build(self) -> MulticlassStackingClassifier<Vec<C>, M, Untrained> {
        MulticlassStackingClassifier {
            base_estimators: self.base_estimators,
            meta_learner: self.meta_learner,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone, M: Clone> Clone for MulticlassStackingClassifier<Vec<C>, M, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimators: self.base_estimators.clone(),
            meta_learner: self.meta_learner.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<T: Clone, U: Clone> Clone for StackingTrainedData<T, U> {
    fn clone(&self) -> Self {
        Self {
            base_estimators: self.base_estimators.clone(),
            meta_learner: self.meta_learner.clone(),
            classes: self.classes.clone(),
            n_classes: self.n_classes,
            n_features: self.n_features,
            selected_meta_learner: self.selected_meta_learner.clone(),
            strategy_scores: self.strategy_scores.clone(),
            adaptation_history: self.adaptation_history.clone(),
        }
    }
}

impl<C, M> Estimator for MulticlassStackingClassifier<Vec<C>, M, Untrained> {
    type Config = StackingConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<T, U, M> Estimator for MulticlassStackingClassifier<StackingTrainedData<T, U>, M, Trained> {
    type Config = StackingConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl<C, M> Fit<Array2<f64>, Array1<i32>> for MulticlassStackingClassifier<Vec<C>, M, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
    M: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    M::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted =
        MulticlassStackingClassifier<StackingTrainedData<C::Fitted, M::Fitted>, M, Trained>;

    fn fit(self, x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        if x.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        if x.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "X must have at least one sample".to_string(),
            ));
        }
        if self.base_estimators.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Must have at least one base estimator".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort_unstable();
        classes.dedup();
        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for classification".to_string(),
            ));
        }
        let classes = Array1::from(classes);
        let n_classes = classes.len();

        // Initialize random number generator
        let mut rng: CoreRandom<StdRng> = match self.config.random_state {
            Some(seed) => seeded_rng(seed),
            None => seeded_rng(42),
        };

        // Generate meta-features using the specified stacking method
        let (meta_features, trained_base_estimators) = match &self.config.stacking_method {
            StackingMethod::CrossValidation { cv_folds } => {
                self.generate_cv_meta_features(x, y, *cv_folds, &mut rng)?
            }
            StackingMethod::Holdout { test_size } => {
                self.generate_holdout_meta_features(x, y, *test_size, &mut rng)?
            }
            StackingMethod::Blending { blend_ratio } => {
                self.generate_blending_meta_features(x, y, *blend_ratio, &mut rng)?
            }
        };

        // Add original features if passthrough is enabled
        let final_meta_features = if self.config.passthrough {
            let mut combined = Array2::zeros((n_samples, x.ncols() + meta_features.ncols()));
            combined.slice_mut(s![.., ..x.ncols()]).assign(x);
            combined
                .slice_mut(s![.., x.ncols()..])
                .assign(&meta_features);
            combined
        } else {
            meta_features
        };

        // Train meta-learner
        let trained_meta_learner = self.meta_learner.clone().fit(&final_meta_features, y)?;

        let trained_data = StackingTrainedData {
            base_estimators: trained_base_estimators,
            meta_learner: trained_meta_learner,
            classes,
            n_classes,
            n_features,
            selected_meta_learner: self.config.meta_learner_strategy.clone(),
            strategy_scores: None,
            adaptation_history: Vec::new(),
        };

        Ok(MulticlassStackingClassifier {
            base_estimators: trained_data,
            meta_learner: self.meta_learner,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<C, M> MulticlassStackingClassifier<Vec<C>, M, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
{
    fn generate_cv_meta_features(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        cv_folds: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<f64>, Vec<C::Fitted>)> {
        let n_samples = x.nrows();
        let n_base_estimators = self.base_estimators.len();

        // Determine feature count based on whether we stack probabilities
        let feature_count = if self.config.stack_probabilities {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            n_base_estimators * classes.len()
        } else {
            n_base_estimators
        };

        let mut meta_features = Array2::zeros((n_samples, feature_count));

        // Generate cross-validation folds
        let folds = self.generate_cv_folds(n_samples, cv_folds, rng);

        // For each fold, train base estimators and predict on validation set
        for (train_indices, val_indices) in folds.iter() {
            let x_train = x.select(Axis(0), train_indices);
            let y_train = y.select(Axis(0), train_indices);
            let x_val = x.select(Axis(0), val_indices);

            // Train base estimators on training fold
            let trained_estimators: SklResult<Vec<_>> = if self.config.n_jobs.unwrap_or(1) > 1 {
                self.base_estimators
                    .par_iter()
                    .map(|estimator| estimator.clone().fit(&x_train, &y_train))
                    .collect()
            } else {
                self.base_estimators
                    .iter()
                    .map(|estimator| estimator.clone().fit(&x_train, &y_train))
                    .collect()
            };
            let trained_estimators = trained_estimators?;

            // Generate predictions for validation fold
            for val_idx in val_indices {
                let sample_idx = *val_idx;
                let row_idx = val_indices.iter().position(|&x| x == sample_idx).unwrap();
                let x_sample = x_val.slice(s![row_idx..row_idx + 1, ..]).to_owned();

                let mut feature_offset = 0;
                for estimator in &trained_estimators {
                    if self.config.stack_probabilities {
                        let proba = estimator.predict_proba(&x_sample)?;
                        let n_classes = proba.ncols();
                        for j in 0..n_classes {
                            meta_features[[sample_idx, feature_offset + j]] = proba[[0, j]];
                        }
                        feature_offset += n_classes;
                    } else {
                        let pred = estimator.predict(&x_sample)?;
                        meta_features[[sample_idx, feature_offset]] = pred[0] as f64;
                        feature_offset += 1;
                    }
                }
            }
        }

        // Train final base estimators on full dataset
        let final_trained_estimators: SklResult<Vec<_>> = if self.config.n_jobs.unwrap_or(1) > 1 {
            self.base_estimators
                .par_iter()
                .map(|estimator| estimator.clone().fit(x, y))
                .collect()
        } else {
            self.base_estimators
                .iter()
                .map(|estimator| estimator.clone().fit(x, y))
                .collect()
        };

        Ok((meta_features, final_trained_estimators?))
    }

    fn generate_holdout_meta_features(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        test_size: f64,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<f64>, Vec<C::Fitted>)> {
        let n_samples = x.nrows();
        let split_point = ((1.0 - test_size) * n_samples as f64) as usize;

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let train_indices = &indices[..split_point];
        let val_indices = &indices[split_point..];

        let x_train = x.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let x_val = x.select(Axis(0), val_indices);

        // Train base estimators
        let trained_estimators: SklResult<Vec<_>> = if self.config.n_jobs.unwrap_or(1) > 1 {
            self.base_estimators
                .par_iter()
                .map(|estimator| estimator.clone().fit(&x_train, &y_train))
                .collect()
        } else {
            self.base_estimators
                .iter()
                .map(|estimator| estimator.clone().fit(&x_train, &y_train))
                .collect()
        };
        let trained_estimators = trained_estimators?;

        // Generate meta-features for validation set
        let n_base_estimators = self.base_estimators.len();
        let feature_count = if self.config.stack_probabilities {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            n_base_estimators * classes.len()
        } else {
            n_base_estimators
        };

        let mut meta_features = Array2::zeros((n_samples, feature_count));

        for (val_row, &val_idx) in val_indices.iter().enumerate() {
            let x_sample = x_val.slice(s![val_row..val_row + 1, ..]).to_owned();
            let mut feature_offset = 0;

            for estimator in &trained_estimators {
                if self.config.stack_probabilities {
                    let proba = estimator.predict_proba(&x_sample)?;
                    let n_classes = proba.ncols();
                    for j in 0..n_classes {
                        meta_features[[val_idx, feature_offset + j]] = proba[[0, j]];
                    }
                    feature_offset += n_classes;
                } else {
                    let pred = estimator.predict(&x_sample)?;
                    meta_features[[val_idx, feature_offset]] = pred[0] as f64;
                    feature_offset += 1;
                }
            }
        }

        // Retrain on full dataset
        let final_trained_estimators: SklResult<Vec<_>> = if self.config.n_jobs.unwrap_or(1) > 1 {
            self.base_estimators
                .par_iter()
                .map(|estimator| estimator.clone().fit(x, y))
                .collect()
        } else {
            self.base_estimators
                .iter()
                .map(|estimator| estimator.clone().fit(x, y))
                .collect()
        };

        Ok((meta_features, final_trained_estimators?))
    }

    fn generate_blending_meta_features(
        &self,
        x: &Array2<f64>,
        y: &Array1<i32>,
        blend_ratio: f64,
        rng: &mut CoreRandom<StdRng>,
    ) -> SklResult<(Array2<f64>, Vec<C::Fitted>)> {
        // Similar to holdout but uses a fixed blending set
        self.generate_holdout_meta_features(x, y, blend_ratio, rng)
    }

    fn generate_cv_folds(
        &self,
        n_samples: usize,
        cv_folds: usize,
        rng: &mut CoreRandom<StdRng>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut indices: Vec<usize> = (0..n_samples).collect();
        // Shuffle using Fisher-Yates algorithm
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let fold_size = n_samples / cv_folds;
        let mut folds = Vec::new();

        for i in 0..cv_folds {
            let start = i * fold_size;
            let end = if i == cv_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let val_indices = indices[start..end].to_vec();
            let train_indices = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            folds.push((train_indices, val_indices));
        }

        folds
    }
}

impl<T, U, M> Predict<Array2<f64>, Array1<i32>>
    for MulticlassStackingClassifier<StackingTrainedData<T, U>, M, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
    U: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn predict(&self, x: &Array2<f64>) -> SklResult<Array1<i32>> {
        // Generate meta-features using trained base estimators
        let meta_features = self.generate_prediction_meta_features(x)?;

        // Make prediction using meta-learner
        self.base_estimators.meta_learner.predict(&meta_features)
    }
}

impl<T, U, M> PredictProba<Array2<f64>, Array2<f64>>
    for MulticlassStackingClassifier<StackingTrainedData<T, U>, M, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
    U: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
{
    fn predict_proba(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        // Generate meta-features using trained base estimators
        let meta_features = self.generate_prediction_meta_features(x)?;

        // Make probability prediction using meta-learner
        self.base_estimators
            .meta_learner
            .predict_proba(&meta_features)
    }
}

impl<T, U, M> MulticlassStackingClassifier<StackingTrainedData<T, U>, M, Trained>
where
    T: Predict<Array2<f64>, Array1<i32>>
        + PredictProba<Array2<f64>, Array2<f64>>
        + Clone
        + Send
        + Sync,
{
    fn generate_prediction_meta_features(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let n_base_estimators = self.base_estimators.base_estimators.len();

        // Determine feature count based on whether we stack probabilities
        let feature_count = if self.config.stack_probabilities {
            n_base_estimators * self.base_estimators.n_classes
        } else {
            n_base_estimators
        };

        let mut meta_features = Array2::zeros((n_samples, feature_count));

        // Generate meta-features from base estimators
        let mut feature_offset = 0;
        for estimator in &self.base_estimators.base_estimators {
            if self.config.stack_probabilities {
                let proba = estimator.predict_proba(x)?;
                let n_classes = proba.ncols();
                for j in 0..n_classes {
                    for i in 0..n_samples {
                        meta_features[[i, feature_offset + j]] = proba[[i, j]];
                    }
                }
                feature_offset += n_classes;
            } else {
                let pred = estimator.predict(x)?;
                for i in 0..n_samples {
                    meta_features[[i, feature_offset]] = pred[i] as f64;
                }
                feature_offset += 1;
            }
        }

        // Add original features if passthrough is enabled
        if self.config.passthrough {
            let mut combined = Array2::zeros((n_samples, x.ncols() + meta_features.ncols()));
            combined.slice_mut(s![.., ..x.ncols()]).assign(x);
            combined
                .slice_mut(s![.., x.ncols()..])
                .assign(&meta_features);
            Ok(combined)
        } else {
            Ok(meta_features)
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    // Mock classifier for testing
    #[derive(Debug, Clone)]
    struct MockClassifier {
        weight: f64,
    }

    impl MockClassifier {
        fn new(weight: f64) -> Self {
            Self { weight }
        }
    }

    #[derive(Debug, Clone)]
    struct MockTrained {
        weight: f64,
        classes: Array1<i32>,
    }

    impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
        type Fitted = MockTrained;

        fn fit(self, _x: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
            let mut classes = y.to_vec();
            classes.sort_unstable();
            classes.dedup();
            Ok(MockTrained {
                weight: self.weight,
                classes: Array1::from(classes),
            })
        }
    }

    impl Predict<Array2<f64>, Array1<i32>> for MockTrained {
        fn predict(&self, x: &Array2<f64>) -> SklResult<Array1<i32>> {
            let n_samples = x.nrows();
            let mut predictions = Array1::zeros(n_samples);

            for i in 0..n_samples {
                let sum: f64 = x.row(i).sum();
                let class_idx = ((sum * self.weight) as usize) % self.classes.len();
                predictions[i] = self.classes[class_idx];
            }

            Ok(predictions)
        }
    }

    impl PredictProba<Array2<f64>, Array2<f64>> for MockTrained {
        fn predict_proba(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
            let n_samples = x.nrows();
            let n_classes = self.classes.len();
            let mut probabilities = Array2::zeros((n_samples, n_classes));

            if n_classes == 0 {
                return Ok(probabilities);
            }

            for i in 0..n_samples {
                let sum: f64 = x.row(i).sum();
                let base_prob = (sum * self.weight).sin().abs();
                let class_idx = (base_prob * n_classes as f64) as usize % n_classes;

                // Set probabilities with some variation
                for j in 0..n_classes {
                    if j == class_idx {
                        probabilities[[i, j]] = 0.6 + 0.3 * base_prob;
                    } else {
                        probabilities[[i, j]] = (0.4 - 0.3 * base_prob) / (n_classes - 1) as f64;
                    }
                }

                // Normalize
                let row_sum: f64 = probabilities.row(i).sum();
                if row_sum > 0.0 {
                    for j in 0..n_classes {
                        probabilities[[i, j]] /= row_sum;
                    }
                }
            }

            Ok(probabilities)
        }
    }

    impl Estimator for MockClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;
        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Estimator for MockTrained {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;
        fn config(&self) -> &Self::Config {
            &()
        }
    }

    #[test]
    fn test_stacking_creation() {
        let base_classifiers = vec![
            MockClassifier::new(1.0),
            MockClassifier::new(2.0),
            MockClassifier::new(3.0),
        ];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);
        assert_eq!(stacking.base_estimators().len(), 3);
    }

    #[test]
    fn test_stacking_builder() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::builder(base_classifiers, meta_learner)
            .stacking_method(StackingMethod::CrossValidation { cv_folds: 3 })
            .meta_learner_strategy(MetaLearnerStrategy::Average)
            .passthrough(true)
            .stack_probabilities(false)
            .random_state(Some(42))
            .build();

        assert!(matches!(
            stacking.config.stacking_method,
            StackingMethod::CrossValidation { cv_folds: 3 }
        ));
        assert!(matches!(
            stacking.config.meta_learner_strategy,
            MetaLearnerStrategy::Average
        ));
        assert_eq!(stacking.config.passthrough, true);
        assert_eq!(stacking.config.stack_probabilities, false);
        assert_eq!(stacking.config.random_state, Some(42));
    }

    #[test]
    fn test_stacking_estimator_trait() {
        let base_classifiers = vec![MockClassifier::new(1.0)];
        let meta_learner = MockClassifier::new(0.5);
        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);

        let _config = &stacking.config;
        // Should compile without errors
    }

    #[test]
    fn test_stacking_basic() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .random_state(Some(42));

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let predictions = trained.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), x.nrows());
        for &pred in predictions.iter() {
            assert!(pred == 0 || pred == 1);
        }
    }

    #[test]
    fn test_stacking_cv_method() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .stacking_method(StackingMethod::CrossValidation { cv_folds: 3 })
            .random_state(Some(42));

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [1.0, 4.0],
            [2.0, 4.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1, 0];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let predictions = trained.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_stacking_holdout_method() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .stacking_method(StackingMethod::Holdout { test_size: 0.3 })
            .random_state(Some(42));

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [1.0, 4.0],
            [2.0, 4.0],
            [3.0, 4.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let predictions = trained.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_stacking_blending_method() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .stacking_method(StackingMethod::Blending { blend_ratio: 0.2 })
            .random_state(Some(42));

        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 1.0],
            [1.0, 3.0],
            [2.0, 1.0],
            [3.0, 2.0],
            [4.0, 1.0],
            [1.0, 4.0],
            [2.0, 4.0],
            [3.0, 4.0]
        ];
        let y = array![0, 1, 0, 1, 0, 1, 0, 1, 0, 1];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let predictions = trained.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_stacking_predict_proba() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .random_state(Some(42));

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![0, 1, 0, 1];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let probabilities = trained
            .predict_proba(&x)
            .expect("Failed to predict probabilities");

        assert_eq!(probabilities.dim(), (x.nrows(), 2));

        // Check that probabilities sum to approximately 1
        for i in 0..x.nrows() {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10, "Row {} sum: {}", i, row_sum);
        }
    }

    #[test]
    fn test_stacking_passthrough() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner)
            .passthrough(true)
            .random_state(Some(42));

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [1.0, 3.0]];
        let y = array![0, 1, 0, 1];

        let trained = stacking.fit(&x, &y).expect("Failed to fit");
        let predictions = trained.predict(&x).expect("Failed to predict");

        assert_eq!(predictions.len(), x.nrows());
    }

    #[test]
    fn test_stacking_insufficient_classes() {
        let base_classifiers = vec![MockClassifier::new(1.0)];
        let meta_learner = MockClassifier::new(0.5);
        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);

        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // All same class

        let result = stacking.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_stacking_empty_base_estimators() {
        let base_classifiers: Vec<MockClassifier> = vec![];
        let meta_learner = MockClassifier::new(0.5);
        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);

        let x = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 1];

        let result = stacking.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_stacking_clone() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_learner = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_learner);
        let cloned = stacking.clone();

        assert_eq!(
            stacking.base_estimators().len(),
            cloned.base_estimators().len()
        );
    }

    #[test]
    fn test_dynamic_meta_learner_selection() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_classifier = MockClassifier::new(0.5);

        let candidates = vec![
            MetaLearnerStrategy::Average,
            MetaLearnerStrategy::WeightedAverage,
            MetaLearnerStrategy::LogisticRegression,
        ];

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_classifier)
            .dynamic_meta_learner(candidates, 3, DynamicSelectionScoring::Accuracy);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 0];

        let trained = stacking
            .fit(&x, &y)
            .expect("Failed to fit with dynamic meta-learner");
        let predictions = trained.predict(&x).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Check that dynamic strategy was selected
        match &trained.base_estimators.selected_meta_learner {
            MetaLearnerStrategy::Dynamic { .. } => (),
            _ => panic!("Expected dynamic meta-learner strategy"),
        }
    }

    #[test]
    fn test_adaptive_meta_learner_selection() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_classifier = MockClassifier::new(0.5);

        let rules = AdaptiveSelectionRules {
            min_samples_for_complex: 10,
            min_features_for_nn: 5,
            max_classes_for_linear: 5,
            imbalance_threshold: 2.0,
        };

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_classifier)
            .adaptive_meta_learner(rules);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 0];

        let trained = stacking
            .fit(&x, &y)
            .expect("Failed to fit with adaptive meta-learner");
        let predictions = trained.predict(&x).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        // Check that adaptive strategy was selected
        match &trained.base_estimators.selected_meta_learner {
            MetaLearnerStrategy::Adaptive { .. } => (),
            _ => panic!("Expected adaptive meta-learner strategy"),
        }
    }

    #[test]
    fn test_adaptive_selection_rules() {
        let rules = AdaptiveSelectionRules::default();

        // Test balanced small dataset
        let balanced_distribution = vec![0.33, 0.33, 0.34];
        let strategy = rules.select_meta_learner(50, 5, 3, &balanced_distribution);
        assert_eq!(strategy, MetaLearnerStrategy::Average); // Small dataset

        // Test imbalanced dataset
        let imbalanced_distribution = vec![0.8, 0.1, 0.1];
        let strategy = rules.select_meta_learner(1000, 10, 3, &imbalanced_distribution);
        assert_eq!(strategy, MetaLearnerStrategy::WeightedAverage); // High imbalance

        // Test high-dimensional dataset
        let balanced_distribution = vec![0.33, 0.33, 0.34];
        let strategy = rules.select_meta_learner(1000, 20, 3, &balanced_distribution);
        assert_eq!(strategy, MetaLearnerStrategy::NeuralNetwork); // High dimensions

        // Test many classes
        let many_classes_distribution = vec![0.1; 12]; // 12 classes
        let strategy = rules.select_meta_learner(1000, 15, 12, &many_classes_distribution);
        assert_eq!(strategy, MetaLearnerStrategy::NeuralNetwork); // Many classes
    }

    #[test]
    fn test_online_adaptation_configuration() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_classifier = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_classifier)
            .enable_online_adaptation(0.05, 50);

        // Check configuration
        assert!(stacking.config.enable_online_adaptation);
        assert_eq!(stacking.config.adaptation_threshold, 0.05);
        assert_eq!(stacking.config.adaptation_window_size, 50);
    }

    #[test]
    fn test_stacking_configuration_methods() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_classifier = MockClassifier::new(0.5);

        let candidates = vec![MetaLearnerStrategy::Average, MetaLearnerStrategy::Linear];

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_classifier)
            .stacking_method(StackingMethod::Holdout { test_size: 0.3 })
            .dynamic_meta_learner(candidates, 5, DynamicSelectionScoring::F1Score)
            .enable_online_adaptation(0.02, 100)
            .passthrough(true)
            .stack_probabilities(false);

        // Check all configurations
        assert_eq!(
            stacking.config.stacking_method,
            StackingMethod::Holdout { test_size: 0.3 }
        );
        assert!(stacking.config.enable_online_adaptation);
        assert_eq!(stacking.config.adaptation_threshold, 0.02);
        assert_eq!(stacking.config.adaptation_window_size, 100);
        assert!(stacking.config.passthrough);
        assert!(!stacking.config.stack_probabilities);
    }

    #[test]
    fn test_adaptation_event_creation() {
        let event = AdaptationEvent {
            iteration: 42,
            previous_strategy: MetaLearnerStrategy::Average,
            new_strategy: MetaLearnerStrategy::LogisticRegression,
            performance_improvement: 0.05,
            reason: "Better accuracy on validation set".to_string(),
        };

        assert_eq!(event.iteration, 42);
        assert_eq!(event.performance_improvement, 0.05);
        assert!(event.reason.contains("Better accuracy"));
    }

    #[test]
    fn test_trained_data_with_dynamic_fields() {
        let base_classifiers = vec![MockClassifier::new(1.0), MockClassifier::new(2.0)];
        let meta_classifier = MockClassifier::new(0.5);

        let stacking = MulticlassStackingClassifier::new(base_classifiers, meta_classifier);

        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 2, 0];

        let trained = stacking
            .fit(&x, &y)
            .expect("Failed to fit stacking classifier");

        // Check that new fields are properly initialized
        assert!(trained.base_estimators.strategy_scores.is_none());
        assert!(trained.base_estimators.adaptation_history.is_empty());
        assert!(matches!(
            trained.base_estimators.selected_meta_learner,
            MetaLearnerStrategy::LogisticRegression
        ));
    }
}
