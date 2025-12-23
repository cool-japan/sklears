//! One-vs-Rest (One-vs-All) multiclass classification strategy

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Configuration for One-vs-Rest Classifier
#[derive(Debug, Clone, Default)]
pub struct OneVsRestConfig {
    /// Number of parallel jobs (-1 for all cores)
    pub n_jobs: Option<i32>,
}

/// One-vs-Rest (One-vs-All) Classifier
///
/// This strategy consists in fitting one classifier per class. For each classifier,
/// the class is fitted against all the other classes. In addition to its computational
/// efficiency (only n_classes classifiers are needed), one advantage of this approach is
/// its interpretability. Since each class is represented by one and only one classifier,
/// it is possible to gain knowledge about the class by inspecting its corresponding classifier.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::OneVsRestClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let ovr = OneVsRestClassifier::new(base_classifier);
/// ```
#[derive(Debug)]
pub struct OneVsRestClassifier<C, S = Untrained> {
    base_estimator: C,
    config: OneVsRestConfig,
    state: PhantomData<S>,
}

impl<C> OneVsRestClassifier<C, Untrained> {
    /// Create a new OneVsRestClassifier instance with a base estimator
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: OneVsRestConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for OneVsRestClassifier
    pub fn builder(base_estimator: C) -> OneVsRestBuilder<C> {
        OneVsRestBuilder::new(base_estimator)
    }

    /// Set the number of parallel jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Get a reference to the base estimator
    pub fn base_estimator(&self) -> &C {
        &self.base_estimator
    }
}

/// Builder for OneVsRestClassifier
#[derive(Debug)]
pub struct OneVsRestBuilder<C> {
    base_estimator: C,
    config: OneVsRestConfig,
}

impl<C> OneVsRestBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: OneVsRestConfig::default(),
        }
    }

    /// Set the number of parallel jobs
    ///
    /// # Arguments
    /// * `n_jobs` - Number of parallel jobs to run. None for sequential, Some(-1) for all cores
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.config.n_jobs = n_jobs;
        self
    }

    /// Enable parallel training using all available cores
    pub fn parallel(mut self) -> Self {
        self.config.n_jobs = Some(-1);
        self
    }

    /// Disable parallel training (use sequential training)
    pub fn sequential(mut self) -> Self {
        self.config.n_jobs = None;
        self
    }

    /// Build the OneVsRestClassifier
    pub fn build(self) -> OneVsRestClassifier<C, Untrained> {
        OneVsRestClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for OneVsRestClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for OneVsRestBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for OneVsRestClassifier<C, Untrained> {
    type Config = OneVsRestConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

// Trained data container for OneVsRestClassifier
#[derive(Debug)]
pub struct OneVsRestTrainedData<T> {
    estimators: Vec<T>,
    classes: Array1<i32>,
    n_features: usize,
}

impl<T: Clone> Clone for OneVsRestTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            n_features: self.n_features,
        }
    }
}

pub type TrainedOneVsRest<T> = OneVsRestClassifier<OneVsRestTrainedData<T>, Trained>;

/// Implementation for classifiers that can fit binary problems
impl<C> Fit<Array2<Float>, Array1<i32>> for OneVsRestClassifier<C, Untrained>
where
    C: Clone + Send + Sync + Fit<Array2<Float>, Array1<Float>>,
    C::Fitted: Predict<Array2<Float>, Array1<Float>> + Send,
{
    type Fitted = TrainedOneVsRest<C::Fitted>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let (_n_samples, n_features) = x.dim();

        // Get unique classes
        let mut classes: Vec<i32> = y
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();

        if classes.len() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for multiclass classification".to_string(),
            ));
        }

        // Prepare binary classification problems
        let binary_problems: Vec<_> = classes
            .iter()
            .map(|&target_class| {
                // Create binary labels: 1.0 for target class, 0.0 for others
                let binary_y: Array1<Float> =
                    y.mapv(|label| if label == target_class { 1.0 } else { 0.0 });
                (target_class, binary_y)
            })
            .collect();

        // Train binary classifiers (with parallel support if enabled)
        let estimators: SklResult<Vec<_>> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            binary_problems
                .into_par_iter()
                .map(|(_, binary_y)| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        } else {
            // Sequential training
            binary_problems
                .into_iter()
                .map(|(_, binary_y)| self.base_estimator.clone().fit(x, &binary_y))
                .collect::<SklResult<Vec<_>>>()
        };

        let estimators = estimators?;

        Ok(OneVsRestClassifier {
            base_estimator: OneVsRestTrainedData {
                estimators,
                classes: Array1::from(classes),
                n_features,
            },
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<T> TrainedOneVsRest<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> usize {
        self.base_estimator.classes.len()
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }
}

impl<T> Predict<Array2<Float>, Array1<i32>> for TrainedOneVsRest<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict(&self, x: &Array2<Float>) -> SklResult<Array1<i32>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        // Get decision scores from each binary classifier
        for (class_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
            let scores = estimator.predict(x)?;
            decision_scores
                .slice_mut(scirs2_core::ndarray::s![.., class_idx])
                .assign(&scores);
        }

        // Predict class with highest score
        let predictions = decision_scores.map_axis(Axis(1), |row| {
            let max_idx = row
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            self.base_estimator.classes[max_idx]
        });

        Ok(predictions)
    }
}

/// Implementation of probability predictions for OneVsRest
impl<T> PredictProba<Array2<Float>, Array2<Float>> for TrainedOneVsRest<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    fn predict_proba(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut scores = Array2::zeros((n_samples, n_classes));

        // Get scores from each binary classifier
        for (class_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
            let class_scores = estimator.predict(x)?;
            scores
                .slice_mut(scirs2_core::ndarray::s![.., class_idx])
                .assign(&class_scores);
        }

        // Convert scores to probabilities using softmax
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        for (sample_idx, sample_scores) in scores.axis_iter(Axis(0)).enumerate() {
            // Apply softmax to normalize scores to probabilities
            let max_score = sample_scores
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_scores: Array1<Float> = sample_scores.mapv(|s| (s - max_score).exp());
            let sum_exp = exp_scores.sum();

            for (class_idx, &exp_score) in exp_scores.iter().enumerate() {
                probabilities[[sample_idx, class_idx]] = exp_score / sum_exp;
            }
        }

        Ok(probabilities)
    }
}

/// Enhanced prediction with probability threshold support
impl<T> TrainedOneVsRest<T>
where
    T: Predict<Array2<Float>, Array1<Float>>,
{
    /// Predict with probability threshold
    /// Only makes a positive prediction if the probability exceeds the threshold
    pub fn predict_with_threshold(
        &self,
        x: &Array2<Float>,
        threshold: Float,
    ) -> SklResult<Array1<Option<i32>>> {
        let probabilities = self.predict_proba(x)?;
        let n_samples = x.nrows();
        let mut predictions = Array1::from_elem(n_samples, None);

        for (sample_idx, sample_probs) in probabilities.axis_iter(Axis(0)).enumerate() {
            let (max_idx, &max_prob) = sample_probs
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap();

            if max_prob >= threshold {
                predictions[sample_idx] = Some(self.base_estimator.classes[max_idx]);
            }
        }

        Ok(predictions)
    }

    /// Get decision function scores (raw predictions from binary classifiers)
    pub fn decision_function(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = x.dim();

        if n_features != self.base_estimator.n_features {
            return Err(SklearsError::InvalidInput(
                "Number of features doesn't match training data".to_string(),
            ));
        }

        let n_classes = self.base_estimator.classes.len();
        let mut decision_scores = Array2::zeros((n_samples, n_classes));

        // Get raw scores from each binary classifier
        for (class_idx, estimator) in self.base_estimator.estimators.iter().enumerate() {
            let scores = estimator.predict(x)?;
            decision_scores
                .slice_mut(scirs2_core::ndarray::s![.., class_idx])
                .assign(&scores);
        }

        Ok(decision_scores)
    }
}
