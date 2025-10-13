//! Bootstrap Aggregating (Bagging) Classifier
//!
//! This module implements the Bootstrap Aggregating (Bagging) ensemble method for multiclass classification.
//! Bagging is an ensemble method that fits base classifiers each on random subsets of the original dataset.
//! It uses averaging to combine the predictions from different base estimators, which can reduce overfitting and variance.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{rngs::StdRng, Random};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Trained, Untrained},
};
use std::marker::PhantomData;

/// Bootstrap aggregating (Bagging) configuration
#[derive(Debug, Clone)]
pub struct BaggingConfig {
    /// Number of base estimators
    pub n_estimators: usize,
    /// Maximum number of samples to draw for each base estimator
    pub max_samples: Option<usize>,
    /// Whether to sample with replacement
    pub bootstrap: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Number of parallel jobs
    pub n_jobs: Option<i32>,
    /// Whether to use warm start (continue training from existing estimators)
    pub warm_start: bool,
}

impl Default for BaggingConfig {
    fn default() -> Self {
        Self {
            n_estimators: 10,
            max_samples: None,
            bootstrap: true,
            random_state: None,
            n_jobs: None,
            warm_start: false,
        }
    }
}

/// Bootstrap Aggregating (Bagging) Multiclass Classifier
///
/// Bagging is an ensemble method that fits base classifiers each on random
/// subsets of the original dataset. It uses averaging to combine the predictions
/// from different base estimators, which can reduce overfitting and variance.
///
/// # Type Parameters
///
/// * `C` - The base classifier type that implements Fit and Predict
/// * `S` - The state type (Untrained or Trained)
///
/// # Examples
///
/// ```
/// use sklears_multiclass::ensemble::BaggingClassifier;
/// use scirs2_autograd::ndarray::array;
///
/// // Example with a hypothetical base classifier
/// // let base_classifier = SomeClassifier::new();
/// // let bagging = BaggingClassifier::new(base_classifier)
/// //     .n_estimators(50)
/// //     .bootstrap(true);
/// ```
#[derive(Debug)]
pub struct BaggingClassifier<C, S = Untrained> {
    base_estimator: C,
    config: BaggingConfig,
    state: PhantomData<S>,
}

/// Trained data for Bagging classifier
#[derive(Debug)]
pub struct BaggingTrainedData<T> {
    /// Vector of trained estimators
    estimators: Vec<T>,
    /// Classes seen during training
    classes: Array1<i32>,
    /// Number of classes
    n_classes: usize,
    /// Out-of-bag score (if available)
    oob_score: Option<f64>,
}

impl<T: Clone> Clone for BaggingTrainedData<T> {
    fn clone(&self) -> Self {
        Self {
            estimators: self.estimators.clone(),
            classes: self.classes.clone(),
            n_classes: self.n_classes,
            oob_score: self.oob_score,
        }
    }
}

impl<C> BaggingClassifier<C, Untrained> {
    /// Create a new Bagging classifier
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: BaggingConfig::default(),
            state: PhantomData,
        }
    }

    /// Create a builder for Bagging classifier
    pub fn builder(base_estimator: C) -> BaggingBuilder<C> {
        BaggingBuilder::new(base_estimator)
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum samples to draw
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Set whether to bootstrap (sample with replacement)
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
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

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }
}

/// Builder for Bagging classifier
#[derive(Debug)]
pub struct BaggingBuilder<C> {
    base_estimator: C,
    config: BaggingConfig,
}

impl<C> BaggingBuilder<C> {
    /// Create a new builder
    pub fn new(base_estimator: C) -> Self {
        Self {
            base_estimator,
            config: BaggingConfig::default(),
        }
    }

    /// Set the number of estimators
    pub fn n_estimators(mut self, n_estimators: usize) -> Self {
        self.config.n_estimators = n_estimators;
        self
    }

    /// Set the maximum samples to draw
    pub fn max_samples(mut self, max_samples: Option<usize>) -> Self {
        self.config.max_samples = max_samples;
        self
    }

    /// Set whether to bootstrap
    pub fn bootstrap(mut self, bootstrap: bool) -> Self {
        self.config.bootstrap = bootstrap;
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

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Build the Bagging classifier
    pub fn build(self) -> BaggingClassifier<C, Untrained> {
        BaggingClassifier {
            base_estimator: self.base_estimator,
            config: self.config,
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for BaggingClassifier<C, Untrained> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
            state: PhantomData,
        }
    }
}

impl<C: Clone> Clone for BaggingBuilder<C> {
    fn clone(&self) -> Self {
        Self {
            base_estimator: self.base_estimator.clone(),
            config: self.config.clone(),
        }
    }
}

impl<C> Estimator for BaggingClassifier<C, Untrained> {
    type Config = BaggingConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

type TrainedBagging<T> = BaggingClassifier<BaggingTrainedData<T>, Trained>;

impl<T> TrainedBagging<T>
where
    T: Predict<Array2<f64>, Array1<i32>>,
{
    /// Get the out-of-bag score
    pub fn oob_score(&self) -> Option<f64> {
        self.base_estimator.oob_score
    }

    /// Get the classes
    pub fn classes(&self) -> &Array1<i32> {
        &self.base_estimator.classes
    }

    /// Get the fitted estimators
    pub fn estimators(&self) -> &[T] {
        &self.base_estimator.estimators
    }

    /// Get the number of estimators
    pub fn n_estimators_(&self) -> usize {
        self.base_estimator.estimators.len()
    }
}

impl<T> TrainedBagging<T>
where
    T: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    /// Continue training with additional estimators (warm start)
    ///
    /// This method allows adding more estimators to an already trained bagging classifier
    /// when warm start is enabled. The new estimators will be trained on bootstrap samples
    /// from the provided data and added to the existing ensemble.
    ///
    /// # Arguments
    /// * `base_estimator` - The base classifier to clone and train
    /// * `X` - Training data features
    /// * `y` - Training data labels
    /// * `additional_estimators` - Number of additional estimators to add
    ///
    /// # Returns
    /// A new trained bagging classifier with the additional estimators
    ///
    /// # Errors
    /// Returns an error if warm start is not enabled or if training fails
    pub fn partial_fit<C>(
        mut self,
        base_estimator: C,
        X: &Array2<f64>,
        y: &Array1<i32>,
        additional_estimators: usize,
    ) -> SklResult<Self>
    where
        C: Clone + Fit<Array2<f64>, Array1<i32>, Fitted = T> + Send + Sync,
    {
        // Check if warm start is enabled
        if !self.config.warm_start {
            return Err(SklearsError::InvalidInput(
                "Warm start is not enabled. Set warm_start=true to use partial_fit".to_string(),
            ));
        }

        validate::check_consistent_length(X, y)?;
        let (n_samples, _n_features) = X.dim();

        // Verify that the classes are consistent
        let mut new_classes: Vec<i32> = y.iter().cloned().collect();
        new_classes.sort_unstable();
        new_classes.dedup();

        let existing_classes: Vec<i32> = self.base_estimator.classes.to_vec();
        for &class in &new_classes {
            if !existing_classes.contains(&class) {
                return Err(SklearsError::InvalidInput(format!(
                    "New class {} not seen during initial training",
                    class
                )));
            }
        }

        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed + self.base_estimator.estimators.len() as u64),
            None => Random::seed(42),
        };

        // Train additional estimators
        let mut new_estimators: Vec<T> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            (0..additional_estimators)
                .into_par_iter()
                .map(|i| {
                    let mut local_rng = Random::seed(
                        self.config.random_state.unwrap_or(0)
                            + (self.base_estimator.estimators.len() + i) as u64,
                    );
                    let (X_bootstrap, y_bootstrap) =
                        self.bootstrap_sample(X, y, max_samples, &mut local_rng)?;
                    base_estimator.clone().fit(&X_bootstrap, &y_bootstrap)
                })
                .collect::<SklResult<Vec<_>>>()?
        } else {
            // Sequential training
            let mut estimators = Vec::new();
            for _i in 0..additional_estimators {
                let (X_bootstrap, y_bootstrap) =
                    self.bootstrap_sample(X, y, max_samples, &mut rng)?;
                let estimator = base_estimator.clone().fit(&X_bootstrap, &y_bootstrap)?;
                estimators.push(estimator);
            }
            estimators
        };

        // Add new estimators to existing ones
        self.base_estimator.estimators.append(&mut new_estimators);

        Ok(self)
    }

    /// Bootstrap sample helper method for warm start
    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        max_samples: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<i32>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let sample_size = max_samples.min(n_samples);

        let mut X_bootstrap = Array2::zeros((sample_size, n_features));
        let mut y_bootstrap = Array1::zeros(sample_size);

        if self.config.bootstrap {
            // Sample with replacement
            for i in 0..sample_size {
                let idx = rng.random_range(0, n_samples);
                X_bootstrap.row_mut(i).assign(&X.row(idx));
                y_bootstrap[i] = y[idx];
            }
        } else {
            // Sample without replacement
            let mut indices: Vec<usize> = (0..n_samples).collect();
            // Shuffle indices using Fisher-Yates algorithm
            for i in (1..indices.len()).rev() {
                let j = rng.random_range(0, i + 1);
                indices.swap(i, j);
            }
            indices.truncate(sample_size);

            for (i, &idx) in indices.iter().enumerate() {
                X_bootstrap.row_mut(i).assign(&X.row(idx));
                y_bootstrap[i] = y[idx];
            }
        }

        Ok((X_bootstrap, y_bootstrap))
    }
}

impl<C> Fit<Array2<f64>, Array1<i32>> for BaggingClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    type Fitted = TrainedBagging<C::Fitted>;

    fn fit(self, X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(X, y)?;
        let (n_samples, _n_features) = X.dim();

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

        let max_samples = self.config.max_samples.unwrap_or(n_samples);
        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        // Train estimators in parallel or sequentially
        let estimators: Vec<C::Fitted> = if self.config.n_jobs.is_some_and(|n| n != 1) {
            // Parallel training
            (0..self.config.n_estimators)
                .into_par_iter()
                .map(|i| {
                    let mut local_rng =
                        Random::seed(self.config.random_state.unwrap_or(0) + i as u64);
                    let (X_bootstrap, y_bootstrap) =
                        self.bootstrap_sample(X, y, max_samples, &mut local_rng)?;
                    self.base_estimator.clone().fit(&X_bootstrap, &y_bootstrap)
                })
                .collect::<SklResult<Vec<_>>>()?
        } else {
            // Sequential training
            let mut estimators = Vec::new();
            for _i in 0..self.config.n_estimators {
                let (X_bootstrap, y_bootstrap) =
                    self.bootstrap_sample(X, y, max_samples, &mut rng)?;
                let estimator = self
                    .base_estimator
                    .clone()
                    .fit(&X_bootstrap, &y_bootstrap)?;
                estimators.push(estimator);
            }
            estimators
        };

        let trained_data = BaggingTrainedData {
            estimators,
            classes: classes_array,
            n_classes,
            oob_score: None, // Could be computed if needed
        };

        Ok(BaggingClassifier {
            base_estimator: trained_data,
            config: self.config,
            state: PhantomData,
        })
    }
}

impl<C> BaggingClassifier<C, Untrained>
where
    C: Clone + Fit<Array2<f64>, Array1<i32>> + Send + Sync,
    C::Fitted: Predict<Array2<f64>, Array1<i32>> + Clone + Send + Sync,
{
    fn bootstrap_sample(
        &self,
        X: &Array2<f64>,
        y: &Array1<i32>,
        max_samples: usize,
        rng: &mut Random<StdRng>,
    ) -> SklResult<(Array2<f64>, Array1<i32>)> {
        let n_samples = X.nrows();
        let n_features = X.ncols();
        let sample_size = max_samples.min(n_samples);

        let mut X_bootstrap = Array2::zeros((sample_size, n_features));
        let mut y_bootstrap = Array1::zeros(sample_size);

        for i in 0..sample_size {
            let idx = if self.config.bootstrap {
                // Sample with replacement
                rng.random_range(0, n_samples)
            } else {
                // Sample without replacement (would need more complex logic for true without replacement)
                rng.random_range(0, n_samples)
            };

            X_bootstrap.row_mut(i).assign(&X.row(idx));
            y_bootstrap[i] = y[idx];
        }

        Ok((X_bootstrap, y_bootstrap))
    }
}

impl<T> Predict<Array2<f64>, Array1<i32>> for TrainedBagging<T>
where
    T: Predict<Array2<f64>, Array1<i32>>,
{
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
        let n_samples = X.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        // Accumulate votes from all estimators
        for estimator in &trained_data.estimators {
            let predictions = estimator.predict(X)?;

            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred) {
                    votes[[i, class_idx]] += 1.0;
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

impl<T> PredictProba<Array2<f64>, Array2<f64>> for TrainedBagging<T>
where
    T: Predict<Array2<f64>, Array1<i32>>,
{
    fn predict_proba(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let trained_data = &self.base_estimator;

        // Initialize vote matrix
        let mut votes = Array2::<f64>::zeros((n_samples, trained_data.n_classes));

        // Accumulate votes from all estimators
        for estimator in &trained_data.estimators {
            let predictions = estimator.predict(X)?;

            for (i, &pred) in predictions.iter().enumerate() {
                if let Some(class_idx) = trained_data.classes.iter().position(|&c| c == pred) {
                    votes[[i, class_idx]] += 1.0;
                }
            }
        }

        // Convert votes to probabilities
        let mut probabilities = Array2::<f64>::zeros((n_samples, trained_data.n_classes));
        let n_estimators = trained_data.estimators.len() as f64;

        for i in 0..n_samples {
            for j in 0..trained_data.n_classes {
                probabilities[[i, j]] = votes[[i, j]] / n_estimators;
            }
        }

        Ok(probabilities)
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
        fitted: bool,
    }

    impl MockClassifier {
        fn new() -> Self {
            Self { fitted: false }
        }
    }

    #[derive(Debug, Clone)]
    struct MockClassifierTrained {
        classes: Array1<i32>,
    }

    impl Estimator for MockClassifier {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Array2<f64>, Array1<i32>> for MockClassifier {
        type Fitted = MockClassifierTrained;

        fn fit(self, _X: &Array2<f64>, y: &Array1<i32>) -> SklResult<Self::Fitted> {
            let mut classes: Vec<i32> = y.iter().cloned().collect();
            classes.sort_unstable();
            classes.dedup();
            let classes_array = Array1::from_vec(classes);

            Ok(MockClassifierTrained {
                classes: classes_array,
            })
        }
    }

    impl Predict<Array2<f64>, Array1<i32>> for MockClassifierTrained {
        fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<i32>> {
            let n_samples = X.nrows();
            let mut predictions = Array1::zeros(n_samples);

            // Simple mock prediction: alternate between classes
            for i in 0..n_samples {
                predictions[i] = self.classes[i % self.classes.len()];
            }

            Ok(predictions)
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_basic() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier)
            .n_estimators(5)
            .bootstrap(true);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = bagging.fit(&X, &y).expect("Failed to fit Bagging");
        assert_eq!(trained.classes().len(), 2);
        assert_eq!(trained.n_estimators_(), 5);

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);

        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (4, 2));
    }

    #[test]
    fn test_bagging_builder() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::builder(base_classifier)
            .n_estimators(8)
            .max_samples(Some(3))
            .bootstrap(false)
            .random_state(Some(42))
            .n_jobs(Some(1))
            .build();

        assert_eq!(bagging.config().n_estimators, 8);
        assert_eq!(bagging.config().max_samples, Some(3));
        assert_eq!(bagging.config().bootstrap, false);
        assert_eq!(bagging.config().random_state, Some(42));
        assert_eq!(bagging.config().n_jobs, Some(1));
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_multiclass() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier).n_estimators(4);

        let X = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0]
        ];
        let y = array![0, 1, 2, 0, 1, 2];

        let trained = bagging
            .fit(&X, &y)
            .expect("Failed to fit multiclass Bagging");
        assert_eq!(trained.classes().len(), 3);

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 6);

        let probabilities = trained
            .predict_proba(&X)
            .expect("Failed to predict probabilities");
        assert_eq!(probabilities.dim(), (6, 3));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!(
                (row_sum - 1.0).abs() < 1e-10,
                "Probabilities should sum to 1"
            );
        }
    }

    #[test]
    fn test_bagging_clone() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier);

        let _cloned = bagging.clone();
        // If clone works, this test passes
    }

    #[test]
    fn test_bagging_config_default() {
        let config = BaggingConfig::default();
        assert_eq!(config.n_estimators, 10);
        assert_eq!(config.max_samples, None);
        assert_eq!(config.bootstrap, true);
        assert_eq!(config.random_state, None);
        assert_eq!(config.n_jobs, None);
    }

    #[test]
    fn test_bagging_estimator_trait() {
        let classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(classifier);

        let config = bagging.config();
        assert_eq!(config.n_estimators, 10);
        assert_eq!(config.bootstrap, true);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_insufficient_classes() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier);

        let X = array![[1.0, 2.0], [2.0, 3.0]];
        let y = array![0, 0]; // Only one class

        let result = bagging.fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_parallel_training() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier)
            .n_estimators(3)
            .n_jobs(Some(-1)); // Use parallel training

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = bagging
            .fit(&X, &y)
            .expect("Failed to fit Bagging with parallel training");
        assert_eq!(trained.n_estimators_(), 3);

        let predictions = trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_bagging_warm_start_configuration() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier)
            .n_estimators(5)
            .warm_start(true);

        assert!(bagging.config.warm_start);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_warm_start_partial_fit() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(true)
            .random_state(Some(42));

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        // Initial training
        let trained = bagging.fit(&X, &y).expect("Failed to fit initial model");
        assert_eq!(trained.n_estimators_(), 3);

        // Continue training with partial_fit
        let extended_trained = trained
            .partial_fit(base_classifier, &X, &y, 2)
            .expect("Failed to perform partial fit");

        assert_eq!(extended_trained.n_estimators_(), 5);

        let predictions = extended_trained.predict(&X).expect("Failed to predict");
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_warm_start_without_flag() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(false); // Warm start disabled

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = bagging.fit(&X, &y).expect("Failed to fit initial model");

        // Should fail because warm start is not enabled
        let result = trained.partial_fit(base_classifier, &X, &y, 2);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_warm_start_new_classes() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::new(base_classifier.clone())
            .n_estimators(3)
            .warm_start(true);

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = bagging.fit(&X, &y).expect("Failed to fit initial model");

        // Try to add new classes (should fail)
        let X_new = array![[5.0, 6.0], [6.0, 7.0]];
        let y_new = array![2, 2]; // New class not seen before

        let result = trained.partial_fit(base_classifier, &X_new, &y_new, 1);
        assert!(result.is_err());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_bagging_warm_start_builder() {
        let base_classifier = MockClassifier::new();
        let bagging = BaggingClassifier::builder(base_classifier.clone())
            .n_estimators(2)
            .warm_start(true)
            .bootstrap(true)
            .random_state(Some(123))
            .build();

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];
        let y = array![0, 1, 0, 1];

        let trained = bagging.fit(&X, &y).expect("Failed to fit initial model");
        assert_eq!(trained.n_estimators_(), 2);
        assert!(trained.config.warm_start);

        let extended = trained
            .partial_fit(base_classifier, &X, &y, 3)
            .expect("Failed to extend model");
        assert_eq!(extended.n_estimators_(), 5);
    }

    #[test]
    fn test_bagging_warm_start_config_default() {
        let config = BaggingConfig::default();
        assert_eq!(config.warm_start, false);
    }
}
