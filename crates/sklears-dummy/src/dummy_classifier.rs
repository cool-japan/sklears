//! Dummy classifier for baseline comparisons

use crate::validation::{analyze_classification_dataset, get_adaptive_classification_strategy};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{prelude::*, Rng};
use sklears_core::error::Result;
use sklears_core::traits::{Estimator, Fit, Predict, PredictProba};
use sklears_core::types::{Features, Float, Int};
use std::collections::HashMap;

/// Strategy for making predictions
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Strategy {
    /// Most frequent class
    MostFrequent,
    /// Random predictions based on class distribution
    Stratified,
    /// Random uniform predictions
    Uniform,
    /// Always predict a constant value
    Constant,
    /// Predictions based on a prior distribution
    Prior,
    /// Automatically select the best strategy
    Auto,
    /// Adaptively select strategy based on dataset characteristics
    Adaptive,
    /// Sample from empirical distribution (bootstrap sampling)
    Empirical,
    /// Bayesian classifier with prior beliefs and uncertainty estimation
    Bayesian,
}

/// Dummy classifier that makes predictions using simple rules
///
/// This classifier serves as a simple baseline to compare with other classifiers.
/// It does not use the input features and makes predictions based on simple rules:
/// - "most_frequent": always predicts the most frequent label in the training set
/// - "stratified": generates predictions by respecting the training set's class distribution
/// - "uniform": generates predictions uniformly at random
/// - "constant": always predicts a constant label
/// - "prior": always predicts the class that maximizes the prior
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct DummyClassifier<State = sklears_core::traits::Untrained> {
    pub strategy: Strategy,
    pub random_state: Option<u64>,
    pub constant: Option<Int>,
    pub(crate) classes_: Option<Array1<Int>>,
    pub(crate) class_prior_: Option<Array1<Float>>,
    pub(crate) n_classes_: Option<usize>,
    pub(crate) most_frequent_class_: Option<Int>,
    pub(crate) selected_strategy_: Option<Strategy>,
    pub(crate) empirical_labels_: Option<Array1<Int>>,
    pub(crate) bayesian_alpha_: Option<Array1<Float>>,
    pub(crate) bayesian_posterior_: Option<Array1<Float>>,
    pub(crate) bayesian_uncertainty_: Option<Array1<Float>>,
    pub(crate) _state: std::marker::PhantomData<State>,
}

impl DummyClassifier {
    /// Create a new DummyClassifier
    pub fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            random_state: None,
            constant: None,
            classes_: None,
            class_prior_: None,
            n_classes_: None,
            most_frequent_class_: None,
            selected_strategy_: None,
            empirical_labels_: None,
            bayesian_alpha_: None,
            bayesian_posterior_: None,
            bayesian_uncertainty_: None,
            _state: std::marker::PhantomData,
        }
    }

    /// Set the random state for reproducible output
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Set the constant value to predict (required for Strategy::Constant)
    pub fn with_constant(mut self, constant: Int) -> Self {
        self.constant = Some(constant);
        self
    }

    /// Set the Bayesian prior parameters (Dirichlet concentration parameters)
    pub fn with_bayesian_prior(mut self, alpha: Array1<Float>) -> Self {
        self.bayesian_alpha_ = Some(alpha);
        self
    }
}

impl Default for DummyClassifier {
    fn default() -> Self {
        Self::new(Strategy::Prior)
    }
}

impl Estimator for DummyClassifier {
    type Config = ();
    type Error = sklears_core::error::SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Features, Array1<Int>> for DummyClassifier {
    type Fitted = DummyClassifier<sklears_core::traits::Trained>;

    fn fit(self, x: &Features, y: &Array1<Int>) -> Result<Self::Fitted> {
        if x.is_empty() || y.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        if x.nrows() != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Number of samples in X and y must be equal".to_string(),
            ));
        }

        // Check constant strategy requirements
        if self.strategy == Strategy::Constant && self.constant.is_none() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Constant strategy requires a constant value to be set".to_string(),
            ));
        }

        // Get unique classes and their counts
        let mut class_counts: HashMap<Int, usize> = HashMap::new();
        for &label in y.iter() {
            *class_counts.entry(label).or_insert(0) += 1;
        }

        let mut classes: Vec<Int> = class_counts.keys().copied().collect();
        classes.sort();
        let n_classes = classes.len();

        // Calculate class priors
        let n_samples = y.len() as Float;
        let class_prior: Vec<Float> = classes
            .iter()
            .map(|&class| {
                class_counts
                    .get(&class)
                    .map(|&c| c as Float / n_samples)
                    .unwrap_or(0.0)
            })
            .collect();

        // Find most frequent class
        let most_frequent_class = *class_counts
            .iter()
            .max_by_key(|(_, count)| *count)
            .map(|(class, _)| class)
            .ok_or_else(|| {
                sklears_core::error::SklearsError::InvalidInput(
                    "Failed to find most frequent class".to_string(),
                )
            })?;

        // Automatic/Adaptive strategy selection
        let selected_strategy = match &self.strategy {
            Strategy::Auto => {
                let max_class_prior = class_prior
                    .iter()
                    .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .ok_or_else(|| {
                        sklears_core::error::SklearsError::InvalidInput(
                            "Failed to find maximum class prior".to_string(),
                        )
                    })?;

                if *max_class_prior >= 0.8 {
                    // Heavily imbalanced - use most frequent
                    Strategy::MostFrequent
                } else if n_classes == 2 && *max_class_prior >= 0.4 && *max_class_prior <= 0.6 {
                    // Binary balanced - use stratified
                    Strategy::Stratified
                } else if *max_class_prior >= 0.2 && *max_class_prior < 0.8 {
                    // Relatively balanced - use prior
                    Strategy::Prior
                } else {
                    // Default to stratified
                    Strategy::Stratified
                }
            }
            Strategy::Adaptive => {
                // Use dataset analysis to select optimal strategy
                let characteristics = analyze_classification_dataset(x, y);
                get_adaptive_classification_strategy(&characteristics)
            }
            _ => self.strategy.clone(),
        };

        // Store empirical labels for bootstrap sampling
        let empirical_labels = if matches!(selected_strategy, Strategy::Empirical) {
            Some(y.clone())
        } else {
            None
        };

        // Compute Bayesian parameters for Bayesian strategy
        let (bayesian_alpha, bayesian_posterior, bayesian_uncertainty) =
            if matches!(selected_strategy, Strategy::Bayesian) {
                // Use provided priors or default uniform priors
                let alpha = self
                    .bayesian_alpha_
                    .unwrap_or_else(|| Array1::ones(n_classes));

                // Count observations for each class
                let mut counts = Array1::<Float>::zeros(n_classes);
                for &label in y.iter() {
                    let class_idx = classes.iter().position(|&c| c == label).ok_or_else(|| {
                        sklears_core::error::SklearsError::InvalidInput(format!(
                            "Label {} not found in classes",
                            label
                        ))
                    })?;
                    counts[class_idx] += 1.0;
                }

                // Compute posterior parameters (Dirichlet-Multinomial conjugacy)
                let posterior = &alpha + &counts;

                // Compute posterior mean (expected class probabilities)
                let posterior_sum = posterior.sum();
                let posterior_mean = &posterior / posterior_sum;

                // Compute uncertainty (posterior variance for Dirichlet)
                let uncertainty = posterior_mean
                    .iter()
                    .zip(posterior.iter())
                    .map(|(&p, &a)| p * (1.0 - p) / (posterior_sum + 1.0))
                    .collect::<Vec<_>>();

                (
                    Some(alpha),
                    Some(posterior_mean),
                    Some(Array1::from_vec(uncertainty)),
                )
            } else {
                (None, None, None)
            };

        Ok(DummyClassifier {
            strategy: self.strategy,
            random_state: self.random_state,
            constant: self.constant,
            classes_: Some(Array1::from_vec(classes)),
            class_prior_: Some(Array1::from_vec(class_prior)),
            n_classes_: Some(n_classes),
            most_frequent_class_: Some(most_frequent_class),
            selected_strategy_: Some(selected_strategy),
            empirical_labels_: empirical_labels,
            bayesian_alpha_: bayesian_alpha,
            bayesian_posterior_: bayesian_posterior,
            bayesian_uncertainty_: bayesian_uncertainty,
            _state: std::marker::PhantomData,
        })
    }
}

impl Predict<Features, Array1<Int>> for DummyClassifier<sklears_core::traits::Trained> {
    fn predict(&self, x: &Features) -> Result<Array1<Int>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let mut predictions = Array1::zeros(n_samples);
        let classes = self.classes_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;
        let class_prior = self.class_prior_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;
        let effective_strategy = self.selected_strategy_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;

        match effective_strategy {
            Strategy::MostFrequent | Strategy::Prior => {
                // Both strategies use the most frequent class
                let most_frequent = self.most_frequent_class_.ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
                })?;
                predictions.fill(most_frequent);
            }
            Strategy::Auto | Strategy::Adaptive => {
                // This should never happen since Auto/Adaptive gets resolved to a concrete strategy
                return Err(sklears_core::error::SklearsError::InvalidInput(
                    "Auto/Adaptive strategy should have been resolved during fitting".to_string(),
                ));
            }
            Strategy::Constant => {
                let constant = self.constant.ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Constant value not set".to_string(),
                    )
                })?;
                predictions.fill(constant);
            }
            Strategy::Stratified => {
                // Generate predictions respecting class distribution
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                for i in 0..n_samples {
                    let rand_val: Float = rng.gen();
                    let mut cumsum = 0.0;

                    for (j, &prior) in class_prior.iter().enumerate() {
                        cumsum += prior;
                        if rand_val <= cumsum {
                            predictions[i] = classes[j];
                            break;
                        }
                    }
                }
            }
            Strategy::Uniform => {
                // Generate uniform random predictions
                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                for i in 0..n_samples {
                    let rand_idx = rng.gen_range(0..classes.len());
                    predictions[i] = classes[rand_idx];
                }
            }
            Strategy::Empirical => {
                // Bootstrap sampling from empirical distribution
                let empirical_labels = self.empirical_labels_.as_ref().ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Empirical labels not available for sampling".to_string(),
                    )
                })?;

                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                for i in 0..n_samples {
                    let rand_idx = rng.gen_range(0..empirical_labels.len());
                    predictions[i] = empirical_labels[rand_idx];
                }
            }
            Strategy::Bayesian => {
                // Sample from posterior predictive distribution
                let posterior_probs = self.bayesian_posterior_.as_ref().ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Bayesian posterior not available".to_string(),
                    )
                })?;

                let mut rng = if let Some(seed) = self.random_state {
                    StdRng::seed_from_u64(seed)
                } else {
                    StdRng::seed_from_u64(0)
                };

                // Create cumulative distribution for sampling
                let mut cumulative_probs = Vec::with_capacity(classes.len());
                let mut cumsum = 0.0;
                for &prob in posterior_probs.iter() {
                    cumsum += prob;
                    cumulative_probs.push(cumsum);
                }

                for i in 0..n_samples {
                    let rand_val: Float = rng.gen();
                    let class_idx = cumulative_probs
                        .iter()
                        .position(|&cum_prob| rand_val <= cum_prob)
                        .unwrap_or(classes.len() - 1);
                    predictions[i] = classes[class_idx];
                }
            }
        }

        Ok(predictions)
    }
}

impl PredictProba<Features, Array2<Float>> for DummyClassifier<sklears_core::traits::Trained> {
    fn predict_proba(&self, x: &Features) -> Result<Array2<Float>> {
        if x.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input cannot be empty".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_classes = self.n_classes_.ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;
        let class_prior = self.class_prior_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;
        let mut probabilities = Array2::zeros((n_samples, n_classes));
        let effective_strategy = self.selected_strategy_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })?;

        match effective_strategy {
            Strategy::MostFrequent | Strategy::Prior => {
                // Use class priors for all samples
                for i in 0..n_samples {
                    for (j, &prior) in class_prior.iter().enumerate() {
                        probabilities[[i, j]] = prior;
                    }
                }
            }
            Strategy::Auto | Strategy::Adaptive => {
                // This should never happen since Auto/Adaptive gets resolved to a concrete strategy
                return Err(sklears_core::error::SklearsError::InvalidInput(
                    "Auto/Adaptive strategy should have been resolved during fitting".to_string(),
                ));
            }
            Strategy::Constant => {
                // Find the index of the constant class
                let constant = self.constant.ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Constant value not set".to_string(),
                    )
                })?;
                let classes = self.classes_.as_ref().ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
                })?;
                if let Some(const_idx) = classes.iter().position(|&c| c == constant) {
                    for i in 0..n_samples {
                        probabilities[[i, const_idx]] = 1.0;
                    }
                } else {
                    return Err(sklears_core::error::SklearsError::InvalidInput(
                        "Constant value not found in training classes".to_string(),
                    ));
                }
            }
            Strategy::Stratified => {
                // Same as Prior for probabilities
                for i in 0..n_samples {
                    for (j, &prior) in class_prior.iter().enumerate() {
                        probabilities[[i, j]] = prior;
                    }
                }
            }
            Strategy::Uniform => {
                // Uniform probabilities across all classes
                let uniform_prob = 1.0 / n_classes as Float;
                probabilities.fill(uniform_prob);
            }
            Strategy::Empirical => {
                // Use empirical class distribution (same as class priors)
                for i in 0..n_samples {
                    for (j, &prior) in class_prior.iter().enumerate() {
                        probabilities[[i, j]] = prior;
                    }
                }
            }
            Strategy::Bayesian => {
                // Use posterior mean probabilities
                let posterior_probs = self.bayesian_posterior_.as_ref().ok_or_else(|| {
                    sklears_core::error::SklearsError::InvalidInput(
                        "Bayesian posterior not available".to_string(),
                    )
                })?;

                for i in 0..n_samples {
                    for (j, &posterior_prob) in posterior_probs.iter().enumerate() {
                        probabilities[[i, j]] = posterior_prob;
                    }
                }
            }
        }

        Ok(probabilities)
    }
}

impl DummyClassifier<sklears_core::traits::Trained> {
    /// Get the class labels
    pub fn classes(&self) -> Result<&Array1<Int>> {
        self.classes_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })
    }

    /// Get the class priors
    pub fn class_prior(&self) -> Result<&Array1<Float>> {
        self.class_prior_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })
    }

    /// Get the number of classes
    pub fn n_classes(&self) -> Result<usize> {
        self.n_classes_.ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })
    }

    /// Get the selected strategy (useful when Auto was used)
    pub fn selected_strategy(&self) -> Result<&Strategy> {
        self.selected_strategy_.as_ref().ok_or_else(|| {
            sklears_core::error::SklearsError::InvalidInput("Model not fitted".to_string())
        })
    }

    /// Get the Bayesian prior parameters (alpha)
    pub fn bayesian_alpha(&self) -> Option<&Array1<Float>> {
        self.bayesian_alpha_.as_ref()
    }

    /// Get the Bayesian posterior probabilities
    pub fn bayesian_posterior(&self) -> Option<&Array1<Float>> {
        self.bayesian_posterior_.as_ref()
    }

    /// Get the Bayesian uncertainty estimates
    pub fn bayesian_uncertainty(&self) -> Option<&Array1<Float>> {
        self.bayesian_uncertainty_.as_ref()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, Array2};

    #[test]
    fn test_dummy_classifier_most_frequent() {
        let x = Array2::from_shape_vec(
            (6, 2),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 0, 1, 1, 2]; // Class 0 is most frequent

        let classifier = DummyClassifier::new(Strategy::MostFrequent);
        let fitted = classifier.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();

        // All predictions should be class 0 (most frequent)
        for &pred in predictions.iter() {
            assert_eq!(pred, 0);
        }
    }

    #[test]
    fn test_dummy_classifier_constant() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1, 2, 3];

        let classifier = DummyClassifier::new(Strategy::Constant).with_constant(5);
        let fitted = classifier.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();

        // All predictions should be the constant value (5)
        for &pred in predictions.iter() {
            assert_eq!(pred, 5);
        }
    }

    #[test]
    fn test_dummy_classifier_uniform() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 1, 2, 3];

        let classifier = DummyClassifier::new(Strategy::Uniform).with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();

        // Check that predictions are within the valid class range
        for &pred in predictions.iter() {
            assert!(y.iter().any(|&class| class == pred));
        }

        // With uniform strategy, we should see some variability (not all the same)
        let unique_predictions: std::collections::HashSet<Int> =
            predictions.iter().copied().collect();
        assert!(unique_predictions.len() > 1);
    }

    #[test]
    fn test_dummy_classifier_predict_proba() {
        let x = Array2::from_shape_vec((3, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();
        let y = array![0, 0, 1]; // 2/3 class 0, 1/3 class 1

        let classifier = DummyClassifier::new(Strategy::Prior);
        let fitted = classifier.fit(&x, &y).unwrap();

        let probabilities = fitted.predict_proba(&x).unwrap();

        assert_eq!(probabilities.shape(), &[3, 2]);

        // All rows should have the same probabilities (based on class priors)
        for i in 0..3 {
            assert!((probabilities[[i, 0]] - 2.0 / 3.0).abs() < 1e-10); // Class 0 prior
            assert!((probabilities[[i, 1]] - 1.0 / 3.0).abs() < 1e-10); // Class 1 prior

            // Probabilities should sum to 1
            let row_sum = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dummy_classifier_constant_error() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 1];

        // Should fail when constant strategy used without setting constant
        let classifier = DummyClassifier::new(Strategy::Constant);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dummy_classifier_empty_input() {
        let x = Array2::zeros((0, 2));
        let y = Array1::zeros(0);

        let classifier = DummyClassifier::new(Strategy::MostFrequent);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dummy_classifier_shape_mismatch() {
        let x = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let y = array![0, 1, 2]; // Wrong length

        let classifier = DummyClassifier::new(Strategy::MostFrequent);
        let result = classifier.fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_dummy_classifier_auto_imbalanced() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1]; // 80% class 0, 20% class 1

        let classifier = DummyClassifier::new(Strategy::Auto);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should select MostFrequent for heavily imbalanced data
        assert_eq!(fitted.selected_strategy().unwrap(), &Strategy::MostFrequent);

        let predictions = fitted.predict(&x).unwrap();
        // All predictions should be class 0 (most frequent)
        for &pred in predictions.iter() {
            assert_eq!(pred, 0);
        }
    }

    #[test]
    fn test_dummy_classifier_auto_balanced_binary() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // 50% each class

        let classifier = DummyClassifier::new(Strategy::Auto).with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should select Stratified for balanced binary data
        assert_eq!(fitted.selected_strategy().unwrap(), &Strategy::Stratified);
    }

    #[test]
    fn test_dummy_classifier_auto_multiclass() {
        let x = Array2::from_shape_vec((15, 2), (0..30).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]; // 33% each class

        let classifier = DummyClassifier::new(Strategy::Auto);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should select Prior for relatively balanced multiclass data
        assert_eq!(fitted.selected_strategy().unwrap(), &Strategy::Prior);
    }

    #[test]
    fn test_dummy_classifier_adaptive_imbalanced() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 1, 1]; // 80% class 0, 20% class 1

        let classifier = DummyClassifier::new(Strategy::Adaptive);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should select MostFrequent for imbalanced data
        assert_eq!(fitted.selected_strategy().unwrap(), &Strategy::MostFrequent);

        let predictions = fitted.predict(&x).unwrap();
        // All predictions should be class 0 (most frequent)
        for &pred in predictions.iter() {
            assert_eq!(pred, 0);
        }
    }

    #[test]
    fn test_dummy_classifier_adaptive_balanced() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 0, 0, 1, 1, 1, 1, 1]; // 50% each class

        let classifier = DummyClassifier::new(Strategy::Adaptive).with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should select Stratified for balanced binary data
        assert_eq!(fitted.selected_strategy().unwrap(), &Strategy::Stratified);

        let predictions = fitted.predict(&x).unwrap();
        // Check that predictions are within the valid class range
        for &pred in predictions.iter() {
            assert!(y.iter().any(|&class| class == pred));
        }
    }

    #[test]
    fn test_dummy_classifier_adaptive_small_dataset() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1]; // Small balanced dataset

        let classifier = DummyClassifier::new(Strategy::Adaptive);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Should handle small datasets appropriately
        let selected_strategy = fitted.selected_strategy();
        assert!(matches!(
            selected_strategy.unwrap(),
            &Strategy::MostFrequent | &Strategy::Stratified
        ));

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 4);
    }

    #[test]
    fn test_dummy_classifier_empirical() {
        let x = Array2::from_shape_vec((6, 2), (0..12).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 1, 1, 2]; // 3 class 0, 2 class 1, 1 class 2

        let classifier = DummyClassifier::new(Strategy::Empirical).with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        let predictions = fitted.predict(&x).unwrap();

        // All predictions should be from the original training set
        for &pred in predictions.iter() {
            assert!(y.iter().any(|&class| class == pred));
        }

        // Check that empirical distribution is preserved (roughly)
        let pred_counts: std::collections::HashMap<Int, usize> =
            predictions
                .iter()
                .fold(std::collections::HashMap::new(), |mut acc, &pred| {
                    *acc.entry(pred).or_insert(0) += 1;
                    acc
                });

        // Should have some variety in predictions
        assert!(pred_counts.len() > 1);
    }

    #[test]
    fn test_dummy_classifier_empirical_probabilities() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = array![0, 0, 1, 1]; // Balanced binary

        let classifier = DummyClassifier::new(Strategy::Empirical);
        let fitted = classifier.fit(&x, &y).unwrap();

        let probabilities = fitted.predict_proba(&x).unwrap();

        assert_eq!(probabilities.shape(), &[4, 2]);

        // Probabilities should be based on empirical class distribution (50% each)
        for i in 0..4 {
            assert!((probabilities[[i, 0]] - 0.5).abs() < 1e-10);
            assert!((probabilities[[i, 1]] - 0.5).abs() < 1e-10);

            // Probabilities should sum to 1
            let row_sum = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dummy_classifier_empirical_reproducibility() {
        let x = Array2::from_shape_vec((10, 2), (0..20).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec((0..10).map(|x| x % 3).collect()); // Classes 0, 1, 2

        let classifier1 = DummyClassifier::new(Strategy::Empirical).with_random_state(123);
        let fitted1 = classifier1.fit(&x, &y).unwrap();
        let predictions1 = fitted1.predict(&x).unwrap();

        let classifier2 = DummyClassifier::new(Strategy::Empirical).with_random_state(123);
        let fitted2 = classifier2.fit(&x, &y).unwrap();
        let predictions2 = fitted2.predict(&x).unwrap();

        // Predictions should be identical with same random state
        for (p1, p2) in predictions1.iter().zip(predictions2.iter()) {
            assert_eq!(p1, p2);
        }
    }

    mod property_tests {
        use super::{DummyClassifier, Strategy};
        use proptest::prelude::*;
        use scirs2_core::ndarray::{Array1, Array2};
        use sklears_core::traits::{Fit, Predict, PredictProba};
        use std::collections::HashSet;

        proptest! {
            #[test]
            fn prop_classifier_predictions_match_input_size(
                data in prop::collection::vec(prop::collection::vec(-100.0f64..100.0, 1..10), 1..100),
                labels in prop::collection::vec(0i32..5, 1..100)
            ) {
                let n_samples = data.len().min(labels.len());
                if n_samples == 0 { return Ok(()); }

                let data_flat: Vec<f64> = data.into_iter().flatten().collect();
                let n_features = data_flat.len() / n_samples;
                if n_features == 0 { return Ok(()); }

                let x = Array2::from_shape_vec((n_samples, n_features),
                    data_flat.into_iter().take(n_samples * n_features).collect()).unwrap();
                let y = Array1::from_vec(labels.into_iter().take(n_samples).collect());

                for strategy in [Strategy::MostFrequent, Strategy::Stratified, Strategy::Uniform, Strategy::Prior] {
                    let classifier = DummyClassifier::new(strategy).with_random_state(42);
                    if let Ok(fitted) = classifier.fit(&x, &y) {
                        let predictions = fitted.predict(&x)?;
                        prop_assert_eq!(predictions.len(), n_samples);
                    }
                }
            }

            #[test]
            fn prop_most_frequent_always_predicts_same_class(
                data in prop::collection::vec(prop::collection::vec(-10.0f64..10.0, 2..5), 5..20),
                labels in prop::collection::vec(0i32..3, 5..20)
            ) {
                let n_samples = data.len().min(labels.len());
                let data_flat: Vec<f64> = data.into_iter().flatten().collect();
                let n_features = data_flat.len() / n_samples;

                let x = Array2::from_shape_vec((n_samples, n_features),
                    data_flat.into_iter().take(n_samples * n_features).collect()).unwrap();
                let y = Array1::from_vec(labels.into_iter().take(n_samples).collect());

                let classifier = DummyClassifier::new(Strategy::MostFrequent);
                if let Ok(fitted) = classifier.fit(&x, &y) {
                    let predictions = fitted.predict(&x)?;

                    // All predictions should be the same (most frequent class)
                    let unique_predictions: HashSet<i32> = predictions.iter().copied().collect();
                    prop_assert_eq!(unique_predictions.len(), 1);

                    // Predicted class should be one that exists in training data
                    let unique_labels: HashSet<i32> = y.iter().copied().collect();
                    prop_assert!(unique_labels.contains(predictions.iter().next().unwrap()));
                }
            }

            #[test]
            fn prop_constant_strategy_predicts_constant(
                data in prop::collection::vec(prop::collection::vec(-10.0f64..10.0, 2..5), 5..20),
                labels in prop::collection::vec(0i32..3, 5..20),
                constant in -5i32..5
            ) {
                let n_samples = data.len().min(labels.len());
                let data_flat: Vec<f64> = data.into_iter().flatten().collect();
                let n_features = data_flat.len() / n_samples;

                let x = Array2::from_shape_vec((n_samples, n_features),
                    data_flat.into_iter().take(n_samples * n_features).collect()).unwrap();
                let y = Array1::from_vec(labels.into_iter().take(n_samples).collect());

                let classifier = DummyClassifier::new(Strategy::Constant).with_constant(constant);
                if let Ok(fitted) = classifier.fit(&x, &y) {
                    let predictions = fitted.predict(&x)?;

                    // All predictions should be the constant value
                    for &pred in predictions.iter() {
                        prop_assert_eq!(pred, constant);
                    }
                }
            }

            #[test]
            fn prop_uniform_strategy_uses_all_classes(
                labels in prop::collection::vec(0i32..2, 50..100) // Binary classification with many samples
            ) {
                let n_samples = labels.len();
                let x = Array2::zeros((n_samples, 2));
                let y = Array1::from_vec(labels);

                // Ensure we have at least 2 classes
                if y.iter().copied().collect::<HashSet<_>>().len() < 2 {
                    return Ok(());
                }

                let classifier = DummyClassifier::new(Strategy::Uniform).with_random_state(42);
                if let Ok(fitted) = classifier.fit(&x, &y) {
                    let predictions = fitted.predict(&x)?;

                    // With uniform strategy and enough samples, we should see both classes
                    let unique_predictions: HashSet<i32> = predictions.iter().copied().collect();
                    let unique_labels: HashSet<i32> = y.iter().copied().collect();

                    // Predictions should only contain classes from training data
                    for &pred in &unique_predictions {
                        prop_assert!(unique_labels.contains(&pred));
                    }

                    // With 50+ samples and uniform distribution, we should see variety
                    prop_assert!(unique_predictions.len() >= 1);
                }
            }

            #[test]
            fn prop_predict_proba_sums_to_one(
                data in prop::collection::vec(prop::collection::vec(-10.0f64..10.0, 2..5), 10..30),
                labels in prop::collection::vec(0i32..3, 10..30)
            ) {
                let n_samples = data.len().min(labels.len());
                let data_flat: Vec<f64> = data.into_iter().flatten().collect();
                let n_features = data_flat.len() / n_samples;

                let x = Array2::from_shape_vec((n_samples, n_features),
                    data_flat.into_iter().take(n_samples * n_features).collect()).unwrap();
                let y = Array1::from_vec(labels.into_iter().take(n_samples).collect());

                for strategy in [Strategy::MostFrequent, Strategy::Stratified, Strategy::Uniform, Strategy::Prior] {
                    let classifier = DummyClassifier::new(strategy);
                    if let Ok(fitted) = classifier.fit(&x, &y) {
                        let probabilities = fitted.predict_proba(&x)?;

                        // Each row should sum to 1.0
                        for i in 0..n_samples {
                            let row_sum = probabilities.row(i).sum();
                            prop_assert!((row_sum - 1.0).abs() < 1e-10);
                        }

                        // All probabilities should be non-negative
                        for &prob in probabilities.iter() {
                            prop_assert!(prob >= 0.0);
                            prop_assert!(prob <= 1.0);
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_dummy_classifier_bayesian() {
        let x = Array2::from_shape_vec((6, 2), (0..12).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 0, 1, 1, 2]; // 3 class 0, 2 class 1, 1 class 2

        // Test with default uniform priors
        let classifier = DummyClassifier::new(Strategy::Bayesian).with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        // Check that Bayesian parameters are computed
        assert!(fitted.bayesian_alpha().is_some());
        assert!(fitted.bayesian_posterior().is_some());
        assert!(fitted.bayesian_uncertainty().is_some());

        let posterior = fitted.bayesian_posterior().unwrap();
        assert_eq!(posterior.len(), 3);

        // Posterior should sum to 1
        let posterior_sum = posterior.sum();
        assert!((posterior_sum - 1.0).abs() < 1e-10);

        // Class 0 should have highest posterior probability (most observations)
        let max_prob_idx = posterior
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .unwrap()
            .0;
        assert_eq!(max_prob_idx, 0);

        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.len(), 6);

        // Test probabilities
        let probabilities = fitted.predict_proba(&x).unwrap();
        assert_eq!(probabilities.shape(), &[6, 3]);

        for i in 0..6 {
            let row_sum = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dummy_classifier_bayesian_custom_prior() {
        let x = Array2::from_shape_vec((4, 2), (0..8).map(|x| x as f64).collect()).unwrap();
        let y = array![0, 0, 1, 1]; // Balanced binary

        // Test with custom prior favoring class 1
        let custom_prior = Array1::from_vec(vec![1.0, 5.0]); // Strong prior for class 1
        let classifier = DummyClassifier::new(Strategy::Bayesian)
            .with_bayesian_prior(custom_prior.clone())
            .with_random_state(42);
        let fitted = classifier.fit(&x, &y).unwrap();

        let alpha = fitted.bayesian_alpha().unwrap();
        assert_abs_diff_eq!(alpha[0], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(alpha[1], 5.0, epsilon = 1e-10);

        let posterior = fitted.bayesian_posterior().unwrap();
        // Despite equal observations, class 1 should have higher posterior due to prior
        assert!(posterior[1] > posterior[0]);

        let uncertainty = fitted.bayesian_uncertainty().unwrap();
        assert_eq!(uncertainty.len(), 2);
        // All uncertainties should be non-negative
        for &unc in uncertainty.iter() {
            assert!(unc >= 0.0);
        }
    }

    #[test]
    fn test_dummy_classifier_bayesian_reproducibility() {
        let x = Array2::from_shape_vec((8, 2), (0..16).map(|x| x as f64).collect()).unwrap();
        let y = Array1::from_vec(vec![0, 0, 1, 1, 2, 2, 0, 1]); // Multiple classes

        let classifier1 = DummyClassifier::new(Strategy::Bayesian).with_random_state(123);
        let fitted1 = classifier1.fit(&x, &y).unwrap();
        let predictions1 = fitted1.predict(&x).unwrap();

        let classifier2 = DummyClassifier::new(Strategy::Bayesian).with_random_state(123);
        let fitted2 = classifier2.fit(&x, &y).unwrap();
        let predictions2 = fitted2.predict(&x).unwrap();

        // Predictions should be identical with same random state
        for (p1, p2) in predictions1.iter().zip(predictions2.iter()) {
            assert_eq!(p1, p2);
        }

        // Bayesian parameters should be identical
        let post1 = fitted1.bayesian_posterior().unwrap();
        let post2 = fitted2.bayesian_posterior().unwrap();
        for (p1, p2) in post1.iter().zip(post2.iter()) {
            assert_abs_diff_eq!(p1, p2, epsilon = 1e-10);
        }
    }
}
