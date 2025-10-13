//! Multinomial Naive Bayes classifier implementation

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{safe_log, NaiveBayesMixin};

/// Configuration for Multinomial Naive Bayes
#[derive(Debug, Clone)]
pub struct MultinomialNBConfig {
    /// Additive (Laplace/Lidstone) smoothing parameter
    pub alpha: f64,
    /// Whether to learn class prior probabilities
    pub fit_prior: bool,
    /// Prior probabilities of the classes
    pub class_prior: Option<Array1<f64>>,
}

impl Default for MultinomialNBConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            class_prior: None,
        }
    }
}

/// Multinomial Naive Bayes classifier
///
/// For discrete count data like text classification.
#[derive(Debug, Clone)]
pub struct MultinomialNB<State = Untrained> {
    config: MultinomialNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    feature_log_prob_: Option<Array2<f64>>,
    class_log_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
    n_features_: Option<usize>,
}

impl MultinomialNB<Untrained> {
    /// Create a new Multinomial Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: MultinomialNBConfig::default(),
            state: PhantomData,
            feature_log_prob_: None,
            class_log_prior_: None,
            classes_: None,
            n_features_: None,
        }
    }

    /// Set smoothing parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be >= 0");
        }
        self.config.alpha = alpha;
        self
    }

    /// Set whether to learn class prior probabilities
    pub fn fit_prior(mut self, fit_prior: bool) -> Self {
        self.config.fit_prior = fit_prior;
        self
    }

    /// Set class prior probabilities
    pub fn class_prior(mut self, class_prior: Array1<f64>) -> Self {
        self.config.class_prior = Some(class_prior);
        self
    }
}

impl Default for MultinomialNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MultinomialNB<Untrained> {
    type Float = Float;
    type Config = MultinomialNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MultinomialNB<Untrained> {
    type Fitted = MultinomialNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check for negative values
        if x.iter().any(|&val| val < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Multinomial Naive Bayes requires non-negative features".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Initialize feature counts
        let mut feature_count: Array2<f64> = Array2::zeros((n_classes, n_features));
        let mut class_count: Array1<f64> = Array1::zeros(n_classes);

        // Count features for each class
        for (i, &label) in y.iter().enumerate() {
            let class_idx = classes.iter().position(|&c| c == label).unwrap();

            let sample = x.row(i);
            let new_counts = &feature_count.row(class_idx) + &sample;
            feature_count.row_mut(class_idx).assign(&new_counts);
            class_count[class_idx] += sample.sum();
        }

        // Compute log probabilities
        let smoothed_fc = &feature_count + self.config.alpha;
        let smoothed_cc = &class_count + self.config.alpha * n_features as f64;

        let mut feature_log_prob = Array2::zeros((n_classes, n_features));
        for i in 0..n_classes {
            for j in 0..n_features {
                feature_log_prob[[i, j]] = safe_log(smoothed_fc[[i, j]] / smoothed_cc[i]);
            }
        }

        // Compute class log priors
        let class_log_prior = if self.config.fit_prior {
            if let Some(ref priors) = self.config.class_prior {
                priors.mapv(safe_log)
            } else {
                let total_count = class_count.sum();
                if total_count == 0.0 {
                    Array1::from_elem(n_classes, -(n_classes as f64).ln())
                } else {
                    class_count.mapv(|c| safe_log(c / total_count))
                }
            }
        } else {
            Array1::from_elem(n_classes, -(n_classes as f64).ln())
        };

        Ok(MultinomialNB {
            config: self.config,
            state: PhantomData,
            feature_log_prob_: Some(feature_log_prob),
            class_log_prior_: Some(class_log_prior),
            classes_: Some(classes),
            n_features_: Some(n_features),
        })
    }
}

impl MultinomialNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();
        let class_log_prior = self.class_log_prior_.as_ref().unwrap();

        // Check for negative values
        if x.iter().any(|&val| val < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Multinomial Naive Bayes requires non-negative features".to_string(),
            ));
        }

        // Compute log likelihood: X @ feature_log_prob.T + class_log_prior
        let log_likelihood = x.dot(&feature_log_prob.t());
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            for j in 0..n_classes {
                joint_log_likelihood[[i, j]] = log_likelihood[[i, j]] + class_log_prior[j];
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MultinomialNB<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let classes = self.classes_.as_ref().unwrap();

        // Find the class with maximum log probability for each sample
        Ok(log_prob.map_axis(Axis(1), |row| {
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

impl PredictProba<Array2<Float>, Array2<f64>> for MultinomialNB<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities
        for i in 0..n_samples {
            let row = log_prob.row(i);
            let max_log_prob = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            // Compute exp(log_prob - max_log_prob) for numerical stability
            let mut exp_sum = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_prob[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            // Normalize
            for j in 0..n_classes {
                proba[[i, j]] /= exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Score<Array2<Float>, Array1<i32>> for MultinomialNB<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<i32>) -> Result<f64> {
        let predictions = self.predict(x)?;
        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        Ok(correct as f64 / y.len() as f64)
    }
}

impl NaiveBayesMixin for MultinomialNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_log_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        self.feature_log_prob_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
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
    fn test_multinomial_nb_basic() {
        // Simple count data (e.g., word counts in documents)
        let x = array![
            [3.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0]
        ];
        let y = array![0, 0, 1, 1];

        let model = MultinomialNB::new().alpha(1.0).fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_multinomial_nb_smoothing() {
        // Test with alpha = 0 (no smoothing) and some zero counts
        let x = array![[2.0, 0.0], [1.0, 1.0], [0.0, 2.0], [0.0, 3.0]];
        let y = array![0, 0, 1, 1];

        let model = MultinomialNB::new()
            .alpha(0.1) // Small smoothing
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_multinomial_nb_predict_proba() {
        let x = array![[2.0, 1.0], [1.0, 2.0], [3.0, 0.0], [0.0, 3.0]];
        let y = array![0, 1, 0, 1];

        let model = MultinomialNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    #[should_panic(expected = "alpha must be >= 0")]
    fn test_multinomial_nb_negative_alpha() {
        MultinomialNB::new().alpha(-1.0);
    }

    #[test]
    fn test_multinomial_nb_negative_features() {
        let x = array![
            [1.0, -1.0], // Negative feature
            [2.0, 3.0]
        ];
        let y = array![0, 1];

        let result = MultinomialNB::new().fit(&x, &y);
        assert!(result.is_err());
    }
}
