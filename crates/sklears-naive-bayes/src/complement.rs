//! Complement Naive Bayes classifier implementation

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

/// Configuration for Complement Naive Bayes
#[derive(Debug, Clone)]
pub struct ComplementNBConfig {
    /// Additive (Laplace/Lidstone) smoothing parameter
    pub alpha: f64,
    /// Whether to learn class prior probabilities
    pub fit_prior: bool,
    /// Whether to normalize weights
    pub norm: bool,
    /// Prior probabilities of the classes
    pub class_prior: Option<Array1<f64>>,
}

impl Default for ComplementNBConfig {
    fn default() -> Self {
        Self {
            alpha: 1.0,
            fit_prior: true,
            norm: false,
            class_prior: None,
        }
    }
}

/// Complement Naive Bayes classifier
///
/// The Complement Naive Bayes classifier was designed to correct the "severe
/// assumptions" made by the standard Multinomial Naive Bayes classifier.
/// It is particularly suited for imbalanced data sets.
#[derive(Debug, Clone)]
pub struct ComplementNB<State = Untrained> {
    config: ComplementNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    feature_log_prob_: Option<Array2<f64>>,
    class_log_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
    n_features_: Option<usize>,
}

impl ComplementNB<Untrained> {
    /// Create a new Complement Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: ComplementNBConfig::default(),
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

    /// Set whether to normalize weights
    pub fn norm(mut self, norm: bool) -> Self {
        self.config.norm = norm;
        self
    }

    /// Set class prior probabilities
    pub fn class_prior(mut self, class_prior: Array1<f64>) -> Self {
        self.config.class_prior = Some(class_prior);
        self
    }
}

impl Default for ComplementNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ComplementNB<Untrained> {
    type Float = Float;
    type Config = ComplementNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for ComplementNB<Untrained> {
    type Fitted = ComplementNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check for negative values
        if x.iter().any(|&val| val < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Complement Naive Bayes requires non-negative features".to_string(),
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

        // Compute complement counts (all features not in the class)
        let feature_all = feature_count.sum_axis(Axis(0));
        let mut complement_count = Array2::zeros((n_classes, n_features));
        for i in 0..n_classes {
            for j in 0..n_features {
                complement_count[[i, j]] = feature_all[j] - feature_count[[i, j]];
            }
        }

        // Apply smoothing
        let smoothed_cc = &complement_count + self.config.alpha;
        let smoothed_sum = smoothed_cc.sum_axis(Axis(1));

        // Compute log probabilities
        let mut logged = Array2::zeros((n_classes, n_features));
        for i in 0..n_classes {
            for j in 0..n_features {
                logged[[i, j]] = safe_log(smoothed_cc[[i, j]] / smoothed_sum[i]);
            }
        }

        // Apply normalization if requested
        let feature_log_prob = if self.config.norm {
            let summed = logged.sum_axis(Axis(1));
            let mut normalized = Array2::zeros((n_classes, n_features));
            for i in 0..n_classes {
                for j in 0..n_features {
                    normalized[[i, j]] = logged[[i, j]] / summed[i];
                }
            }
            normalized
        } else {
            // ComplementNB uses negative of logged values
            -&logged
        };

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

        Ok(ComplementNB {
            config: self.config,
            state: PhantomData,
            feature_log_prob_: Some(feature_log_prob),
            class_log_prior_: Some(class_log_prior),
            classes_: Some(classes),
            n_features_: Some(n_features),
        })
    }
}

impl ComplementNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let feature_log_prob = self.feature_log_prob_.as_ref().unwrap();
        let class_log_prior = self.class_log_prior_.as_ref().unwrap();

        // Check for negative values
        if x.iter().any(|&val| val < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Complement Naive Bayes requires non-negative features".to_string(),
            ));
        }

        // Compute log likelihood: X @ feature_log_prob.T
        let jll = x.dot(&feature_log_prob.t());

        // Add class log prior only if we have a single class (edge case)
        let n_classes = self.classes_.as_ref().unwrap().len();
        if n_classes == 1 {
            let n_samples = x.nrows();
            let mut result = Array2::zeros((n_samples, 1));
            for i in 0..n_samples {
                result[[i, 0]] = jll[[i, 0]] + class_log_prior[0];
            }
            Ok(result)
        } else {
            Ok(jll)
        }
    }
}

impl Predict<Array2<Float>, Array1<i32>> for ComplementNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for ComplementNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for ComplementNB<Trained> {
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

impl NaiveBayesMixin for ComplementNB<Trained> {
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
    fn test_complement_nb_creation() {
        let nb = ComplementNB::new();
        assert_eq!(nb.config.alpha, 1.0);
        assert_eq!(nb.config.fit_prior, true);
        assert_eq!(nb.config.norm, false);
    }

    #[test]
    fn test_complement_nb_basic() {
        // Simple count data (e.g., word counts in documents)
        // Class 0 is the minority class
        let x = array![
            [3.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [0.0, 2.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 1.0, 0.0, 4.0]
        ];
        let y = array![0, 0, 1, 1, 1, 1]; // Imbalanced: 2 samples of class 0, 4 of class 1

        let model = ComplementNB::new().alpha(1.0).fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();

        // Test score - should perform well on imbalanced data
        let score = model.score(&x, &y).unwrap();
        assert!(score >= 0.8); // Should get most predictions right
    }

    #[test]
    fn test_complement_nb_with_normalization() {
        let x = array![[2.0, 1.0], [1.0, 2.0], [3.0, 0.0], [0.0, 3.0]];
        let y = array![0, 1, 0, 1];

        // Test with normalization
        let model = ComplementNB::new().norm(true).fit(&x, &y).unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_complement_nb_predict_proba() {
        let x = array![[2.0, 1.0], [1.0, 2.0], [3.0, 0.0], [0.0, 3.0]];
        let y = array![0, 1, 0, 1];

        let model = ComplementNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_complement_nb_negative_features() {
        let x = array![
            [1.0, -1.0], // Negative feature
            [2.0, 3.0]
        ];
        let y = array![0, 1];

        let result = ComplementNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_complement_nb_imbalanced_dataset() {
        // Highly imbalanced dataset
        let x = array![
            [5.0, 1.0, 0.0],
            [4.0, 2.0, 0.0],
            [0.0, 3.0, 5.0],
            [0.0, 2.0, 4.0],
            [0.0, 1.0, 3.0],
            [1.0, 1.0, 2.0],
            [0.0, 0.0, 2.0],
            [0.0, 0.0, 1.0]
        ];
        let y = array![0, 0, 1, 1, 1, 1, 1, 1]; // 2 of class 0, 6 of class 1

        let model = ComplementNB::new().alpha(0.5).fit(&x, &y).unwrap();

        // Should still perform reasonably well
        let score = model.score(&x, &y).unwrap();
        assert!(score >= 0.75);
    }

    #[test]
    fn test_complement_nb_single_class_edge_case() {
        // Edge case with only one class
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0]];
        let y = array![0, 0, 0]; // All same class

        let model = ComplementNB::new().fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();

        // Should predict the only class for all samples
        assert_eq!(predictions, y);
    }

    #[test]
    #[should_panic(expected = "alpha must be >= 0")]
    fn test_complement_nb_negative_alpha() {
        ComplementNB::new().alpha(-1.0);
    }
}
