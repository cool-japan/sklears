//! Poisson Naive Bayes classifier implementation

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{smoothing::enhanced_log, NaiveBayesMixin};

/// Configuration for Poisson Naive Bayes
#[derive(Debug, Clone)]
pub struct PoissonNBConfig {
    /// Additive smoothing parameter for zero counts
    pub alpha: f64,
    /// Whether to learn class prior probabilities
    pub fit_prior: bool,
    /// Prior probabilities of the classes
    pub class_prior: Option<Array1<f64>>,
}

impl Default for PoissonNBConfig {
    fn default() -> Self {
        Self {
            alpha: 1e-10,
            fit_prior: true,
            class_prior: None,
        }
    }
}

/// Poisson Naive Bayes classifier
///
/// Suitable for count data where features follow a Poisson distribution.
/// Each feature represents counts of events (e.g., word occurrences, user actions).
#[derive(Debug, Clone)]
pub struct PoissonNB<State = Untrained> {
    config: PoissonNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    theta_: Option<Array2<f64>>, // Poisson rates (lambda) for each feature per class
    class_log_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
    n_features_: Option<usize>,
}

impl PoissonNB<Untrained> {
    /// Create a new Poisson Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: PoissonNBConfig::default(),
            state: PhantomData,
            theta_: None,
            class_log_prior_: None,
            classes_: None,
            n_features_: None,
        }
    }

    /// Set smoothing parameter for zero counts
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

impl Default for PoissonNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PoissonNB<Untrained> {
    type Float = Float;
    type Config = PoissonNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for PoissonNB<Untrained> {
    type Fitted = PoissonNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check for negative values
        if x.iter().any(|&val| val < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Poisson Naive Bayes requires non-negative count features".to_string(),
            ));
        }

        // Check for non-integer values (Poisson is for counts)
        for &val in x.iter() {
            if val.fract() != 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Poisson Naive Bayes expects integer count features".to_string(),
                ));
            }
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Initialize parameters
        let mut theta = Array2::zeros((n_classes, n_features));
        let mut class_count = Array1::zeros(n_classes);

        // Compute Poisson rates (lambda) for each class and feature
        for (class_idx, &class_label) in classes.iter().enumerate() {
            // Get samples belonging to this class
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if mask.is_empty() {
                continue;
            }

            class_count[class_idx] = mask.len() as f64;

            // Select rows belonging to this class
            let x_class = x.select(Axis(0), &mask);

            // Compute Poisson rate (lambda) for each feature
            // For Poisson distribution: lambda = mean of the observations
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let mean = feature_values.mean().unwrap_or(0.0);

                // Add smoothing for numerical stability
                theta[[class_idx, feature_idx]] = mean + self.config.alpha;
            }
        }

        // Compute class log priors
        let class_log_prior = if self.config.fit_prior {
            if let Some(ref priors) = self.config.class_prior {
                if priors.len() != n_classes {
                    return Err(SklearsError::InvalidInput(format!(
                        "Number of priors ({}) doesn't match number of classes ({})",
                        priors.len(),
                        n_classes
                    )));
                }
                let sum = priors.sum();
                if (sum - 1.0).abs() > 1e-10 {
                    return Err(SklearsError::InvalidInput(
                        "The sum of the priors should be 1.0".to_string(),
                    ));
                }
                priors.mapv(enhanced_log)
            } else {
                let n_samples = y.len() as f64;
                class_count.mapv(|c| enhanced_log(c / n_samples))
            }
        } else {
            Array1::from_elem(n_classes, -(n_classes as f64).ln())
        };

        Ok(PoissonNB {
            config: self.config,
            state: PhantomData,
            theta_: Some(theta),
            class_log_prior_: Some(class_log_prior),
            classes_: Some(classes),
            n_features_: Some(n_features),
        })
    }
}

impl PoissonNB<Trained> {
    /// Compute log probability mass function for Poisson distribution
    fn poisson_log_pmf(k: f64, lambda: f64) -> f64 {
        if k < 0.0 || k.fract() != 0.0 {
            return f64::NEG_INFINITY;
        }

        if lambda <= 0.0 {
            return if k == 0.0 { 0.0 } else { f64::NEG_INFINITY };
        }

        // Poisson PMF: P(k; λ) = (λ^k * e^(-λ)) / k!
        // Log PMF: log(P(k; λ)) = k * log(λ) - λ - log(k!)

        // Use log-gamma function for log(k!) = log(Γ(k+1))
        let log_k_factorial = if k == 0.0 {
            0.0
        } else {
            (1..=(k as i32)).map(|i| (i as f64).ln()).sum::<f64>()
        };

        k * lambda.ln() - lambda - log_k_factorial
    }

    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let theta = self.theta_.as_ref().unwrap();
        let class_log_prior = self.class_log_prior_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();

        // Check for valid count data
        for &val in x.iter() {
            if val < 0.0 || val.fract() != 0.0 {
                return Err(SklearsError::InvalidInput(
                    "Poisson Naive Bayes requires non-negative integer count features".to_string(),
                ));
            }
        }

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for sample_idx in 0..n_samples {
            let sample = x.row(sample_idx);

            for class_idx in 0..n_classes {
                let mut log_prob = class_log_prior[class_idx];

                // Compute log likelihood for each feature using Poisson PMF
                for (feature_idx, &count) in sample.iter().enumerate() {
                    let lambda = theta[[class_idx, feature_idx]];
                    log_prob += Self::poisson_log_pmf(count, lambda);
                }

                joint_log_likelihood[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for PoissonNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for PoissonNB<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities using log-sum-exp for numerical stability
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

impl Score<Array2<Float>, Array1<i32>> for PoissonNB<Trained> {
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

impl NaiveBayesMixin for PoissonNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_log_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For Poisson NB, this returns the log of the rate parameters (theta)
        self.theta_.as_ref().unwrap()
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
    fn test_poisson_nb_creation() {
        let nb = PoissonNB::new();
        assert_abs_diff_eq!(nb.config.alpha, 1e-10, epsilon = 1e-12);
        assert_eq!(nb.config.fit_prior, true);
    }

    #[test]
    fn test_poisson_log_pmf() {
        // Test Poisson PMF computation
        let lambda = 2.0;

        // P(0; 2) = e^(-2) ≈ 0.1353
        let log_prob_0 = PoissonNB::<Trained>::poisson_log_pmf(0.0, lambda);
        assert_abs_diff_eq!(log_prob_0, -2.0, epsilon = 1e-10);

        // P(1; 2) = 2 * e^(-2) ≈ 0.2707
        let log_prob_1 = PoissonNB::<Trained>::poisson_log_pmf(1.0, lambda);
        assert_abs_diff_eq!(log_prob_1, 2.0_f64.ln() - 2.0, epsilon = 1e-10);

        // P(2; 2) = 2^2 * e^(-2) / 2! = 2 * e^(-2) ≈ 0.2707
        let log_prob_2 = PoissonNB::<Trained>::poisson_log_pmf(2.0, lambda);
        assert_abs_diff_eq!(
            log_prob_2,
            2.0 * 2.0_f64.ln() - 2.0 - 2.0_f64.ln(),
            epsilon = 1e-10
        );
    }

    #[test]
    fn test_poisson_nb_basic() {
        // Count data (e.g., number of events per time period)
        let x = array![
            [3.0, 1.0, 0.0],
            [2.0, 2.0, 0.0],
            [4.0, 0.0, 1.0],
            [1.0, 0.0, 2.0],
            [0.0, 3.0, 4.0],
            [0.0, 4.0, 3.0]
        ];
        let y = array![0, 0, 0, 1, 1, 1];

        let model = PoissonNB::new().alpha(1e-10).fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_poisson_nb_predict_proba() {
        let x = array![[2.0, 1.0], [1.0, 2.0], [3.0, 0.0], [0.0, 3.0]];
        let y = array![0, 1, 0, 1];

        let model = PoissonNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // All probabilities should be positive
        for &prob in proba.iter() {
            assert!(prob > 0.0);
            assert!(prob <= 1.0);
        }
    }

    #[test]
    fn test_poisson_nb_with_smoothing() {
        // Test with different alpha values
        let x = array![[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]];
        let y = array![0, 1, 0, 1];

        let model = PoissonNB::new()
            .alpha(0.1) // Different smoothing
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);
    }

    #[test]
    fn test_poisson_nb_with_custom_priors() {
        let x = array![[2.0, 1.0], [1.0, 2.0]];
        let y = array![0, 1];

        // Set custom priors
        let priors = array![0.3, 0.7];
        let model = PoissonNB::new().class_prior(priors).fit(&x, &y).unwrap();

        // The model should work with custom priors
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_poisson_nb_negative_features() {
        let x = array![
            [1.0, -1.0], // Negative count
            [2.0, 3.0]
        ];
        let y = array![0, 1];

        let result = PoissonNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_nb_non_integer_features() {
        let x = array![
            [1.5, 2.0], // Non-integer count
            [2.0, 3.0]
        ];
        let y = array![0, 1];

        let result = PoissonNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_poisson_nb_zero_counts() {
        // Test with data containing many zero counts
        let x = array![
            [0.0, 5.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 0.0, 4.0],
            [0.0, 0.0, 2.0]
        ];
        let y = array![0, 0, 1, 1];

        let model = PoissonNB::new().alpha(1e-5).fit(&x, &y).unwrap();

        // Should handle zero counts gracefully
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    #[should_panic(expected = "alpha must be >= 0")]
    fn test_poisson_nb_negative_alpha() {
        PoissonNB::new().alpha(-1.0);
    }
}
