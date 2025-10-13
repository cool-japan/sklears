//! Gamma Naive Bayes classifier implementation

// SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result},
    prelude::{Predict, SklearsError},
    traits::{Estimator, Fit, PredictProba, Score, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

use crate::{compute_class_prior, safe_log, NaiveBayesMixin};

/// Configuration for Gamma Naive Bayes
#[derive(Debug, Clone)]
pub struct GammaNBConfig {
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
    /// Whether to estimate parameters using method of moments (true) or MLE (false)
    pub method_of_moments: bool,
}

impl Default for GammaNBConfig {
    fn default() -> Self {
        Self {
            priors: None,
            method_of_moments: true,
        }
    }
}

/// Gamma Naive Bayes classifier
///
/// For each class, the likelihood of the features is assumed to follow a Gamma distribution.
/// This is particularly suitable for positive continuous data like waiting times, prices, etc.
#[derive(Debug, Clone)]
pub struct GammaNB<State = Untrained> {
    config: GammaNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    shape_: Option<Array2<f64>>, // Shape parameter (alpha) of each feature per class
    scale_: Option<Array2<f64>>, // Scale parameter (beta) of each feature per class
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl GammaNB<Untrained> {
    /// Create a new Gamma Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: GammaNBConfig::default(),
            state: PhantomData,
            shape_: None,
            scale_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }

    /// Set parameter estimation method
    pub fn method_of_moments(mut self, use_mom: bool) -> Self {
        self.config.method_of_moments = use_mom;
        self
    }
}

impl Default for GammaNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GammaNB<Untrained> {
    type Float = Float;
    type Config = GammaNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for GammaNB<Untrained> {
    type Fitted = GammaNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check that all values are positive (required for Gamma distribution)
        if x.iter().any(|&val| val <= 0.0) {
            return Err(SklearsError::InvalidInput(
                "Gamma Naive Bayes requires all feature values to be positive".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Initialize parameters
        let mut shape = Array2::zeros((n_classes, n_features));
        let mut scale = Array2::zeros((n_classes, n_features));

        // Compute class counts and priors
        let (_class_count, class_prior) = if let Some(ref priors) = self.config.priors {
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
            let class_count = Array1::zeros(n_classes);
            (class_count, priors.clone())
        } else {
            compute_class_prior(y, &classes)
        };

        // Estimate Gamma parameters for each class and feature
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

            // Select rows belonging to this class
            let x_class = x.select(Axis(0), &mask);

            // Estimate parameters for each feature
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);

                if feature_values.is_empty() {
                    continue;
                }

                let (alpha, beta) = if self.config.method_of_moments {
                    // Method of moments estimation
                    let mean = feature_values.mean().unwrap_or(1.0);
                    let variance = if feature_values.len() > 1 {
                        feature_values.mapv(|v| (v - mean).powi(2)).sum()
                            / (feature_values.len() as f64)
                    } else {
                        mean // fallback for single sample
                    };

                    if variance <= 0.0 {
                        (1.0, mean) // fallback parameters
                    } else {
                        let alpha = mean * mean / variance;
                        let beta = variance / mean;
                        (alpha.max(1e-10), beta.max(1e-10))
                    }
                } else {
                    // Maximum Likelihood Estimation for Gamma distribution
                    gamma_mle_estimate(&feature_values.to_vec())
                };

                shape[[class_idx, feature_idx]] = alpha;
                scale[[class_idx, feature_idx]] = beta;
            }
        }

        Ok(GammaNB {
            config: self.config,
            state: PhantomData,
            shape_: Some(shape),
            scale_: Some(scale),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl GammaNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let shape = self.shape_.as_ref().unwrap();
        let scale = self.scale_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let class_shape = shape.row(class_idx);
            let class_scale = scale.row(class_idx);

            // Compute log likelihood for each sample
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = 0.0;

                // Gamma log likelihood
                for (feature_idx, &x_val) in x_sample.iter().enumerate() {
                    if x_val <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Gamma distribution requires positive values".to_string(),
                        ));
                    }

                    let alpha = class_shape[feature_idx];
                    let beta = class_scale[feature_idx];

                    // Gamma PDF: (x^(α-1) * exp(-x/β)) / (β^α * Γ(α))
                    // Log PDF: (α-1)*log(x) - x/β - α*log(β) - log(Γ(α))
                    log_prob += (alpha - 1.0) * x_val.ln()
                        - x_val / beta
                        - alpha * beta.ln()
                        - gamma_ln(alpha);
                }

                joint_log_likelihood[[sample_idx, class_idx]] =
                    log_prob + safe_log(class_prior[class_idx]);
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for GammaNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for GammaNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for GammaNB<Trained> {
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

impl NaiveBayesMixin for GammaNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For Gamma NB, return shape parameters as a proxy
        self.shape_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

/// Estimate Gamma distribution parameters using Maximum Likelihood Estimation
fn gamma_mle_estimate(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (1.0, 1.0);
    }

    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let log_mean = values.iter().map(|&x| x.ln()).sum::<f64>() / n;

    // Initial estimate using method of moments
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n;
    let mut alpha = if variance > 0.0 {
        mean * mean / variance
    } else {
        1.0
    };

    // Newton-Raphson iteration to refine alpha estimate
    for _ in 0..10 {
        let digamma_alpha = digamma(alpha);
        let trigamma_alpha = trigamma(alpha);

        let f = alpha.ln() - digamma_alpha - (mean.ln() - log_mean);
        let df = 1.0 / alpha - trigamma_alpha;

        let delta = f / df;
        alpha -= delta;

        if delta.abs() < 1e-8 {
            break;
        }

        // Ensure alpha stays positive
        alpha = alpha.max(1e-10);
    }

    let beta = mean / alpha;
    (alpha.max(1e-10), beta.max(1e-10))
}

/// Logarithm of the gamma function (approximation using Stirling's formula)
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    // Use built-in gamma function if available, otherwise approximate
    // This is a simplified version - in practice you'd want a more accurate implementation
    if x < 1.0 {
        // Γ(x) = Γ(x+1) / x
        gamma_ln(x + 1.0) - x.ln()
    } else {
        // Stirling's approximation: ln(Γ(x)) ≈ (x-0.5)*ln(x) - x + 0.5*ln(2π)
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

/// Digamma function (derivative of log gamma function) - approximation
fn digamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 1.0 {
        digamma(x + 1.0) - 1.0 / x
    } else {
        // Asymptotic expansion for large x
        x.ln() - 1.0 / (2.0 * x) - 1.0 / (12.0 * x * x) + 1.0 / (120.0 * x.powi(4))
    }
}

/// Trigamma function (derivative of digamma function) - approximation
fn trigamma(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::INFINITY;
    }

    if x < 1.0 {
        trigamma(x + 1.0) + 1.0 / (x * x)
    } else {
        // Asymptotic expansion for large x
        1.0 / x + 1.0 / (2.0 * x * x) + 1.0 / (6.0 * x * x * x) - 1.0 / (30.0 * x.powi(5))
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
    fn test_gamma_nb_basic() {
        // Simple 2D data with positive values
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [0.5, 1.0],
            [1.5, 2.0],
            [2.5, 3.0],
            [3.5, 4.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = GammaNB::new().fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.5); // Should perform reasonably well
    }

    #[test]
    fn test_gamma_nb_predict_proba() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [1.5, 1.5]];
        let y = array![0, 0, 1, 1];

        let model = GammaNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }

        // Check that all probabilities are positive
        for prob in proba.iter() {
            assert!(*prob >= 0.0);
            assert!(*prob <= 1.0);
        }
    }

    #[test]
    fn test_gamma_nb_with_priors() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [0.5, 0.5], [1.5, 1.5]];
        let y = array![0, 0, 1, 1];

        // Set custom priors
        let priors = array![0.3, 0.7];
        let model = GammaNB::new().priors(priors).fit(&x, &y).unwrap();

        // The model should still work with custom priors
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_gamma_nb_negative_values_error() {
        // Test that negative values cause an error
        let x = array![[1.0, -1.0], [2.0, 2.0]];
        let y = array![0, 1];

        let result = GammaNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_gamma_mle_estimate() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let (alpha, beta) = gamma_mle_estimate(&values);

        assert!(alpha > 0.0);
        assert!(beta > 0.0);

        // The mean should be approximately alpha * beta
        let estimated_mean = alpha * beta;
        let actual_mean = 3.0; // mean of [1,2,3,4,5]
        assert_abs_diff_eq!(estimated_mean, actual_mean, epsilon = 0.5);
    }

    #[test]
    fn test_gamma_ln() {
        // Test some known values
        assert_abs_diff_eq!(gamma_ln(1.0), 0.0, epsilon = 0.1); // Γ(1) = 1, ln(1) = 0 (approx)
        assert_abs_diff_eq!(gamma_ln(2.0), 0.0, epsilon = 0.1); // Γ(2) = 1, ln(1) = 0 (approx)

        // Γ(3) = 2, ln(2) ≈ 0.693
        assert_abs_diff_eq!(gamma_ln(3.0), 2_f64.ln(), epsilon = 0.03);
    }
}
