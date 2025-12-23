//! Beta Naive Bayes classifier implementation

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

/// Configuration for Beta Naive Bayes
#[derive(Debug, Clone)]
pub struct BetaNBConfig {
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
    /// Whether to estimate parameters using method of moments (true) or MLE (false)
    pub method_of_moments: bool,
}

impl Default for BetaNBConfig {
    fn default() -> Self {
        Self {
            priors: None,
            method_of_moments: true,
        }
    }
}

/// Beta Naive Bayes classifier
///
/// For each class, the likelihood of the features is assumed to follow a Beta distribution.
/// This is particularly suitable for proportion data, probabilities, and values in [0,1].
#[derive(Debug, Clone)]
pub struct BetaNB<State = Untrained> {
    config: BetaNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    alpha_: Option<Array2<f64>>, // Alpha parameter of each feature per class
    beta_: Option<Array2<f64>>,  // Beta parameter of each feature per class
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl BetaNB<Untrained> {
    /// Create a new Beta Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: BetaNBConfig::default(),
            state: PhantomData,
            alpha_: None,
            beta_: None,
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

impl Default for BetaNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for BetaNB<Untrained> {
    type Float = Float;
    type Config = BetaNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for BetaNB<Untrained> {
    type Fitted = BetaNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Check that all values are in [0,1] (required for Beta distribution)
        if x.iter().any(|&val| !(0.0..=1.0).contains(&val)) {
            return Err(SklearsError::InvalidInput(
                "Beta Naive Bayes requires all feature values to be in the range [0,1]".to_string(),
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
        let mut alpha = Array2::zeros((n_classes, n_features));
        let mut beta = Array2::zeros((n_classes, n_features));

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

        // Estimate Beta parameters for each class and feature
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

                let (alpha_param, beta_param) = if self.config.method_of_moments {
                    // Method of moments estimation
                    let mean = feature_values.mean().unwrap_or(0.5);
                    let variance = if feature_values.len() > 1 {
                        feature_values.mapv(|v| (v - mean).powi(2)).sum()
                            / (feature_values.len() as f64)
                    } else {
                        0.01 // Small variance for single sample
                    };

                    // Ensure mean is strictly between 0 and 1
                    let safe_mean = mean.clamp(1e-10, 1.0 - 1e-10);
                    let safe_variance = variance
                        .max(1e-10)
                        .min(safe_mean * (1.0 - safe_mean) * 0.99);

                    // Method of moments: α = μ((μ(1-μ)/σ²) - 1), β = (1-μ)((μ(1-μ)/σ²) - 1)
                    let temp = safe_mean * (1.0 - safe_mean) / safe_variance - 1.0;
                    let alpha_est = safe_mean * temp;
                    let beta_est = (1.0 - safe_mean) * temp;

                    (alpha_est.max(1e-10), beta_est.max(1e-10))
                } else {
                    // Maximum Likelihood Estimation for Beta distribution
                    beta_mle_estimate(&feature_values.to_vec())
                };

                alpha[[class_idx, feature_idx]] = alpha_param;
                beta[[class_idx, feature_idx]] = beta_param;
            }
        }

        Ok(BetaNB {
            config: self.config,
            state: PhantomData,
            alpha_: Some(alpha),
            beta_: Some(beta),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl BetaNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let alpha = self.alpha_.as_ref().unwrap();
        let beta = self.beta_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            let class_alpha = alpha.row(class_idx);
            let class_beta = beta.row(class_idx);

            // Compute log likelihood for each sample
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = 0.0;

                // Beta log likelihood
                for (feature_idx, &x_val) in x_sample.iter().enumerate() {
                    if !(0.0..=1.0).contains(&x_val) {
                        return Err(SklearsError::InvalidInput(
                            "Beta distribution requires values in [0,1]".to_string(),
                        ));
                    }

                    let alpha_param = class_alpha[feature_idx];
                    let beta_param = class_beta[feature_idx];

                    // Handle boundary cases
                    let safe_x = if x_val == 0.0 {
                        1e-15
                    } else if x_val == 1.0 {
                        1.0 - 1e-15
                    } else {
                        x_val
                    };

                    // Beta PDF: x^(α-1) * (1-x)^(β-1) / B(α,β)
                    // Log PDF: (α-1)*log(x) + (β-1)*log(1-x) - log(B(α,β))
                    // where B(α,β) = Γ(α)Γ(β)/Γ(α+β)
                    log_prob += (alpha_param - 1.0) * safe_x.ln()
                        + (beta_param - 1.0) * (1.0 - safe_x).ln()
                        - beta_function_ln(alpha_param, beta_param);
                }

                joint_log_likelihood[[sample_idx, class_idx]] =
                    log_prob + safe_log(class_prior[class_idx]);
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for BetaNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for BetaNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for BetaNB<Trained> {
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

impl NaiveBayesMixin for BetaNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For Beta NB, return alpha parameters as a proxy
        self.alpha_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

/// Estimate Beta distribution parameters using Maximum Likelihood Estimation
fn beta_mle_estimate(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (1.0, 1.0);
    }

    // Start with method of moments estimate
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    // Ensure mean is strictly between 0 and 1
    let safe_mean = mean.clamp(1e-10, 1.0 - 1e-10);
    let safe_variance = variance
        .max(1e-10)
        .min(safe_mean * (1.0 - safe_mean) * 0.99);

    let temp = safe_mean * (1.0 - safe_mean) / safe_variance - 1.0;
    let mut alpha = safe_mean * temp;
    let mut beta = (1.0 - safe_mean) * temp;

    // Ensure positive parameters
    alpha = alpha.max(1e-10);
    beta = beta.max(1e-10);

    // Newton-Raphson refinement (simplified version)
    let log_sum = values.iter().map(|&x| x.max(1e-15).ln()).sum::<f64>();
    let log_sum_1_minus = values
        .iter()
        .map(|&x| (1.0 - x).max(1e-15).ln())
        .sum::<f64>();
    let n = values.len() as f64;

    for _ in 0..5 {
        let digamma_alpha = digamma(alpha);
        let digamma_beta = digamma(beta);
        let digamma_sum = digamma(alpha + beta);

        let g1 = log_sum / n - digamma_alpha + digamma_sum;
        let g2 = log_sum_1_minus / n - digamma_beta + digamma_sum;

        let trigamma_alpha = trigamma(alpha);
        let trigamma_beta = trigamma(beta);
        let trigamma_sum = trigamma(alpha + beta);

        let h11 = -trigamma_alpha + trigamma_sum;
        let h12 = trigamma_sum;
        let h22 = -trigamma_beta + trigamma_sum;

        let det = h11 * h22 - h12 * h12;
        if det.abs() < 1e-15 {
            break;
        }

        let delta_alpha = (h22 * g1 - h12 * g2) / det;
        let delta_beta = (h11 * g2 - h12 * g1) / det;

        alpha -= delta_alpha;
        beta -= delta_beta;

        // Ensure parameters stay positive
        alpha = alpha.max(1e-10);
        beta = beta.max(1e-10);

        if delta_alpha.abs() < 1e-8 && delta_beta.abs() < 1e-8 {
            break;
        }
    }

    (alpha, beta)
}

/// Log of the Beta function: ln(B(α,β)) = ln(Γ(α)) + ln(Γ(β)) - ln(Γ(α+β))
fn beta_function_ln(alpha: f64, beta: f64) -> f64 {
    gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta)
}

/// Logarithm of the gamma function (reuse from gamma.rs, simplified)
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }

    if x < 1.0 {
        gamma_ln(x + 1.0) - x.ln()
    } else {
        // Stirling's approximation
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
    fn test_beta_nb_basic() {
        // Simple 2D data with proportion values
        let x = array![
            [0.8, 0.9],
            [0.7, 0.8],
            [0.6, 0.7],
            [0.9, 0.8],
            [0.2, 0.1],
            [0.3, 0.2],
            [0.4, 0.3],
            [0.1, 0.2]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = BetaNB::new().fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.5); // Should perform reasonably well
    }

    #[test]
    fn test_beta_nb_predict_proba() {
        let x = array![[0.8, 0.9], [0.7, 0.8], [0.2, 0.1], [0.3, 0.2]];
        let y = array![0, 0, 1, 1];

        let model = BetaNB::new().fit(&x, &y).unwrap();
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
    fn test_beta_nb_with_priors() {
        let x = array![[0.8, 0.9], [0.7, 0.8], [0.2, 0.1], [0.3, 0.2]];
        let y = array![0, 0, 1, 1];

        // Set custom priors
        let priors = array![0.3, 0.7];
        let model = BetaNB::new().priors(priors).fit(&x, &y).unwrap();

        // The model should still work with custom priors
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_beta_nb_out_of_range_error() {
        // Test that values outside [0,1] cause an error
        let x = array![[0.5, 1.5], [0.8, 0.9]];
        let y = array![0, 1];

        let result = BetaNB::new().fit(&x, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_beta_mle_estimate() {
        let values = vec![0.1, 0.2, 0.3, 0.4, 0.5];
        let (alpha, beta) = beta_mle_estimate(&values);

        assert!(alpha > 0.0);
        assert!(beta > 0.0);

        // The mean should be approximately alpha / (alpha + beta)
        let estimated_mean = alpha / (alpha + beta);
        let actual_mean = 0.3; // mean of [0.1,0.2,0.3,0.4,0.5]
        assert_abs_diff_eq!(estimated_mean, actual_mean, epsilon = 0.1);
    }

    #[test]
    fn test_beta_function_ln() {
        // Test B(1,1) = 1, so ln(B(1,1)) = 0
        assert_abs_diff_eq!(beta_function_ln(1.0, 1.0), 0.0, epsilon = 0.15);

        // Test B(2,2) = Γ(2)Γ(2)/Γ(4) = 1*1/6 = 1/6, so ln(B(2,2)) = ln(1/6)
        let expected = -6_f64.ln();
        assert_abs_diff_eq!(beta_function_ln(2.0, 2.0), expected, epsilon = 0.1);
    }

    #[test]
    fn test_boundary_values() {
        // Test with boundary values (0 and 1)
        let x = array![[0.0, 1.0], [0.1, 0.9], [1.0, 0.0], [0.9, 0.1]];
        let y = array![0, 0, 1, 1];

        let model = BetaNB::new().fit(&x, &y).unwrap();
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }
}
