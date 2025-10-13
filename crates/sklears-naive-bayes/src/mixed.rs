//! Mixed Naive Bayes classifier implementation for heterogeneous features

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

/// Feature distribution type for Mixed Naive Bayes
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FeatureDistribution {
    /// Gaussian distribution for continuous features
    Gaussian,
    /// Multinomial distribution for count features
    Multinomial,
    /// Bernoulli distribution for binary features
    Bernoulli,
    /// Categorical distribution for categorical features
    Categorical,
    /// Poisson distribution for count data
    Poisson,
    /// Gamma distribution for positive continuous data
    Gamma,
    /// Beta distribution for proportion data [0,1]
    Beta,
}

/// Configuration for Mixed Naive Bayes
#[derive(Debug, Clone)]
pub struct MixedNBConfig {
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
    /// Distribution type for each feature
    pub feature_distributions: Vec<FeatureDistribution>,
    /// Variance smoothing for Gaussian features
    pub var_smoothing: f64,
    /// Alpha smoothing for discrete features
    pub alpha: f64,
}

impl Default for MixedNBConfig {
    fn default() -> Self {
        Self {
            priors: None,
            feature_distributions: Vec::new(),
            var_smoothing: 1e-9,
            alpha: 1.0,
        }
    }
}

/// Mixed Naive Bayes classifier
///
/// Allows different features to follow different probability distributions.
/// This is particularly useful for datasets with heterogeneous feature types.
#[derive(Debug, Clone)]
pub struct MixedNB<State = Untrained> {
    config: MixedNBConfig,
    state: PhantomData<State>,
    // Trained state fields - parameters for each distribution type
    gaussian_params: Option<Array2<(f64, f64)>>, // (mean, variance) for Gaussian features
    discrete_params: Option<Array2<f64>>,        // Probabilities for discrete features
    gamma_params: Option<Array2<(f64, f64)>>,    // (shape, scale) for Gamma features
    beta_params: Option<Array2<(f64, f64)>>,     // (alpha, beta) for Beta features
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl MixedNB<Untrained> {
    /// Create a new Mixed Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: MixedNBConfig::default(),
            state: PhantomData,
            gaussian_params: None,
            discrete_params: None,
            gamma_params: None,
            beta_params: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set feature distributions
    pub fn feature_distributions(mut self, distributions: Vec<FeatureDistribution>) -> Self {
        self.config.feature_distributions = distributions;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }

    /// Set variance smoothing for Gaussian features
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.config.var_smoothing = var_smoothing;
        self
    }

    /// Set alpha smoothing for discrete features
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }
}

impl Default for MixedNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MixedNB<Untrained> {
    type Float = Float;
    type Config = MixedNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<i32>> for MixedNB<Untrained> {
    type Fitted = MixedNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_features = x.ncols();

        // If no distributions specified, infer them from data
        let feature_distributions = if self.config.feature_distributions.is_empty() {
            infer_feature_distributions(x)?
        } else {
            if self.config.feature_distributions.len() != n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of feature distributions ({}) doesn't match number of features ({})",
                    self.config.feature_distributions.len(),
                    n_features
                )));
            }
            self.config.feature_distributions.clone()
        };

        // Validate data against specified distributions
        validate_data_against_distributions(x, &feature_distributions)?;

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();

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

        // Initialize parameter arrays
        let mut gaussian_params = Array2::from_elem((n_classes, n_features), (0.0, 1.0));
        let mut discrete_params = Array2::from_elem((n_classes, n_features), 0.5);
        let mut gamma_params = Array2::from_elem((n_classes, n_features), (1.0, 1.0));
        let mut beta_params = Array2::from_elem((n_classes, n_features), (1.0, 1.0));

        // Estimate parameters for each class and feature
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

            // Estimate parameters for each feature based on its distribution
            for (feature_idx, &distribution) in feature_distributions.iter().enumerate() {
                let feature_values = x_class.column(feature_idx);

                if feature_values.is_empty() {
                    continue;
                }

                match distribution {
                    FeatureDistribution::Gaussian => {
                        let mean = feature_values.mean().unwrap_or(0.0);
                        let variance = if feature_values.len() > 1 {
                            feature_values.mapv(|v| (v - mean).powi(2)).sum()
                                / (feature_values.len() as f64)
                                + self.config.var_smoothing
                        } else {
                            self.config.var_smoothing
                        };
                        gaussian_params[[class_idx, feature_idx]] = (mean, variance);
                    }

                    FeatureDistribution::Bernoulli => {
                        let sum: f64 = feature_values.sum();
                        let count = feature_values.len() as f64;
                        let prob = (sum + self.config.alpha) / (count + 2.0 * self.config.alpha);
                        discrete_params[[class_idx, feature_idx]] = prob;
                    }

                    FeatureDistribution::Multinomial | FeatureDistribution::Categorical => {
                        // For these, we'd typically need count data
                        // For simplicity, treat as the mean frequency
                        let mean = feature_values.mean().unwrap_or(0.5);
                        discrete_params[[class_idx, feature_idx]] = mean;
                    }

                    FeatureDistribution::Poisson => {
                        let lambda = feature_values.mean().unwrap_or(1.0);
                        discrete_params[[class_idx, feature_idx]] = lambda;
                    }

                    FeatureDistribution::Gamma => {
                        let (alpha, beta) = estimate_gamma_parameters(&feature_values.to_vec());
                        gamma_params[[class_idx, feature_idx]] = (alpha, beta);
                    }

                    FeatureDistribution::Beta => {
                        let (alpha, beta) = estimate_beta_parameters(&feature_values.to_vec());
                        beta_params[[class_idx, feature_idx]] = (alpha, beta);
                    }
                }
            }
        }

        Ok(MixedNB {
            config: MixedNBConfig {
                feature_distributions,
                ..self.config
            },
            state: PhantomData,
            gaussian_params: Some(gaussian_params),
            discrete_params: Some(discrete_params),
            gamma_params: Some(gamma_params),
            beta_params: Some(beta_params),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl MixedNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();
        let n_features = x.ncols();
        let class_prior = self.class_prior_.as_ref().unwrap();

        if self.config.feature_distributions.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Feature distributions don't match data dimensions".to_string(),
            ));
        }

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            // Compute log likelihood for each sample
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = 0.0;

                // Sum log probabilities across features
                for (feature_idx, (&x_val, &distribution)) in x_sample
                    .iter()
                    .zip(self.config.feature_distributions.iter())
                    .enumerate()
                {
                    let feature_log_prob = match distribution {
                        FeatureDistribution::Gaussian => {
                            let (mean, variance) =
                                self.gaussian_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            let diff = x_val - mean;
                            -0.5 * (2.0 * std::f64::consts::PI * variance).ln()
                                - 0.5 * diff * diff / variance
                        }

                        FeatureDistribution::Bernoulli => {
                            let prob =
                                self.discrete_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            if x_val == 1.0 {
                                safe_log(prob)
                            } else if x_val == 0.0 {
                                safe_log(1.0 - prob)
                            } else {
                                return Err(SklearsError::InvalidInput(
                                    "Bernoulli features must be 0 or 1".to_string(),
                                ));
                            }
                        }

                        FeatureDistribution::Poisson => {
                            let lambda =
                                self.discrete_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            // Poisson log PMF: x*log(λ) - λ - log(x!)
                            x_val * lambda.ln() - lambda - factorial_ln(x_val as u32)
                        }

                        FeatureDistribution::Gamma => {
                            let (alpha, beta) =
                                self.gamma_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            if x_val <= 0.0 {
                                return Err(SklearsError::InvalidInput(
                                    "Gamma distribution requires positive values".to_string(),
                                ));
                            }
                            (alpha - 1.0) * x_val.ln()
                                - x_val / beta
                                - alpha * beta.ln()
                                - gamma_ln(alpha)
                        }

                        FeatureDistribution::Beta => {
                            let (alpha, beta) =
                                self.beta_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            if !(0.0..=1.0).contains(&x_val) {
                                return Err(SklearsError::InvalidInput(
                                    "Beta distribution requires values in [0,1]".to_string(),
                                ));
                            }
                            let safe_x = x_val.max(1e-15).min(1.0 - 1e-15);
                            (alpha - 1.0) * safe_x.ln() + (beta - 1.0) * (1.0 - safe_x).ln()
                                - beta_function_ln(alpha, beta)
                        }

                        FeatureDistribution::Multinomial | FeatureDistribution::Categorical => {
                            // Simplified - treat as frequency
                            let prob =
                                self.discrete_params.as_ref().unwrap()[[class_idx, feature_idx]];
                            x_val * safe_log(prob)
                        }
                    };

                    log_prob += feature_log_prob;
                }

                joint_log_likelihood[[sample_idx, class_idx]] =
                    log_prob + safe_log(class_prior[class_idx]);
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for MixedNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for MixedNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for MixedNB<Trained> {
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

impl NaiveBayesMixin for MixedNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // Return discrete parameters as a proxy
        self.discrete_params.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

/// Helper functions

/// Infer feature distributions from data characteristics
fn infer_feature_distributions(x: &Array2<Float>) -> Result<Vec<FeatureDistribution>> {
    let mut distributions = Vec::new();

    for feature_idx in 0..x.ncols() {
        let column = x.column(feature_idx);
        let unique_values: std::collections::HashSet<_> =
            column.iter().map(|&v| (v * 1000.0) as i32).collect(); // Rough uniqueness check

        let distribution =
            if unique_values.len() == 2 && column.iter().all(|&v| v == 0.0 || v == 1.0) {
                FeatureDistribution::Bernoulli
            } else if column.iter().all(|&v| (0.0..=1.0).contains(&v)) {
                FeatureDistribution::Beta
            } else if column.iter().all(|&v| v > 0.0) {
                FeatureDistribution::Gamma
            } else if column.iter().all(|&v| v >= 0.0 && v.fract() == 0.0) {
                FeatureDistribution::Poisson
            } else {
                FeatureDistribution::Gaussian
            };

        distributions.push(distribution);
    }

    Ok(distributions)
}

/// Validate that data is compatible with specified distributions
fn validate_data_against_distributions(
    x: &Array2<Float>,
    distributions: &[FeatureDistribution],
) -> Result<()> {
    for (feature_idx, &distribution) in distributions.iter().enumerate() {
        let column = x.column(feature_idx);

        match distribution {
            FeatureDistribution::Bernoulli => {
                if !column.iter().all(|&v| v == 0.0 || v == 1.0) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} marked as Bernoulli but contains non-binary values",
                        feature_idx
                    )));
                }
            }
            FeatureDistribution::Beta => {
                if !column.iter().all(|&v| (0.0..=1.0).contains(&v)) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} marked as Beta but contains values outside [0,1]",
                        feature_idx
                    )));
                }
            }
            FeatureDistribution::Gamma => {
                if !column.iter().all(|&v| v > 0.0) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} marked as Gamma but contains non-positive values",
                        feature_idx
                    )));
                }
            }
            FeatureDistribution::Poisson => {
                if !column.iter().all(|&v| v >= 0.0 && v.fract() == 0.0) {
                    return Err(SklearsError::InvalidInput(format!(
                        "Feature {} marked as Poisson but contains non-integer or negative values",
                        feature_idx
                    )));
                }
            }
            _ => {} // Other distributions are more flexible
        }
    }

    Ok(())
}

/// Simple parameter estimation functions (reused from other modules)
fn estimate_gamma_parameters(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (1.0, 1.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    if variance <= 0.0 {
        (1.0, mean)
    } else {
        let alpha = mean * mean / variance;
        let beta = variance / mean;
        (alpha.max(1e-10), beta.max(1e-10))
    }
}

fn estimate_beta_parameters(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (1.0, 1.0);
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    let safe_mean = mean.max(1e-10).min(1.0 - 1e-10);
    let safe_variance = variance
        .max(1e-10)
        .min(safe_mean * (1.0 - safe_mean) * 0.99);

    let temp = safe_mean * (1.0 - safe_mean) / safe_variance - 1.0;
    let alpha = safe_mean * temp;
    let beta = (1.0 - safe_mean) * temp;

    (alpha.max(1e-10), beta.max(1e-10))
}

// Mathematical helper functions
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if x < 1.0 {
        gamma_ln(x + 1.0) - x.ln()
    } else {
        (x - 0.5) * x.ln() - x + 0.5 * (2.0 * std::f64::consts::PI).ln()
    }
}

fn beta_function_ln(alpha: f64, beta: f64) -> f64 {
    gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta)
}

fn factorial_ln(n: u32) -> f64 {
    if n <= 1 {
        0.0
    } else {
        (2..=n).map(|i| (i as f64).ln()).sum()
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
    fn test_mixed_nb_auto_inference() {
        // Mixed data: binary, proportion, positive continuous, general continuous
        let x = array![
            [0.0, 0.2, 1.5, -1.0],
            [1.0, 0.8, 2.5, 2.0],
            [0.0, 0.1, 0.5, -0.5],
            [1.0, 0.9, 3.0, 1.5],
        ];
        let y = array![0, 1, 0, 1];

        let model = MixedNB::new().fit(&x, &y).unwrap();

        // Check that distributions were inferred
        assert_eq!(
            model.config.feature_distributions[0],
            FeatureDistribution::Bernoulli
        );
        assert_eq!(
            model.config.feature_distributions[1],
            FeatureDistribution::Beta
        );
        assert_eq!(
            model.config.feature_distributions[2],
            FeatureDistribution::Gamma
        );
        assert_eq!(
            model.config.feature_distributions[3],
            FeatureDistribution::Gaussian
        );

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_mixed_nb_explicit_distributions() {
        let x = array![
            [0.0, 0.2, 1.5, -1.0],
            [1.0, 0.8, 2.5, 2.0],
            [0.0, 0.1, 0.5, -0.5],
            [1.0, 0.9, 3.0, 1.5],
        ];
        let y = array![0, 1, 0, 1];

        let distributions = vec![
            FeatureDistribution::Bernoulli,
            FeatureDistribution::Beta,
            FeatureDistribution::Gamma,
            FeatureDistribution::Gaussian,
        ];

        let model = MixedNB::new()
            .feature_distributions(distributions)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_mixed_nb_predict_proba() {
        let x = array![[0.0, 0.2], [1.0, 0.8], [0.0, 0.1], [1.0, 0.9],];
        let y = array![0, 1, 0, 1];

        let model = MixedNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_validation_error() {
        let x = array![[0.0, 1.5], [1.0, 0.8]]; // Second feature > 1, incompatible with Beta
        let y = array![0, 1];

        let distributions = vec![
            FeatureDistribution::Bernoulli,
            FeatureDistribution::Beta, // This should cause validation error
        ];

        let result = MixedNB::new()
            .feature_distributions(distributions)
            .fit(&x, &y);

        assert!(result.is_err());
    }

    #[test]
    fn test_infer_feature_distributions() {
        let x = array![
            [0.0, 0.2, 1.5],
            [1.0, 0.8, 2.5],
            [0.0, 0.1, 0.5],
            [1.0, 0.9, 3.0],
        ];

        let distributions = infer_feature_distributions(&x).unwrap();

        assert_eq!(distributions[0], FeatureDistribution::Bernoulli);
        assert_eq!(distributions[1], FeatureDistribution::Beta);
        assert_eq!(distributions[2], FeatureDistribution::Gamma);
    }
}
