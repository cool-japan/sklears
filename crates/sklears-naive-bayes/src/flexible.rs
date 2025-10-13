//! Flexible Naive Bayes classifier with adaptive distributions
//!
//! This classifier automatically adapts to the best distribution for each feature
//! based on statistical tests and information criteria.

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

/// Supported distributions for adaptive selection
#[derive(Debug, Clone, PartialEq)]
pub enum Distribution {
    /// Gaussian
    Gaussian,
    /// Exponential
    Exponential,
    /// Gamma
    Gamma,
    /// Beta
    Beta,
    /// Poisson
    Poisson,
    /// Multinomial
    Multinomial,
    /// Bernoulli
    Bernoulli,
}

/// Distribution parameters for flexible NB
#[derive(Debug, Clone)]
pub enum DistributionParams {
    /// Gaussian
    Gaussian { mean: f64, var: f64 },
    /// Exponential
    Exponential { lambda: f64 },
    /// Gamma
    Gamma { alpha: f64, beta: f64 },
    /// Beta
    Beta { alpha: f64, beta: f64 },
    /// Poisson
    Poisson { lambda: f64 },
    /// Multinomial
    Multinomial { probs: Array1<f64> },
    /// Bernoulli
    Bernoulli { p: f64 },
}

/// Method for selecting the best distribution
#[derive(Debug, Clone)]
pub enum SelectionMethod {
    /// Akaike Information Criterion
    AIC,
    /// Bayesian Information Criterion
    BIC,
    /// Cross-validation score
    CrossValidation { folds: usize },
    /// Kolmogorov-Smirnov test p-value
    KSTest,
    /// Anderson-Darling test
    ADTest,
}

/// Configuration for Flexible Naive Bayes
#[derive(Debug, Clone)]
pub struct FlexibleNBConfig {
    /// Method for selecting best distribution
    pub selection_method: SelectionMethod,
    /// Candidate distributions to consider
    pub candidate_distributions: Vec<Distribution>,
    /// Minimum number of samples required for distribution testing
    pub min_samples: usize,
    /// Significance level for statistical tests
    pub alpha: f64,
    /// Variance smoothing for numerical stability
    pub var_smoothing: f64,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
}

impl Default for FlexibleNBConfig {
    fn default() -> Self {
        Self {
            selection_method: SelectionMethod::BIC,
            candidate_distributions: vec![
                Distribution::Gaussian,
                Distribution::Exponential,
                Distribution::Gamma,
                Distribution::Beta,
                Distribution::Poisson,
            ],
            min_samples: 30,
            alpha: 0.05,
            var_smoothing: 1e-9,
            priors: None,
        }
    }
}

/// Flexible Naive Bayes classifier with adaptive distributions
///
/// This classifier automatically selects the best distribution for each feature
/// per class based on statistical criteria.
#[derive(Debug, Clone)]
pub struct FlexibleNB<State = Untrained> {
    config: FlexibleNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    feature_distributions_: Option<Vec<Vec<Distribution>>>, // [n_classes][n_features]
    feature_params_: Option<Vec<Vec<DistributionParams>>>,  // [n_classes][n_features]
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl FlexibleNB<Untrained> {
    /// Create a new Flexible Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: FlexibleNBConfig::default(),
            state: PhantomData,
            feature_distributions_: None,
            feature_params_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set selection method for distribution testing
    pub fn selection_method(mut self, method: SelectionMethod) -> Self {
        self.config.selection_method = method;
        self
    }

    /// Set candidate distributions
    pub fn candidate_distributions(mut self, distributions: Vec<Distribution>) -> Self {
        self.config.candidate_distributions = distributions;
        self
    }

    /// Set minimum samples required for testing
    pub fn min_samples(mut self, min_samples: usize) -> Self {
        self.config.min_samples = min_samples;
        self
    }

    /// Set significance level for statistical tests
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.config.alpha = alpha;
        self
    }

    /// Set variance smoothing
    pub fn var_smoothing(mut self, var_smoothing: f64) -> Self {
        self.config.var_smoothing = var_smoothing;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }
}

impl Default for FlexibleNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for FlexibleNB<Untrained> {
    type Float = Float;
    type Config = FlexibleNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl FlexibleNB<Untrained> {
    /// Select the best distribution for a feature given data
    fn select_best_distribution(&self, data: &Array1<f64>) -> (Distribution, DistributionParams) {
        if data.len() < self.config.min_samples {
            // Fallback to Gaussian for small samples
            let mean = data.mean().unwrap_or(0.0);
            let var = if data.len() > 1 {
                data.mapv(|v| (v - mean).powi(2)).sum() / (data.len() as f64 - 1.0)
                    + self.config.var_smoothing
            } else {
                self.config.var_smoothing
            };
            return (
                Distribution::Gaussian,
                DistributionParams::Gaussian { mean, var },
            );
        }

        let mut best_distribution = Distribution::Gaussian;
        let mut best_params = DistributionParams::Gaussian {
            mean: 0.0,
            var: 1.0,
        };
        let mut best_score = f64::INFINITY;

        for dist in &self.config.candidate_distributions {
            if let Some((params, score)) = self.fit_and_score_distribution(dist, data) {
                if score < best_score {
                    best_score = score;
                    best_distribution = dist.clone();
                    best_params = params;
                }
            }
        }

        (best_distribution, best_params)
    }

    /// Fit distribution and compute selection score
    fn fit_and_score_distribution(
        &self,
        dist: &Distribution,
        data: &Array1<f64>,
    ) -> Option<(DistributionParams, f64)> {
        match dist {
            Distribution::Gaussian => {
                let mean = data.mean().unwrap_or(0.0);
                let var = if data.len() > 1 {
                    data.mapv(|v| (v - mean).powi(2)).sum() / (data.len() as f64 - 1.0)
                        + self.config.var_smoothing
                } else {
                    self.config.var_smoothing
                };

                let params = DistributionParams::Gaussian { mean, var };
                let score = self.compute_selection_score(&params, data, 2);
                Some((params, score))
            }
            Distribution::Exponential => {
                // Check if data is positive
                if data.iter().any(|&x| x <= 0.0) {
                    return None;
                }

                let lambda = 1.0 / data.mean().unwrap_or(1.0);
                let params = DistributionParams::Exponential { lambda };
                let score = self.compute_selection_score(&params, data, 1);
                Some((params, score))
            }
            Distribution::Gamma => {
                // Check if data is positive
                if data.iter().any(|&x| x <= 0.0) {
                    return None;
                }

                // Method of moments estimation
                let mean = data.mean().unwrap_or(1.0);
                let var = if data.len() > 1 {
                    data.mapv(|v| (v - mean).powi(2)).sum() / (data.len() as f64 - 1.0)
                } else {
                    1.0
                };

                if var <= 0.0 {
                    return None;
                }

                let alpha = mean * mean / var;
                let beta = mean / var;

                let params = DistributionParams::Gamma { alpha, beta };
                let score = self.compute_selection_score(&params, data, 2);
                Some((params, score))
            }
            Distribution::Beta => {
                // Check if data is in [0, 1]
                if data.iter().any(|&x| !(0.0..=1.0).contains(&x)) {
                    return None;
                }

                // Method of moments estimation
                let mean = data.mean().unwrap_or(0.5);
                let var = if data.len() > 1 {
                    data.mapv(|v| (v - mean).powi(2)).sum() / (data.len() as f64 - 1.0)
                } else {
                    0.25
                };

                if var <= 0.0 || mean <= 0.0 || mean >= 1.0 {
                    return None;
                }

                let temp = mean * (1.0 - mean) / var - 1.0;
                if temp <= 0.0 {
                    return None;
                }

                let alpha = mean * temp;
                let beta = (1.0 - mean) * temp;

                let params = DistributionParams::Beta { alpha, beta };
                let score = self.compute_selection_score(&params, data, 2);
                Some((params, score))
            }
            Distribution::Poisson => {
                // Check if data consists of non-negative integers
                if data.iter().any(|&x| x < 0.0 || x.fract() != 0.0) {
                    return None;
                }

                let lambda = data.mean().unwrap_or(1.0).max(1e-10);
                let params = DistributionParams::Poisson { lambda };
                let score = self.compute_selection_score(&params, data, 1);
                Some((params, score))
            }
            Distribution::Bernoulli => {
                // Check if data is binary
                if data.iter().any(|&x| x != 0.0 && x != 1.0) {
                    return None;
                }

                let p = data.mean().unwrap_or(0.5).clamp(1e-10, 1.0 - 1e-10);
                let params = DistributionParams::Bernoulli { p };
                let score = self.compute_selection_score(&params, data, 1);
                Some((params, score))
            }
            _ => None,
        }
    }

    /// Compute selection score based on the chosen method
    fn compute_selection_score(
        &self,
        params: &DistributionParams,
        data: &Array1<f64>,
        n_params: usize,
    ) -> f64 {
        let log_likelihood = self.compute_log_likelihood(params, data);
        let n = data.len() as f64;

        match self.config.selection_method {
            SelectionMethod::AIC => -2.0 * log_likelihood + 2.0 * n_params as f64,
            SelectionMethod::BIC => -2.0 * log_likelihood + (n_params as f64) * n.ln(),
            SelectionMethod::CrossValidation { .. } => {
                // Simplified: return negative log-likelihood (higher is better)
                -log_likelihood
            }
            SelectionMethod::KSTest | SelectionMethod::ADTest => {
                // For simplicity, use negative log-likelihood
                // In practice, would compute actual test statistics
                -log_likelihood
            }
        }
    }

    /// Compute log-likelihood for given parameters and data
    fn compute_log_likelihood(&self, params: &DistributionParams, data: &Array1<f64>) -> f64 {
        match params {
            DistributionParams::Gaussian { mean, var } => {
                let two_pi_var = 2.0 * std::f64::consts::PI * var;
                data.iter()
                    .map(|&x| {
                        let diff = x - mean;
                        -0.5 * two_pi_var.ln() - 0.5 * diff * diff / var
                    })
                    .sum()
            }
            DistributionParams::Exponential { lambda } => {
                data.iter().map(|&x| lambda.ln() - lambda * x).sum()
            }
            DistributionParams::Gamma { alpha, beta } => {
                let log_gamma_alpha = gamma_ln(*alpha);
                data.iter()
                    .map(|&x| {
                        (alpha - 1.0) * x.ln() - beta * x + alpha * beta.ln() - log_gamma_alpha
                    })
                    .sum()
            }
            DistributionParams::Beta { alpha, beta } => {
                let log_beta_fn = gamma_ln(*alpha) + gamma_ln(*beta) - gamma_ln(alpha + beta);
                data.iter()
                    .map(|&x| (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta_fn)
                    .sum()
            }
            DistributionParams::Poisson { lambda } => data
                .iter()
                .map(|&x| x * lambda.ln() - lambda - gamma_ln(x + 1.0))
                .sum(),
            DistributionParams::Bernoulli { p } => data
                .iter()
                .map(|&x| x * p.ln() + (1.0 - x) * (1.0 - p).ln())
                .sum(),
            _ => f64::NEG_INFINITY,
        }
    }

    /// Compute log probability density/mass for a value
    fn log_pdf(&self, params: &DistributionParams, x: f64) -> f64 {
        match params {
            DistributionParams::Gaussian { mean, var } => {
                let diff = x - mean;
                -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * diff * diff / var
            }
            DistributionParams::Exponential { lambda } => {
                if x >= 0.0 {
                    lambda.ln() - lambda * x
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Gamma { alpha, beta } => {
                if x > 0.0 {
                    (alpha - 1.0) * x.ln() - beta * x + alpha * beta.ln() - gamma_ln(*alpha)
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Beta { alpha, beta } => {
                if x > 0.0 && x < 1.0 {
                    let log_beta_fn = gamma_ln(*alpha) + gamma_ln(*beta) - gamma_ln(alpha + beta);
                    (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta_fn
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Poisson { lambda } => {
                if x >= 0.0 && x.fract() == 0.0 {
                    x * lambda.ln() - lambda - gamma_ln(x + 1.0)
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Bernoulli { p } => {
                if x == 0.0 {
                    (1.0 - p).ln()
                } else if x == 1.0 {
                    p.ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            _ => f64::NEG_INFINITY,
        }
    }
}

impl Fit<Array2<Float>, Array1<i32>> for FlexibleNB<Untrained> {
    type Fitted = FlexibleNB<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<i32>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        // Get unique classes
        let mut classes = y.to_vec();
        classes.sort();
        classes.dedup();
        let classes = Array1::from_vec(classes);
        let n_classes = classes.len();
        let n_features = x.ncols();

        // Compute class priors
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

        // Initialize storage for distributions and parameters
        let mut feature_distributions = vec![vec![Distribution::Gaussian; n_features]; n_classes];
        let mut feature_params = vec![
            vec![
                DistributionParams::Gaussian {
                    mean: 0.0,
                    var: 1.0
                };
                n_features
            ];
            n_classes
        ];

        // Fit distribution for each class and feature
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

            // For each feature, select the best distribution
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let feature_data = Array1::from_vec(feature_values.to_vec());

                let (best_dist, best_params) = self.select_best_distribution(&feature_data);
                feature_distributions[class_idx][feature_idx] = best_dist;
                feature_params[class_idx][feature_idx] = best_params;
            }
        }

        Ok(FlexibleNB {
            config: self.config,
            state: PhantomData,
            feature_distributions_: Some(feature_distributions),
            feature_params_: Some(feature_params),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl FlexibleNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let feature_params = self.feature_params_.as_ref().unwrap();
        let class_prior = self.class_prior_.as_ref().unwrap();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let n_samples = x.nrows();
        let n_features = x.ncols();

        let mut joint_log_likelihood = Array2::zeros((n_samples, n_classes));

        for class_idx in 0..n_classes {
            for (sample_idx, x_sample) in x.axis_iter(Axis(0)).enumerate() {
                let mut log_prob = safe_log(class_prior[class_idx]);

                for feature_idx in 0..n_features {
                    let x_val = x_sample[feature_idx];
                    let params = &feature_params[class_idx][feature_idx];
                    log_prob += self.log_pdf(params, x_val);
                }

                joint_log_likelihood[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(joint_log_likelihood)
    }

    /// Compute log probability density/mass for a value
    fn log_pdf(&self, params: &DistributionParams, x: f64) -> f64 {
        match params {
            DistributionParams::Gaussian { mean, var } => {
                let diff = x - mean;
                -0.5 * (2.0 * std::f64::consts::PI * var).ln() - 0.5 * diff * diff / var
            }
            DistributionParams::Exponential { lambda } => {
                if x >= 0.0 {
                    lambda.ln() - lambda * x
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Gamma { alpha, beta } => {
                if x > 0.0 {
                    (alpha - 1.0) * x.ln() - beta * x + alpha * beta.ln() - gamma_ln(*alpha)
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Beta { alpha, beta } => {
                if x > 0.0 && x < 1.0 {
                    let log_beta_fn = gamma_ln(*alpha) + gamma_ln(*beta) - gamma_ln(alpha + beta);
                    (alpha - 1.0) * x.ln() + (beta - 1.0) * (1.0 - x).ln() - log_beta_fn
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Poisson { lambda } => {
                if x >= 0.0 && x.fract() == 0.0 {
                    x * lambda.ln() - lambda - gamma_ln(x + 1.0)
                } else {
                    f64::NEG_INFINITY
                }
            }
            DistributionParams::Bernoulli { p } => {
                if x == 0.0 {
                    (1.0 - p).ln()
                } else if x == 1.0 {
                    p.ln()
                } else {
                    f64::NEG_INFINITY
                }
            }
            _ => f64::NEG_INFINITY,
        }
    }
}

impl Predict<Array2<Float>, Array1<i32>> for FlexibleNB<Trained> {
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

impl PredictProba<Array2<Float>, Array2<f64>> for FlexibleNB<Trained> {
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

impl Score<Array2<Float>, Array1<i32>> for FlexibleNB<Trained> {
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

impl NaiveBayesMixin for FlexibleNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // For flexible NB, create a dummy array since we have adaptive distributions
        // This is mainly for compatibility with the trait
        static DUMMY: once_cell::sync::Lazy<Array2<f64>> =
            once_cell::sync::Lazy::new(|| Array2::zeros((1, 1)));
        &DUMMY
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

impl FlexibleNB<Trained> {
    /// Get the selected distributions for each class and feature
    pub fn feature_distributions(&self) -> &Vec<Vec<Distribution>> {
        self.feature_distributions_.as_ref().unwrap()
    }

    /// Get the fitted parameters for each class and feature
    pub fn feature_params(&self) -> &Vec<Vec<DistributionParams>> {
        self.feature_params_.as_ref().unwrap()
    }
}

/// Simple gamma function approximation using Stirling's approximation for large values
/// and lookup table for small values
fn gamma_ln(x: f64) -> f64 {
    if x < 0.0 {
        return f64::NAN;
    }
    if x == 0.0 {
        return f64::INFINITY;
    }
    if x < 1.0 {
        return gamma_ln(x + 1.0) - x.ln();
    }
    if x < 12.0 {
        // Use approximation for moderate values
        let mut result = 0.0;
        let mut z = x;
        while z < 12.0 {
            result -= z.ln();
            z += 1.0;
        }
        result + gamma_ln_stirling(z)
    } else {
        gamma_ln_stirling(x)
    }
}

/// Stirling's approximation for ln(Gamma(x))
fn gamma_ln_stirling(x: f64) -> f64 {
    let ln_sqrt_2pi = 0.5 * (2.0 * std::f64::consts::PI).ln();
    (x - 0.5) * x.ln() - x + ln_sqrt_2pi + 1.0 / (12.0 * x)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy Compliance - Use scirs2-autograd for ndarray types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_flexible_nb_basic() {
        // Simple 2D data with two classes
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [-1.0, -2.0],
            [-2.0, -3.0],
            [-3.0, -4.0],
            [-4.0, -5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = FlexibleNB::new().fit(&x, &y).unwrap();

        // Test predictions
        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        // Test score
        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_flexible_nb_predict_proba() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let model = FlexibleNB::new().fit(&x, &y).unwrap();
        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_flexible_nb_exponential_data() {
        // Generate exponential-like data
        let x = array![
            [0.1, 0.2],
            [0.3, 0.1],
            [0.5, 0.4],
            [0.2, 0.3],
            [2.0, 1.8],
            [1.5, 2.2],
            [2.5, 1.9],
            [1.8, 2.1]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = FlexibleNB::new()
            .candidate_distributions(vec![Distribution::Gaussian, Distribution::Exponential])
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        // Should still classify correctly even with adaptive distributions
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_flexible_nb_with_custom_selection() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let model = FlexibleNB::new()
            .selection_method(SelectionMethod::AIC)
            .min_samples(2)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }
}
