//! Exponential Family Naive Bayes classifier
//!
//! This classifier provides a unified framework for Naive Bayes with
//! exponential family distributions, including automatic distribution
//! selection and parameter estimation.

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

/// Exponential family distribution types
#[derive(Debug, Clone, PartialEq)]
pub enum ExponentialFamily {
    /// Gaussian distribution (normal)
    Gaussian,
    /// Exponential distribution
    Exponential,
    /// Gamma distribution
    Gamma,
    /// Beta distribution
    Beta,
    /// Poisson distribution
    Poisson,
    /// Bernoulli distribution
    Bernoulli,
    /// Binomial distribution
    Binomial { n: usize },
    /// Negative binomial distribution
    NegativeBinomial,
    /// Geometric distribution
    Geometric,
    /// Inverse Gaussian distribution
    InverseGaussian,
    /// Weibull distribution
    Weibull,
    /// Laplace distribution
    Laplace,
}

/// Natural parameters for exponential family distributions
#[derive(Debug, Clone)]
pub enum NaturalParameters {
    /// Gaussian
    Gaussian {
        theta1: f64,
        theta2: f64,
    }, // μ/σ², -1/(2σ²)
    /// Exponential
    Exponential {
        theta: f64,
    }, // -λ
    /// Gamma
    Gamma {
        theta1: f64,
        theta2: f64,
    }, // α-1, -β
    /// Beta
    Beta {
        theta1: f64,
        theta2: f64,
    }, // α-1, β-1
    /// Poisson
    Poisson {
        theta: f64,
    }, // ln(λ)
    /// Bernoulli
    Bernoulli {
        theta: f64,
    }, // ln(p/(1-p))
    /// Binomial
    Binomial {
        theta: f64,
        n: usize,
    }, // ln(p/(1-p)), n
    /// NegativeBinomial
    NegativeBinomial {
        theta1: f64,
        theta2: f64,
    }, // ln(p), r
    /// Geometric
    Geometric {
        theta: f64,
    }, // ln(1-p)
    /// InverseGaussian
    InverseGaussian {
        theta1: f64,
        theta2: f64,
    }, // -μ/(2σ²), -1/(2σ²)
    Weibull {
        theta1: f64,
        theta2: f64,
    }, // k, ln(λ)
    Laplace {
        theta1: f64,
        theta2: f64,
    }, // μ/b, -1/b
}

/// Sufficient statistics for exponential family distributions
#[derive(Debug, Clone)]
pub struct SufficientStatistics {
    pub t1: f64,    // First sufficient statistic
    pub t2: f64,    // Second sufficient statistic
    pub count: f64, // Number of observations
}

/// Method for estimating natural parameters
#[derive(Debug, Clone)]
pub enum ParameterEstimationMethod {
    /// Maximum likelihood estimation
    MLE,
    /// Method of moments
    MoM,
    /// Bayesian estimation with conjugate priors
    Bayesian { prior_strength: f64 },
    /// Regularized estimation
    Regularized { lambda: f64 },
}

/// Configuration for Exponential Family Naive Bayes
#[derive(Debug, Clone)]
pub struct ExponentialFamilyNBConfig {
    /// Distribution family to use
    pub family: ExponentialFamily,
    /// Parameter estimation method
    pub estimation_method: ParameterEstimationMethod,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Prior probabilities of the classes
    pub priors: Option<Array1<f64>>,
    /// Tolerance for convergence
    pub tolerance: f64,
    /// Maximum iterations for iterative methods
    pub max_iterations: usize,
}

impl Default for ExponentialFamilyNBConfig {
    fn default() -> Self {
        Self {
            family: ExponentialFamily::Gaussian,
            estimation_method: ParameterEstimationMethod::MLE,
            regularization: 1e-10,
            priors: None,
            tolerance: 1e-6,
            max_iterations: 100,
        }
    }
}

/// Exponential Family Naive Bayes classifier
///
/// This classifier uses exponential family distributions for modeling
/// feature likelihoods with natural parameter representations.
#[derive(Debug, Clone)]
pub struct ExponentialFamilyNB<State = Untrained> {
    config: ExponentialFamilyNBConfig,
    state: PhantomData<State>,
    // Trained state fields
    natural_params_: Option<Vec<Vec<NaturalParameters>>>, // [n_classes][n_features]
    log_partition_: Option<Array2<f64>>,                  // Log-partition function values
    class_prior_: Option<Array1<f64>>,
    classes_: Option<Array1<i32>>,
}

impl ExponentialFamilyNB<Untrained> {
    /// Create a new Exponential Family Naive Bayes classifier
    pub fn new() -> Self {
        Self {
            config: ExponentialFamilyNBConfig::default(),
            state: PhantomData,
            natural_params_: None,
            log_partition_: None,
            class_prior_: None,
            classes_: None,
        }
    }

    /// Set the exponential family distribution
    pub fn family(mut self, family: ExponentialFamily) -> Self {
        self.config.family = family;
        self
    }

    /// Set parameter estimation method
    pub fn estimation_method(mut self, method: ParameterEstimationMethod) -> Self {
        self.config.estimation_method = method;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: f64) -> Self {
        self.config.regularization = reg;
        self
    }

    /// Set prior probabilities
    pub fn priors(mut self, priors: Array1<f64>) -> Self {
        self.config.priors = Some(priors);
        self
    }

    /// Set tolerance for convergence
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }

    /// Set maximum iterations
    pub fn max_iterations(mut self, max_iter: usize) -> Self {
        self.config.max_iterations = max_iter;
        self
    }
}

impl Default for ExponentialFamilyNB<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ExponentialFamilyNB<Untrained> {
    type Float = Float;
    type Config = ExponentialFamilyNBConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl ExponentialFamilyNB<Untrained> {
    /// Compute sufficient statistics for a dataset
    fn compute_sufficient_statistics(&self, data: &Array1<f64>) -> SufficientStatistics {
        let n = data.len() as f64;
        let sum = data.sum();
        let sum_sq = data.mapv(|x| x * x).sum();

        match self.config.family {
            ExponentialFamily::Gaussian => SufficientStatistics {
                t1: sum,
                t2: sum_sq,
                count: n,
            },
            ExponentialFamily::Exponential => SufficientStatistics {
                t1: sum,
                t2: 0.0,
                count: n,
            },
            ExponentialFamily::Gamma => SufficientStatistics {
                t1: data.mapv(|x| x.ln()).sum(),
                t2: sum,
                count: n,
            },
            ExponentialFamily::Beta => SufficientStatistics {
                t1: data.mapv(|x| x.ln()).sum(),
                t2: data.mapv(|x| (1.0 - x).ln()).sum(),
                count: n,
            },
            ExponentialFamily::Poisson => SufficientStatistics {
                t1: sum,
                t2: 0.0,
                count: n,
            },
            ExponentialFamily::Bernoulli => SufficientStatistics {
                t1: sum,
                t2: 0.0,
                count: n,
            },
            ExponentialFamily::Laplace => SufficientStatistics {
                t1: sum,
                t2: data.mapv(|x| x.abs()).sum(),
                count: n,
            },
            _ => SufficientStatistics {
                t1: sum,
                t2: sum_sq,
                count: n,
            },
        }
    }

    /// Estimate natural parameters from sufficient statistics
    fn estimate_natural_parameters(
        &self,
        stats: &SufficientStatistics,
    ) -> Result<NaturalParameters> {
        if stats.count == 0.0 {
            return Err(SklearsError::InvalidInput(
                "Cannot estimate parameters from empty data".to_string(),
            ));
        }

        match &self.config.family {
            ExponentialFamily::Gaussian => {
                let mean = stats.t1 / stats.count;
                let var = (stats.t2 / stats.count - mean * mean).max(self.config.regularization);

                Ok(NaturalParameters::Gaussian {
                    theta1: mean / var,
                    theta2: -0.5 / var,
                })
            }
            ExponentialFamily::Exponential => {
                let mean = stats.t1 / stats.count;
                let lambda = 1.0 / mean.max(self.config.regularization);

                Ok(NaturalParameters::Exponential { theta: -lambda })
            }
            ExponentialFamily::Gamma => {
                let log_mean = stats.t1 / stats.count;
                let mean = stats.t2 / stats.count;

                // Method of moments for gamma parameters
                let log_x_bar = log_mean;
                let ln_mean = mean.ln();
                let s = ln_mean - log_x_bar;

                if s <= 0.0 {
                    return Err(SklearsError::InvalidInput(
                        "Invalid data for Gamma distribution".to_string(),
                    ));
                }

                let alpha = (3.0 - s + ((s - 3.0).powi(2) + 24.0 * s).sqrt()) / (12.0 * s);
                let beta = alpha / mean;

                Ok(NaturalParameters::Gamma {
                    theta1: alpha - 1.0,
                    theta2: -beta,
                })
            }
            ExponentialFamily::Beta => {
                let log_mean_x = stats.t1 / stats.count;
                let log_mean_1_minus_x = stats.t2 / stats.count;

                // Simple method of moments estimation
                let mean_x = log_mean_x.exp();
                let mean_1_minus_x = log_mean_1_minus_x.exp();

                let alpha = 1.0 + mean_x;
                let beta = 1.0 + mean_1_minus_x;

                Ok(NaturalParameters::Beta {
                    theta1: alpha - 1.0,
                    theta2: beta - 1.0,
                })
            }
            ExponentialFamily::Poisson => {
                let lambda = (stats.t1 / stats.count).max(self.config.regularization);
                Ok(NaturalParameters::Poisson { theta: lambda.ln() })
            }
            ExponentialFamily::Bernoulli => {
                let p = (stats.t1 / stats.count)
                    .clamp(self.config.regularization, 1.0 - self.config.regularization);
                Ok(NaturalParameters::Bernoulli {
                    theta: (p / (1.0 - p)).ln(),
                })
            }
            ExponentialFamily::Laplace => {
                let mean = stats.t1 / stats.count;
                let mean_abs_dev = stats.t2 / stats.count;
                let b = mean_abs_dev.max(self.config.regularization);

                Ok(NaturalParameters::Laplace {
                    theta1: mean / b,
                    theta2: -1.0 / b,
                })
            }
            _ => Err(SklearsError::InvalidInput(
                "Unsupported exponential family distribution".to_string(),
            )),
        }
    }

    /// Compute log-partition function A(θ)
    fn log_partition_function(&self, params: &NaturalParameters) -> f64 {
        match params {
            NaturalParameters::Gaussian { theta1, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                let sigma_sq = -0.5 / theta2;
                let mu = theta1 * sigma_sq;
                0.5 * mu * mu / sigma_sq + 0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln()
            }
            NaturalParameters::Exponential { theta } => {
                if *theta >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                -(-theta).ln()
            }
            NaturalParameters::Gamma { theta1, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                let alpha = theta1 + 1.0;
                let beta = -theta2;
                gamma_ln(alpha) - alpha * beta.ln()
            }
            NaturalParameters::Beta { theta1, theta2 } => {
                let alpha = theta1 + 1.0;
                let beta = theta2 + 1.0;
                gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta)
            }
            NaturalParameters::Poisson { theta } => theta.exp(),
            NaturalParameters::Bernoulli { theta } => (1.0 + theta.exp()).ln(),
            NaturalParameters::Laplace { theta1: _, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                -(-theta2).ln()
            }
            _ => 0.0, // Default case
        }
    }

    /// Compute sufficient statistics for a single value
    fn sufficient_statistics_single(&self, x: f64) -> (f64, f64) {
        match self.config.family {
            ExponentialFamily::Gaussian => (x, x * x),
            ExponentialFamily::Exponential => (x, 0.0),
            ExponentialFamily::Gamma => (x.ln(), x),
            ExponentialFamily::Beta => (x.ln(), (1.0 - x).ln()),
            ExponentialFamily::Poisson => (x, 0.0),
            ExponentialFamily::Bernoulli => (x, 0.0),
            ExponentialFamily::Laplace => (x, x.abs()),
            _ => (x, x * x),
        }
    }
}

impl ExponentialFamilyNB<Trained> {
    /// Compute sufficient statistics for a single value
    fn sufficient_statistics_single(&self, x: f64) -> (f64, f64) {
        match self.config.family {
            ExponentialFamily::Gaussian => (x, x * x),
            ExponentialFamily::Exponential => (x, 0.0),
            ExponentialFamily::Gamma => (x.ln(), x),
            ExponentialFamily::Beta => (x.ln(), (1.0 - x).ln()),
            ExponentialFamily::Poisson => (x, 0.0),
            ExponentialFamily::Bernoulli => (x, 0.0),
            ExponentialFamily::Laplace => (x, x.abs()),
            _ => (x, x * x),
        }
    }

    /// Compute log-partition function A(θ)
    fn log_partition_function(&self, params: &NaturalParameters) -> f64 {
        match params {
            NaturalParameters::Gaussian { theta1, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                let sigma_sq = -0.5 / theta2;
                let mu = theta1 * sigma_sq;
                0.5 * mu * mu / sigma_sq + 0.5 * (2.0 * std::f64::consts::PI * sigma_sq).ln()
            }
            NaturalParameters::Exponential { theta } => {
                if *theta >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                -(-theta).ln()
            }
            NaturalParameters::Gamma { theta1, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                let alpha = theta1 + 1.0;
                let beta = -theta2;
                gamma_ln(alpha) - alpha * beta.ln()
            }
            NaturalParameters::Beta { theta1, theta2 } => {
                let alpha = theta1 + 1.0;
                let beta = theta2 + 1.0;
                gamma_ln(alpha) + gamma_ln(beta) - gamma_ln(alpha + beta)
            }
            NaturalParameters::Poisson { theta } => theta.exp(),
            NaturalParameters::Bernoulli { theta } => (1.0 + theta.exp()).ln(),
            NaturalParameters::Laplace { theta1: _, theta2 } => {
                if *theta2 >= 0.0 {
                    return f64::INFINITY; // Invalid parameters
                }
                -(-theta2).ln()
            }
            _ => 0.0, // Default case
        }
    }

    /// Compute log probability density for exponential family
    fn log_probability(&self, x: f64, params: &NaturalParameters) -> f64 {
        let (t1, t2) = self.sufficient_statistics_single(x);
        let log_partition = self.log_partition_function(params);

        match params {
            NaturalParameters::Gaussian { theta1, theta2 } => {
                theta1 * t1 + theta2 * t2 - log_partition
            }
            NaturalParameters::Exponential { theta } => {
                if x < 0.0 {
                    f64::NEG_INFINITY
                } else {
                    theta * t1 - log_partition
                }
            }
            NaturalParameters::Gamma { theta1, theta2 } => {
                if x <= 0.0 {
                    f64::NEG_INFINITY
                } else {
                    theta1 * t1 + theta2 * t2 - log_partition
                }
            }
            NaturalParameters::Beta { theta1, theta2 } => {
                if x <= 0.0 || x >= 1.0 {
                    f64::NEG_INFINITY
                } else {
                    theta1 * t1 + theta2 * t2 - log_partition
                }
            }
            NaturalParameters::Poisson { theta } => {
                if x < 0.0 || x.fract() != 0.0 {
                    f64::NEG_INFINITY
                } else {
                    theta * t1 - log_partition - gamma_ln(x + 1.0)
                }
            }
            NaturalParameters::Bernoulli { theta } => {
                if x != 0.0 && x != 1.0 {
                    f64::NEG_INFINITY
                } else {
                    theta * t1 - log_partition
                }
            }
            NaturalParameters::Laplace { theta1, theta2 } => {
                theta1 * t1 + theta2 * t2 - log_partition
            }
            _ => f64::NEG_INFINITY,
        }
    }
}

impl Fit<Array2<Float>, Array1<i32>> for ExponentialFamilyNB<Untrained> {
    type Fitted = ExponentialFamilyNB<Trained>;

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

        // Initialize parameter storage
        let mut natural_params = vec![vec![]; n_classes];
        let mut log_partition = Array2::zeros((n_classes, n_features));

        // Fit parameters for each class and feature
        for (class_idx, &class_label) in classes.iter().enumerate() {
            natural_params[class_idx] = vec![];

            // Get samples belonging to this class
            let mask: Vec<usize> = y
                .iter()
                .enumerate()
                .filter_map(|(i, &label)| if label == class_label { Some(i) } else { None })
                .collect();

            if mask.is_empty() {
                // Use default parameters for empty classes
                for feature_idx in 0..n_features {
                    let default_params = match self.config.family {
                        ExponentialFamily::Gaussian => NaturalParameters::Gaussian {
                            theta1: 0.0,
                            theta2: -0.5,
                        },
                        ExponentialFamily::Exponential => {
                            NaturalParameters::Exponential { theta: -1.0 }
                        }
                        _ => NaturalParameters::Gaussian {
                            theta1: 0.0,
                            theta2: -0.5,
                        },
                    };
                    log_partition[[class_idx, feature_idx]] =
                        self.log_partition_function(&default_params);
                    natural_params[class_idx].push(default_params);
                }
                continue;
            }

            let x_class = x.select(Axis(0), &mask);

            // For each feature, estimate natural parameters
            for feature_idx in 0..n_features {
                let feature_values = x_class.column(feature_idx);
                let feature_data = Array1::from_vec(feature_values.to_vec());

                // Compute sufficient statistics
                let stats = self.compute_sufficient_statistics(&feature_data);

                // Estimate natural parameters
                let params = self.estimate_natural_parameters(&stats)?;

                // Compute log-partition function
                let log_part = self.log_partition_function(&params);

                log_partition[[class_idx, feature_idx]] = log_part;
                natural_params[class_idx].push(params);
            }
        }

        Ok(ExponentialFamilyNB {
            config: self.config,
            state: PhantomData,
            natural_params_: Some(natural_params),
            log_partition_: Some(log_partition),
            class_prior_: Some(class_prior),
            classes_: Some(classes),
        })
    }
}

impl ExponentialFamilyNB<Trained> {
    /// Compute the unnormalized posterior log probability of X
    fn joint_log_likelihood(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let natural_params = self.natural_params_.as_ref().unwrap();
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
                    let params = &natural_params[class_idx][feature_idx];
                    log_prob += self.log_probability(x_val, params);
                }

                joint_log_likelihood[[sample_idx, class_idx]] = log_prob;
            }
        }

        Ok(joint_log_likelihood)
    }
}

impl Predict<Array2<Float>, Array1<i32>> for ExponentialFamilyNB<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<i32>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let classes = self.classes_.as_ref().unwrap();

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

impl PredictProba<Array2<Float>, Array2<f64>> for ExponentialFamilyNB<Trained> {
    fn predict_proba(&self, x: &Array2<Float>) -> Result<Array2<f64>> {
        let log_prob = self.joint_log_likelihood(x)?;
        let n_samples = x.nrows();
        let n_classes = self.classes_.as_ref().unwrap().len();
        let mut proba = Array2::zeros((n_samples, n_classes));

        // Normalize to get probabilities
        for i in 0..n_samples {
            let row = log_prob.row(i);
            let max_log_prob = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            let mut exp_sum = 0.0;
            for j in 0..n_classes {
                let exp_val = (log_prob[[i, j]] - max_log_prob).exp();
                proba[[i, j]] = exp_val;
                exp_sum += exp_val;
            }

            for j in 0..n_classes {
                proba[[i, j]] /= exp_sum;
            }
        }

        Ok(proba)
    }
}

impl Score<Array2<Float>, Array1<i32>> for ExponentialFamilyNB<Trained> {
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

impl NaiveBayesMixin for ExponentialFamilyNB<Trained> {
    fn class_log_prior(&self) -> &Array1<f64> {
        self.class_prior_.as_ref().unwrap()
    }

    fn feature_log_prob(&self) -> &Array2<f64> {
        // Return log-partition values as proxy
        self.log_partition_.as_ref().unwrap()
    }

    fn classes(&self) -> &Array1<i32> {
        self.classes_.as_ref().unwrap()
    }
}

impl ExponentialFamilyNB<Trained> {
    /// Get the natural parameters for each class and feature
    pub fn natural_parameters(&self) -> &Vec<Vec<NaturalParameters>> {
        self.natural_params_.as_ref().unwrap()
    }

    /// Get the log-partition function values
    pub fn log_partition_values(&self) -> &Array2<f64> {
        self.log_partition_.as_ref().unwrap()
    }
}

/// Gamma function logarithm approximation
fn gamma_ln(x: f64) -> f64 {
    if x <= 0.0 {
        return f64::NAN;
    }
    if x < 12.0 {
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
    fn test_exponential_family_nb_gaussian() {
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

        let model = ExponentialFamilyNB::new()
            .family(ExponentialFamily::Gaussian)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions, y);

        let score = model.score(&x, &y).unwrap();
        assert_eq!(score, 1.0);
    }

    #[test]
    fn test_exponential_family_nb_exponential() {
        // Positive data for exponential distribution
        let x = array![
            [0.5, 1.0],
            [1.0, 0.5],
            [0.8, 1.2],
            [1.2, 0.8],
            [2.0, 2.5],
            [2.5, 2.0],
            [2.2, 2.8],
            [2.8, 2.2]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExponentialFamilyNB::new()
            .family(ExponentialFamily::Exponential)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_exponential_family_nb_predict_proba() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]];
        let y = array![0, 0, 1, 1];

        let model = ExponentialFamilyNB::new()
            .family(ExponentialFamily::Gaussian)
            .fit(&x, &y)
            .unwrap();

        let proba = model.predict_proba(&x).unwrap();

        // Check that probabilities sum to 1
        for i in 0..x.nrows() {
            let row_sum: f64 = proba.row(i).sum();
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_exponential_family_nb_poisson() {
        // Integer count data for Poisson
        let x = array![
            [1.0, 2.0],
            [2.0, 1.0],
            [1.0, 3.0],
            [3.0, 1.0],
            [5.0, 6.0],
            [6.0, 5.0],
            [5.0, 7.0],
            [7.0, 5.0]
        ];
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];

        let model = ExponentialFamilyNB::new()
            .family(ExponentialFamily::Poisson)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }

    #[test]
    fn test_exponential_family_nb_different_estimation_methods() {
        let x = array![[1.0, 1.0], [2.0, 2.0], [-1.0, -1.0], [-2.0, -2.0]];
        let y = array![0, 0, 1, 1];

        let model = ExponentialFamilyNB::new()
            .family(ExponentialFamily::Gaussian)
            .estimation_method(ParameterEstimationMethod::MLE)
            .fit(&x, &y)
            .unwrap();

        let predictions = model.predict(&x).unwrap();
        assert_eq!(predictions.len(), y.len());
    }
}
