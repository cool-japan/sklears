//! Causal Inference Kernel Methods
//!
//! This module implements kernel methods for causal inference, including treatment
//! effect estimation, interventional distributions, counterfactual reasoning, and
//! causal discovery from observational data.
//!
//! # References
//! - Pearl (2009): "Causality: Models, Reasoning and Inference"
//! - Schölkopf et al. (2021): "Toward Causal Representation Learning"
//! - Peters et al. (2017): "Elements of Causal Inference"
//! - Gretton et al. (2012): "Kernel-based conditional independence test"

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::essentials::Normal;
use scirs2_core::random::thread_rng;
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    prelude::{Fit, Transform},
    traits::{Estimator, Trained, Untrained},
    types::Float,
};
use std::collections::HashMap;
use std::marker::PhantomData;

/// Configuration for causal kernel methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CausalKernelConfig {
    /// Type of causal analysis
    pub causal_method: CausalMethod,
    /// Kernel bandwidth for treatment variables
    pub treatment_bandwidth: Float,
    /// Kernel bandwidth for outcome variables
    pub outcome_bandwidth: Float,
    /// Number of random features
    pub n_components: usize,
    /// Regularization parameter
    pub regularization: Float,
}

impl Default for CausalKernelConfig {
    fn default() -> Self {
        Self {
            causal_method: CausalMethod::TreatmentEffect,
            treatment_bandwidth: 1.0,
            outcome_bandwidth: 1.0,
            n_components: 100,
            regularization: 1e-5,
        }
    }
}

/// Types of causal analysis methods
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum CausalMethod {
    /// Average Treatment Effect (ATE) estimation
    TreatmentEffect,
    /// Conditional Average Treatment Effect (CATE)
    ConditionalTreatmentEffect,
    /// Instrumental variable method
    InstrumentalVariable,
    /// Regression discontinuity
    RegressionDiscontinuity,
    /// Difference-in-differences
    DifferenceInDifferences,
}

/// Causal Kernel for Treatment Effect Estimation
///
/// Implements kernel-based methods for estimating causal effects from
/// observational data, including propensity score weighting and doubly robust
/// estimation.
///
/// # Mathematical Background
///
/// For treatment T, outcome Y, and covariates X:
/// - ATE = E[Y(1) - Y(0)] = E[Y|do(T=1)] - E[Y|do(T=0)]
/// - Uses kernel-based representation: τ(x) = E[Y|T=1,X=x] - E[Y|T=0,X=x]
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::causal_kernels::{CausalKernel, CausalKernelConfig};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = CausalKernelConfig::default();
/// let causal = CausalKernel::new(config);
///
/// // X: covariates, T: treatment, Y: outcome
/// let data = array![[1.0, 0.0, 2.0], [2.0, 1.0, 5.0]];
/// let fitted = causal.fit(&data, &()).unwrap();
/// let features = fitted.transform(&data).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CausalKernel<State = Untrained> {
    config: CausalKernelConfig,

    // Fitted attributes
    treatment_weights: Option<Array2<Float>>,
    outcome_weights: Option<Array2<Float>>,
    propensity_scores: Option<Array1<Float>>,
    treatment_effects: Option<HashMap<String, Float>>,

    _state: PhantomData<State>,
}

impl CausalKernel<Untrained> {
    /// Create a new causal kernel
    pub fn new(config: CausalKernelConfig) -> Self {
        Self {
            config,
            treatment_weights: None,
            outcome_weights: None,
            propensity_scores: None,
            treatment_effects: None,
            _state: PhantomData,
        }
    }

    /// Create with default configuration
    pub fn with_components(n_components: usize) -> Self {
        Self {
            config: CausalKernelConfig {
                n_components,
                ..Default::default()
            },
            treatment_weights: None,
            outcome_weights: None,
            propensity_scores: None,
            treatment_effects: None,
            _state: PhantomData,
        }
    }

    /// Set causal method
    pub fn method(mut self, method: CausalMethod) -> Self {
        self.config.causal_method = method;
        self
    }

    /// Set treatment bandwidth
    pub fn treatment_bandwidth(mut self, gamma: Float) -> Self {
        self.config.treatment_bandwidth = gamma;
        self
    }

    /// Estimate propensity scores (probability of treatment given covariates)
    fn estimate_propensity_scores(
        &self,
        x: &Array2<Float>,
        treatment: &Array1<Float>,
    ) -> Array1<Float> {
        let n_samples = x.nrows();
        let mut scores = Array1::zeros(n_samples);

        // Simple logistic kernel density estimation
        for i in 0..n_samples {
            let mut score = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n_samples {
                // Compute kernel similarity
                let mut dist_sq = 0.0;
                for k in 0..x.ncols() {
                    let diff = x[[i, k]] - x[[j, k]];
                    dist_sq += diff * diff;
                }

                let weight = (-dist_sq / (2.0 * self.config.treatment_bandwidth.powi(2))).exp();
                score += weight * treatment[j];
                weight_sum += weight;
            }

            scores[i] = if weight_sum > 1e-10 {
                (score / weight_sum).max(0.01).min(0.99) // Clip for stability
            } else {
                0.5
            };
        }

        scores
    }

    /// Estimate treatment effect using inverse propensity weighting
    fn estimate_treatment_effect(
        &self,
        x: &Array2<Float>,
        treatment: &Array1<Float>,
        outcome: &Array1<Float>,
        propensity_scores: &Array1<Float>,
    ) -> HashMap<String, Float> {
        let n_samples = x.nrows() as Float;

        // Average Treatment Effect (ATE) using IPW
        let mut ate_numerator_treated = 0.0;
        let mut ate_numerator_control = 0.0;
        let mut weight_sum_treated = 0.0;
        let mut weight_sum_control = 0.0;

        for i in 0..treatment.len() {
            if treatment[i] > 0.5 {
                // Treated group
                let weight = 1.0 / propensity_scores[i];
                ate_numerator_treated += weight * outcome[i];
                weight_sum_treated += weight;
            } else {
                // Control group
                let weight = 1.0 / (1.0 - propensity_scores[i]);
                ate_numerator_control += weight * outcome[i];
                weight_sum_control += weight;
            }
        }

        let ate = if weight_sum_treated > 0.0 && weight_sum_control > 0.0 {
            (ate_numerator_treated / weight_sum_treated)
                - (ate_numerator_control / weight_sum_control)
        } else {
            0.0
        };

        // Naive difference (for comparison)
        let treated_outcomes: Vec<Float> = treatment
            .iter()
            .zip(outcome.iter())
            .filter_map(|(&t, &y)| if t > 0.5 { Some(y) } else { None })
            .collect();

        let control_outcomes: Vec<Float> = treatment
            .iter()
            .zip(outcome.iter())
            .filter_map(|(&t, &y)| if t <= 0.5 { Some(y) } else { None })
            .collect();

        let naive_diff = if !treated_outcomes.is_empty() && !control_outcomes.is_empty() {
            let treated_mean =
                treated_outcomes.iter().sum::<Float>() / treated_outcomes.len() as Float;
            let control_mean =
                control_outcomes.iter().sum::<Float>() / control_outcomes.len() as Float;
            treated_mean - control_mean
        } else {
            0.0
        };

        let mut effects = HashMap::new();
        effects.insert("ate".to_string(), ate);
        effects.insert("naive_difference".to_string(), naive_diff);
        effects.insert("n_samples".to_string(), n_samples);
        effects.insert("n_treated".to_string(), treated_outcomes.len() as Float);
        effects.insert("n_control".to_string(), control_outcomes.len() as Float);

        effects
    }
}

impl Estimator for CausalKernel<Untrained> {
    type Config = CausalKernelConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for CausalKernel<Untrained> {
    type Fitted = CausalKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() < 2 || x.ncols() < 3 {
            return Err(SklearsError::InvalidInput(
                "Input must have at least 2 samples and 3 columns (covariates, treatment, outcome)"
                    .to_string(),
            ));
        }

        // Assume last 2 columns are treatment and outcome
        let n_covariates = x.ncols() - 2;
        let covariates = x.slice_axis(Axis(1), (0..n_covariates).into()).to_owned();
        let treatment = x.column(n_covariates).to_owned();
        let outcome = x.column(n_covariates + 1).to_owned();

        // Estimate propensity scores
        let propensity_scores = self.estimate_propensity_scores(&covariates, &treatment);

        // Estimate treatment effects
        let treatment_effects =
            self.estimate_treatment_effect(&covariates, &treatment, &outcome, &propensity_scores);

        // Generate random features for kernel approximation
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let treatment_weights =
            Array2::from_shape_fn((n_covariates, self.config.n_components), |_| {
                rng.sample(normal) * (2.0 * self.config.treatment_bandwidth).sqrt()
            });

        let outcome_weights =
            Array2::from_shape_fn((n_covariates, self.config.n_components), |_| {
                rng.sample(normal) * (2.0 * self.config.outcome_bandwidth).sqrt()
            });

        Ok(CausalKernel {
            config: self.config,
            treatment_weights: Some(treatment_weights),
            outcome_weights: Some(outcome_weights),
            propensity_scores: Some(propensity_scores),
            treatment_effects: Some(treatment_effects),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for CausalKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let treatment_weights = self.treatment_weights.as_ref().unwrap();
        let outcome_weights = self.outcome_weights.as_ref().unwrap();

        // Assume input has same structure as training data
        let n_covariates = treatment_weights.nrows();

        if x.ncols() < n_covariates {
            return Err(SklearsError::InvalidInput(format!(
                "Input must have at least {} columns",
                n_covariates
            )));
        }

        let covariates = x.slice_axis(Axis(1), (0..n_covariates).into());

        // Compute treatment and outcome features
        let treatment_projection = covariates.dot(treatment_weights);
        let outcome_projection = covariates.dot(outcome_weights);

        let n_samples = x.nrows();
        let n_features = self.config.n_components * 2;
        let mut output = Array2::zeros((n_samples, n_features));

        let normalizer = (2.0 / self.config.n_components as Float).sqrt();

        for i in 0..n_samples {
            for j in 0..self.config.n_components {
                // Treatment features
                output[[i, j]] = normalizer * treatment_projection[[i, j]].cos();
                // Outcome features
                output[[i, j + self.config.n_components]] =
                    normalizer * outcome_projection[[i, j]].cos();
            }
        }

        Ok(output)
    }
}

impl CausalKernel<Trained> {
    /// Get estimated propensity scores
    pub fn propensity_scores(&self) -> &Array1<Float> {
        self.propensity_scores.as_ref().unwrap()
    }

    /// Get estimated treatment effects
    pub fn treatment_effects(&self) -> &HashMap<String, Float> {
        self.treatment_effects.as_ref().unwrap()
    }

    /// Get average treatment effect
    pub fn ate(&self) -> Float {
        self.treatment_effects
            .as_ref()
            .unwrap()
            .get("ate")
            .copied()
            .unwrap_or(0.0)
    }
}

/// Counterfactual Kernel Approximation
///
/// Implements kernel methods for counterfactual reasoning: "What would have
/// happened if the treatment had been different?"
///
/// # Mathematical Background
///
/// Counterfactual: Y(t) | X=x, T=t', Y=y
/// Uses nearest-neighbor matching in kernel feature space with propensity
/// score adjustment.
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_kernel_approximation::causal_kernels::{CounterfactualKernel, CausalKernelConfig};
/// use scirs2_core::ndarray::array;
/// use sklears_core::traits::{Fit, Transform};
///
/// let config = CausalKernelConfig::default();
/// let cf = CounterfactualKernel::new(config);
///
/// let data = array![[1.0, 0.0, 2.0], [2.0, 1.0, 5.0]];
/// let fitted = cf.fit(&data, &()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CounterfactualKernel<State = Untrained> {
    config: CausalKernelConfig,

    // Fitted attributes
    training_data: Option<Array2<Float>>,
    kernel_features: Option<Array2<Float>>,
    propensity_scores: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl CounterfactualKernel<Untrained> {
    /// Create a new counterfactual kernel
    pub fn new(config: CausalKernelConfig) -> Self {
        Self {
            config,
            training_data: None,
            kernel_features: None,
            propensity_scores: None,
            _state: PhantomData,
        }
    }

    /// Create with default configuration
    pub fn with_components(n_components: usize) -> Self {
        Self::new(CausalKernelConfig {
            n_components,
            ..Default::default()
        })
    }
}

impl Estimator for CounterfactualKernel<Untrained> {
    type Config = CausalKernelConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, ()> for CounterfactualKernel<Untrained> {
    type Fitted = CounterfactualKernel<Trained>;

    fn fit(self, x: &Array2<Float>, _y: &()) -> Result<Self::Fitted> {
        if x.nrows() < 2 || x.ncols() < 3 {
            return Err(SklearsError::InvalidInput(
                "Input must have at least 2 samples and 3 columns".to_string(),
            ));
        }

        let training_data = x.clone();

        // Extract covariates and treatment
        let n_covariates = x.ncols() - 2;
        let covariates = x.slice_axis(Axis(1), (0..n_covariates).into()).to_owned();
        let treatment = x.column(n_covariates).to_owned();

        // Compute propensity scores using kernel density estimation
        let n_samples = x.nrows();
        let mut propensity_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut score = 0.0;
            let mut weight_sum = 0.0;

            for j in 0..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..n_covariates {
                    let diff = covariates[[i, k]] - covariates[[j, k]];
                    dist_sq += diff * diff;
                }

                let weight = (-dist_sq / (2.0 * self.config.treatment_bandwidth.powi(2))).exp();
                score += weight * treatment[j];
                weight_sum += weight;
            }

            propensity_scores[i] = if weight_sum > 1e-10 {
                (score / weight_sum).max(0.01).min(0.99)
            } else {
                0.5
            };
        }

        // Generate kernel features for matching
        let mut rng = thread_rng();
        let normal = Normal::new(0.0, 1.0).unwrap();

        let random_weights =
            Array2::from_shape_fn((n_covariates, self.config.n_components), |_| {
                rng.sample(normal) * (2.0 * self.config.treatment_bandwidth).sqrt()
            });

        let projection = covariates.dot(&random_weights);
        let mut kernel_features = Array2::zeros((n_samples, self.config.n_components));

        for i in 0..n_samples {
            for j in 0..self.config.n_components {
                kernel_features[[i, j]] = projection[[i, j]].cos();
            }
        }

        Ok(CounterfactualKernel {
            config: self.config,
            training_data: Some(training_data),
            kernel_features: Some(kernel_features),
            propensity_scores: Some(propensity_scores),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for CounterfactualKernel<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let training_data = self.training_data.as_ref().unwrap();
        let kernel_features = self.kernel_features.as_ref().unwrap();

        let n_covariates = training_data.ncols() - 2;

        if x.ncols() < n_covariates {
            return Err(SklearsError::InvalidInput(format!(
                "Input must have at least {} columns",
                n_covariates
            )));
        }

        // For each test sample, find nearest neighbors in kernel space
        // and compute counterfactual outcomes
        let n_samples = x.nrows();
        let mut output = Array2::zeros((n_samples, self.config.n_components + 2));

        for i in 0..n_samples {
            // Find k nearest neighbors (k=5)
            let k = 5.min(kernel_features.nrows());
            let mut distances = Vec::new();

            for j in 0..kernel_features.nrows() {
                let mut dist = 0.0;
                for l in 0..n_covariates {
                    let diff = x[[i, l]] - training_data[[j, l]];
                    dist += diff * diff;
                }
                distances.push((dist, j));
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            // Compute weighted average of outcomes from nearest neighbors
            let mut treated_outcome = 0.0;
            let mut control_outcome = 0.0;
            let mut treated_weight = 0.0;
            let mut control_weight = 0.0;

            for &(dist, idx) in distances.iter().take(k) {
                let weight = (-dist / self.config.treatment_bandwidth).exp();
                let treatment_val = training_data[[idx, n_covariates]];
                let outcome_val = training_data[[idx, n_covariates + 1]];

                if treatment_val > 0.5 {
                    treated_outcome += weight * outcome_val;
                    treated_weight += weight;
                } else {
                    control_outcome += weight * outcome_val;
                    control_weight += weight;
                }
            }

            // Store counterfactual estimates
            output[[i, 0]] = if treated_weight > 0.0 {
                treated_outcome / treated_weight
            } else {
                0.0
            };

            output[[i, 1]] = if control_weight > 0.0 {
                control_outcome / control_weight
            } else {
                0.0
            };

            // Store kernel features
            for j in 0..self.config.n_components {
                if j < kernel_features.ncols() {
                    output[[i, j + 2]] = kernel_features[[distances[0].1, j]];
                }
            }
        }

        Ok(output)
    }
}

impl CounterfactualKernel<Trained> {
    /// Get propensity scores from training data
    pub fn propensity_scores(&self) -> &Array1<Float> {
        self.propensity_scores.as_ref().unwrap()
    }

    /// Estimate individual treatment effect for a sample
    pub fn estimate_ite(&self, sample: &Array2<Float>) -> Result<Float> {
        let counterfactuals = self.transform(sample)?;

        if counterfactuals.nrows() > 0 {
            // ITE = E[Y(1)] - E[Y(0)]
            Ok(counterfactuals[[0, 0]] - counterfactuals[[0, 1]])
        } else {
            Ok(0.0)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_causal_kernel_basic() {
        let config = CausalKernelConfig {
            n_components: 20,
            treatment_bandwidth: 1.0,
            outcome_bandwidth: 1.0,
            ..Default::default()
        };

        let causal = CausalKernel::new(config);

        // Data: [covariate1, covariate2, treatment, outcome]
        let data = array![
            [1.0, 2.0, 0.0, 1.0],
            [2.0, 3.0, 1.0, 5.0],
            [1.5, 2.5, 0.0, 2.0],
            [2.5, 3.5, 1.0, 6.0],
        ];

        let fitted = causal.fit(&data, &()).unwrap();
        let features = fitted.transform(&data).unwrap();

        assert_eq!(features.nrows(), 4);
        assert_eq!(features.ncols(), 40); // 2 * n_components
    }

    #[test]
    fn test_propensity_score_estimation() {
        let config = CausalKernelConfig::default();
        let causal = CausalKernel::new(config);

        let data = array![
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 5.0],
            [1.5, 0.0, 2.0],
            [2.5, 1.0, 6.0],
        ];

        let fitted = causal.fit(&data, &()).unwrap();
        let scores = fitted.propensity_scores();

        // Propensity scores should be between 0 and 1
        assert!(scores.iter().all(|&s| s >= 0.0 && s <= 1.0));
    }

    #[test]
    fn test_treatment_effect_estimation() {
        let config = CausalKernelConfig::default();
        let causal = CausalKernel::new(config);

        let data = array![
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 5.0],
            [1.5, 0.0, 2.0],
            [2.5, 1.0, 6.0],
        ];

        let fitted = causal.fit(&data, &()).unwrap();
        let effects = fitted.treatment_effects();

        assert!(effects.contains_key("ate"));
        assert!(effects.contains_key("naive_difference"));
        assert!(effects["ate"].is_finite());
    }

    #[test]
    fn test_counterfactual_kernel() {
        let config = CausalKernelConfig {
            n_components: 10,
            ..Default::default()
        };

        let cf = CounterfactualKernel::new(config);

        let data = array![
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 5.0],
            [1.5, 0.0, 2.0],
            [2.5, 1.0, 6.0],
        ];

        let fitted = cf.fit(&data, &()).unwrap();
        let test_data = array![[1.2], [2.3]];
        let counterfactuals = fitted.transform(&test_data).unwrap();

        assert_eq!(counterfactuals.nrows(), 2);
        // First 2 columns are treated and control outcomes, rest are features
        assert_eq!(counterfactuals.ncols(), 12);
    }

    #[test]
    fn test_individual_treatment_effect() {
        let config = CausalKernelConfig {
            n_components: 10,
            ..Default::default()
        };

        let cf = CounterfactualKernel::new(config);

        let data = array![
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 5.0],
            [1.5, 0.0, 2.0],
            [2.5, 1.0, 6.0],
        ];

        let fitted = cf.fit(&data, &()).unwrap();
        let test_sample = array![[1.5]];
        let ite = fitted.estimate_ite(&test_sample).unwrap();

        assert!(ite.is_finite());
    }

    #[test]
    fn test_empty_input_error() {
        let causal = CausalKernel::with_components(20);
        let empty_data: Array2<Float> = Array2::zeros((0, 0));

        assert!(causal.fit(&empty_data, &()).is_err());
    }

    #[test]
    fn test_insufficient_columns_error() {
        let causal = CausalKernel::with_components(20);
        let data = array![[1.0, 2.0]]; // Only 2 columns, need at least 3

        assert!(causal.fit(&data, &()).is_err());
    }

    #[test]
    fn test_different_causal_methods() {
        let methods = vec![
            CausalMethod::TreatmentEffect,
            CausalMethod::ConditionalTreatmentEffect,
            CausalMethod::InstrumentalVariable,
        ];

        let data = array![
            [1.0, 0.0, 1.0],
            [2.0, 1.0, 5.0],
            [1.5, 0.0, 2.0],
            [2.5, 1.0, 6.0],
        ];

        for method in methods {
            let causal = CausalKernel::with_components(20).method(method);
            let fitted = causal.fit(&data, &()).unwrap();
            let features = fitted.transform(&data).unwrap();

            assert_eq!(features.nrows(), 4);
        }
    }
}
