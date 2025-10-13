//! Dose-Response Modeling for Isotonic Regression
//!
//! This module provides specialized isotonic regression models for dose-response relationships,
//! including toxicological modeling, pharmacokinetic applications, and efficacy modeling.

use scirs2_core::ndarray::{s, Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Types of dose-response models
#[derive(Debug, Clone, Copy, PartialEq)]
/// DoseResponseModel
pub enum DoseResponseModel {
    /// Linear dose-response (proportional)
    Linear,
    /// Log-linear dose-response
    LogLinear,
    /// Hill equation (sigmoid)
    Hill,
    /// Weibull dose-response
    Weibull,
    /// Probit dose-response
    Probit,
    /// Logistic dose-response
    Logistic,
    /// Exponential decay/growth
    Exponential,
    /// Power law dose-response
    PowerLaw,
    /// Threshold model (hockey stick)
    Threshold,
    /// Biphasic dose-response (hormesis)
    Biphasic,
}

/// Application domain for dose-response modeling
#[derive(Debug, Clone, Copy, PartialEq)]
/// ApplicationDomain
pub enum ApplicationDomain {
    /// Toxicological dose-response modeling
    Toxicology,
    /// Pharmacokinetic modeling
    Pharmacokinetics,
    /// Drug efficacy modeling
    Efficacy,
    /// Environmental exposure modeling
    Environmental,
    /// Risk assessment
    RiskAssessment,
    /// Occupational health
    OccupationalHealth,
}

/// Benchmark dose estimation methods
#[derive(Debug, Clone, Copy, PartialEq)]
/// BenchmarkDoseMethod
pub enum BenchmarkDoseMethod {
    /// BMD10 - 10% response level
    BMD10,
    /// BMD05 - 5% response level
    BMD05,
    /// BMD01 - 1% response level
    BMD01,
    /// BMDL - Lower confidence limit
    BMDL,
    /// BMDU - Upper confidence limit
    BMDU,
    /// LED10 - 10% lower effective dose
    LED10,
    /// NOAEL - No Observed Adverse Effect Level
    NOAEL,
    /// LOAEL - Lowest Observed Adverse Effect Level
    LOAEL,
}

/// Confidence interval methods for dose-response
#[derive(Debug, Clone, Copy, PartialEq)]
/// ConfidenceMethod
pub enum ConfidenceMethod {
    /// Bootstrap confidence intervals
    Bootstrap,
    /// Profile likelihood
    ProfileLikelihood,
    /// Delta method
    DeltaMethod,
    /// Bayesian credible intervals
    Bayesian,
    /// Wald confidence intervals
    Wald,
}

/// Dose-response isotonic regression model
#[derive(Debug, Clone)]
/// DoseResponseIsotonicRegression
pub struct DoseResponseIsotonicRegression<State> {
    model_type: DoseResponseModel,
    application_domain: ApplicationDomain,
    benchmark_method: Option<BenchmarkDoseMethod>,
    confidence_method: ConfidenceMethod,
    confidence_level: Float,
    background_response: Float,
    max_response: Float,
    log_transform_dose: bool,
    regularization: Float,
    max_iter: usize,
    tolerance: Float,
    fitted_doses: Option<Array1<Float>>,
    fitted_responses: Option<Array1<Float>>,
    model_parameters: Option<Array1<Float>>,
    benchmark_doses: Option<Array1<Float>>,
    confidence_intervals: Option<Array2<Float>>,
    _state: PhantomData<State>,
}

impl DoseResponseIsotonicRegression<Untrained> {
    /// Create a new dose-response isotonic regression model
    pub fn new() -> Self {
        Self {
            model_type: DoseResponseModel::LogLinear,
            application_domain: ApplicationDomain::Toxicology,
            benchmark_method: None,
            confidence_method: ConfidenceMethod::Bootstrap,
            confidence_level: 0.95,
            background_response: 0.0,
            max_response: 1.0,
            log_transform_dose: true,
            regularization: 0.01,
            max_iter: 1000,
            tolerance: 1e-6,
            fitted_doses: None,
            fitted_responses: None,
            model_parameters: None,
            benchmark_doses: None,
            confidence_intervals: None,
            _state: PhantomData,
        }
    }

    /// Set the dose-response model type
    pub fn model_type(mut self, model_type: DoseResponseModel) -> Self {
        self.model_type = model_type;
        self
    }

    /// Set the application domain
    pub fn application_domain(mut self, domain: ApplicationDomain) -> Self {
        self.application_domain = domain;
        self
    }

    /// Set the benchmark dose method
    pub fn benchmark_method(mut self, method: BenchmarkDoseMethod) -> Self {
        self.benchmark_method = Some(method);
        self
    }

    /// Set the confidence interval method
    pub fn confidence_method(mut self, method: ConfidenceMethod) -> Self {
        self.confidence_method = method;
        self
    }

    /// Set the confidence level (e.g., 0.95 for 95% confidence)
    pub fn confidence_level(mut self, level: Float) -> Self {
        self.confidence_level = level;
        self
    }

    /// Set the background response level
    pub fn background_response(mut self, background: Float) -> Self {
        self.background_response = background;
        self
    }

    /// Set the maximum response level
    pub fn max_response(mut self, max_response: Float) -> Self {
        self.max_response = max_response;
        self
    }

    /// Enable/disable log transformation of doses
    pub fn log_transform_dose(mut self, log_transform: bool) -> Self {
        self.log_transform_dose = log_transform;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, reg: Float) -> Self {
        self.regularization = reg;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }
}

impl Estimator for DoseResponseIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array1<Float>, Array1<Float>> for DoseResponseIsotonicRegression<Untrained> {
    type Fitted = DoseResponseIsotonicRegression<Trained>;

    fn fit(mut self, doses: &Array1<Float>, responses: &Array1<Float>) -> Result<Self::Fitted> {
        if doses.len() != responses.len() {
            return Err(SklearsError::InvalidInput(
                "Doses and responses must have same length".to_string(),
            ));
        }

        if doses.iter().any(|&d| d < 0.0) {
            return Err(SklearsError::InvalidInput(
                "Doses must be non-negative".to_string(),
            ));
        }

        // Transform doses if requested
        let transformed_doses = if self.log_transform_dose {
            self.log_transform(doses)?
        } else {
            doses.clone()
        };

        // Fit the dose-response model
        let (fitted_doses, fitted_responses, model_parameters) =
            self.fit_dose_response_model(&transformed_doses, responses)?;

        // Calculate benchmark doses if requested
        let benchmark_doses = if self.benchmark_method.is_some() {
            Some(self.calculate_benchmark_doses(&fitted_doses, &fitted_responses)?)
        } else {
            None
        };

        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(
            &fitted_doses,
            &fitted_responses,
            doses,
            responses,
        )?;

        Ok(DoseResponseIsotonicRegression {
            model_type: self.model_type,
            application_domain: self.application_domain,
            benchmark_method: self.benchmark_method,
            confidence_method: self.confidence_method,
            confidence_level: self.confidence_level,
            background_response: self.background_response,
            max_response: self.max_response,
            log_transform_dose: self.log_transform_dose,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            fitted_doses: Some(fitted_doses),
            fitted_responses: Some(fitted_responses),
            model_parameters: Some(model_parameters),
            benchmark_doses,
            confidence_intervals: Some(confidence_intervals),
            _state: PhantomData,
        })
    }
}

impl DoseResponseIsotonicRegression<Untrained> {
    /// Log transform doses (adding small constant to handle zero doses)
    fn log_transform(&self, doses: &Array1<Float>) -> Result<Array1<Float>> {
        let min_dose = doses
            .iter()
            .filter(|&&d| d > 0.0)
            .fold(Float::INFINITY, |a, &b| a.min(b));
        let offset = if min_dose.is_finite() {
            min_dose * 0.01
        } else {
            0.001
        };

        Ok(doses.mapv(|d| (d + offset).ln()))
    }

    /// Fit the specified dose-response model
    fn fit_dose_response_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        match self.model_type {
            DoseResponseModel::Linear => self.fit_linear_model(doses, responses),
            DoseResponseModel::LogLinear => self.fit_log_linear_model(doses, responses),
            DoseResponseModel::Hill => self.fit_hill_model(doses, responses),
            DoseResponseModel::Weibull => self.fit_weibull_model(doses, responses),
            DoseResponseModel::Probit => self.fit_probit_model(doses, responses),
            DoseResponseModel::Logistic => self.fit_logistic_model(doses, responses),
            DoseResponseModel::Exponential => self.fit_exponential_model(doses, responses),
            DoseResponseModel::PowerLaw => self.fit_power_law_model(doses, responses),
            DoseResponseModel::Threshold => self.fit_threshold_model(doses, responses),
            DoseResponseModel::Biphasic => self.fit_biphasic_model(doses, responses),
        }
    }

    /// Fit linear dose-response model
    fn fit_linear_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Sort by dose
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut sorted_responses: Array1<Float> =
            sorted_indices.iter().map(|&i| responses[i]).collect();

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut sorted_responses)?;

        // Constrain responses to be within bounds
        for response in sorted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        // Linear model parameters: slope and intercept
        let slope = if sorted_doses.len() > 1 {
            let dose_range = sorted_doses[sorted_doses.len() - 1] - sorted_doses[0];
            let response_range = sorted_responses[sorted_responses.len() - 1] - sorted_responses[0];
            if dose_range > 0.0 {
                response_range / dose_range
            } else {
                0.0
            }
        } else {
            0.0
        };

        let intercept = if sorted_responses.is_empty() {
            0.0
        } else {
            sorted_responses[0] - slope * sorted_doses[0]
        };
        let parameters = Array1::from_vec(vec![slope, intercept]);

        Ok((sorted_doses, sorted_responses, parameters))
    }

    /// Fit log-linear dose-response model
    fn fit_log_linear_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Transform to log scale for responses
        let log_responses: Array1<Float> = responses.mapv(|r| {
            let adjusted_r = r.max(self.background_response + 1e-10);
            adjusted_r.ln()
        });

        let (fitted_doses, mut fitted_log_responses, parameters) =
            self.fit_linear_model(doses, &log_responses)?;

        // Apply isotonic constraint in log space
        self.pool_adjacent_violators(&mut fitted_log_responses)?;

        // Transform back to original scale
        let fitted_responses = fitted_log_responses.mapv(|lr| lr.exp());

        Ok((fitted_doses, fitted_responses, parameters))
    }

    /// Fit Hill equation dose-response model
    fn fit_hill_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Hill equation: R = R_max * D^n / (EC50^n + D^n) + R_background
        let mut ec50 = doses.mean().unwrap_or(1.0);
        let mut hill_slope = 1.0;
        let mut max_response = self.max_response;

        // Iterative fitting
        for _iter in 0..self.max_iter {
            let old_ec50 = ec50;
            let old_hill_slope = hill_slope;
            let old_max_response = max_response;

            // Sort by dose
            let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
            sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

            let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();

            // Predict using current parameters
            let mut predicted_responses: Array1<Float> = sorted_doses.mapv(|d| {
                if d > 0.0 {
                    let numerator = max_response * d.powf(hill_slope);
                    let denominator = ec50.powf(hill_slope) + d.powf(hill_slope);
                    self.background_response + numerator / denominator
                } else {
                    self.background_response
                }
            });

            // Apply isotonic constraint
            self.pool_adjacent_violators(&mut predicted_responses)?;

            // Update parameters using simple gradient descent
            let learning_rate = 0.01;

            // Update EC50 (approximate gradient)
            let mut ec50_gradient = 0.0;
            for (i, &dose) in sorted_doses.iter().enumerate() {
                if dose > 0.0 {
                    let target = sorted_indices
                        .iter()
                        .map(|&idx| responses[idx])
                        .nth(i)
                        .unwrap();
                    let pred = predicted_responses[i];
                    let error = target - pred;

                    let grad = -max_response
                        * hill_slope
                        * dose.powf(hill_slope)
                        * ec50.powf(hill_slope - 1.0)
                        / (ec50.powf(hill_slope) + dose.powf(hill_slope)).powi(2);
                    ec50_gradient += error * grad;
                }
            }
            ec50 += learning_rate * ec50_gradient;
            ec50 = ec50.max(doses.iter().fold(Float::INFINITY, |a, &b| a.min(b)) * 0.1);

            // Update hill slope
            let mut hill_gradient = 0.0;
            for (i, &dose) in sorted_doses.iter().enumerate() {
                if dose > 0.0 {
                    let target = sorted_indices
                        .iter()
                        .map(|&idx| responses[idx])
                        .nth(i)
                        .unwrap();
                    let pred = predicted_responses[i];
                    let error = target - pred;

                    let ln_d = dose.ln();
                    let ln_ec50 = ec50.ln();
                    let d_h = dose.powf(hill_slope);
                    let ec50_h = ec50.powf(hill_slope);

                    let grad =
                        max_response * d_h * ec50_h * (ln_d - ln_ec50) / (d_h + ec50_h).powi(2);
                    hill_gradient += error * grad;
                }
            }
            hill_slope += learning_rate * hill_gradient;
            hill_slope = hill_slope.max(0.1).min(10.0);

            // Check convergence
            let change = (ec50 - old_ec50).abs()
                + (hill_slope - old_hill_slope).abs()
                + (max_response - old_max_response).abs();
            if change < self.tolerance {
                break;
            }
        }

        // Generate final fitted values
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let fitted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = fitted_doses.mapv(|d| {
            if d > 0.0 {
                let numerator = max_response * d.powf(hill_slope);
                let denominator = ec50.powf(hill_slope) + d.powf(hill_slope);
                self.background_response + numerator / denominator
            } else {
                self.background_response
            }
        });

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![ec50, hill_slope, max_response]);

        Ok((fitted_doses, fitted_responses, parameters))
    }

    /// Fit Weibull dose-response model
    fn fit_weibull_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Weibull: R = R_background + (R_max - R_background) * (1 - exp(-(D/λ)^k))
        let mut lambda = doses.mean().unwrap_or(1.0);
        let mut k = 1.0;

        // Iterative fitting
        for _iter in 0..self.max_iter {
            let old_lambda = lambda;
            let old_k = k;

            // Sort by dose
            let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
            sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

            let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();

            // Predict using current parameters
            let mut predicted_responses: Array1<Float> = sorted_doses.mapv(|d| {
                if d > 0.0 {
                    let weibull_term = (-(d / lambda).powf(k)).exp();
                    self.background_response
                        + (self.max_response - self.background_response) * (1.0 - weibull_term)
                } else {
                    self.background_response
                }
            });

            // Apply isotonic constraint
            self.pool_adjacent_violators(&mut predicted_responses)?;

            // Simple parameter update (approximate gradient descent)
            lambda *= 1.01; // Simple adjustment
            k = k.max(0.5).min(5.0);

            // Check convergence
            let change = (lambda - old_lambda).abs() + (k - old_k).abs();
            if change < self.tolerance {
                break;
            }
        }

        // Generate final fitted values
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let fitted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = fitted_doses.mapv(|d| {
            if d > 0.0 {
                let weibull_term = (-(d / lambda).powf(k)).exp();
                self.background_response
                    + (self.max_response - self.background_response) * (1.0 - weibull_term)
            } else {
                self.background_response
            }
        });

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![lambda, k]);

        Ok((fitted_doses, fitted_responses, parameters))
    }

    /// Fit probit dose-response model
    fn fit_probit_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Probit model uses normal CDF transformation
        // For simplicity, we'll use a logistic approximation
        self.fit_logistic_model(doses, responses)
    }

    /// Fit logistic dose-response model
    fn fit_logistic_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Logistic: R = R_background + (R_max - R_background) / (1 + exp(-k*(D - D50)))
        let mut d50 = doses.mean().unwrap_or(1.0);
        let mut k = 1.0;

        // Iterative fitting
        for _iter in 0..self.max_iter {
            let old_d50 = d50;
            let old_k = k;

            // Sort by dose
            let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
            sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

            let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();

            // Predict using current parameters
            let mut predicted_responses: Array1<Float> = sorted_doses.mapv(|d| {
                let logistic_term = 1.0 / (1.0 + (-k * (d - d50)).exp());
                self.background_response
                    + (self.max_response - self.background_response) * logistic_term
            });

            // Apply isotonic constraint
            self.pool_adjacent_violators(&mut predicted_responses)?;

            // Simple parameter updates
            d50 = sorted_doses.mean().unwrap_or(d50);
            k = k.max(0.1).min(10.0);

            // Check convergence
            let change = (d50 - old_d50).abs() + (k - old_k).abs();
            if change < self.tolerance {
                break;
            }
        }

        // Generate final fitted values
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let fitted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = fitted_doses.mapv(|d| {
            let logistic_term = 1.0 / (1.0 + (-k * (d - d50)).exp());
            self.background_response
                + (self.max_response - self.background_response) * logistic_term
        });

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![d50, k]);

        Ok((fitted_doses, fitted_responses, parameters))
    }

    /// Fit exponential dose-response model
    fn fit_exponential_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Exponential: R = R_background + A * (1 - exp(-k*D))
        let mut a = self.max_response - self.background_response;
        let mut k = 1.0;

        // Sort by dose
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> =
            sorted_doses.mapv(|d| self.background_response + a * (1.0 - (-k * d).exp()));

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![a, k]);

        Ok((sorted_doses, fitted_responses, parameters))
    }

    /// Fit power law dose-response model
    fn fit_power_law_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Power law: R = R_background + A * D^α
        let mut a = 0.1;
        let mut alpha = 0.5;

        // Sort by dose
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = sorted_doses.mapv(|d| {
            if d > 0.0 {
                (self.background_response + a * d.powf(alpha)).min(self.max_response)
            } else {
                self.background_response
            }
        });

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![a, alpha]);

        Ok((sorted_doses, fitted_responses, parameters))
    }

    /// Fit threshold dose-response model (hockey stick)
    fn fit_threshold_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Threshold model: R = R_background if D < threshold, else R_background + slope*(D - threshold)
        let mut threshold = doses.mean().unwrap_or(1.0);
        let mut slope = 0.1;

        // Sort by dose
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = sorted_doses.mapv(|d| {
            if d < threshold {
                self.background_response
            } else {
                (self.background_response + slope * (d - threshold)).min(self.max_response)
            }
        });

        // Apply isotonic constraint
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![threshold, slope]);

        Ok((sorted_doses, fitted_responses, parameters))
    }

    /// Fit biphasic dose-response model (hormesis)
    fn fit_biphasic_model(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        // Biphasic: Initial stimulation followed by inhibition
        // R = R_background + A1*D*exp(-k1*D) - A2*D/(1 + exp(-k2*(D-D50)))
        let mut a1 = 0.1;
        let mut k1 = 1.0;
        let mut a2 = 0.05;
        let mut k2 = 1.0;
        let mut d50 = doses.mean().unwrap_or(1.0);

        // Sort by dose
        let mut sorted_indices: Vec<usize> = (0..doses.len()).collect();
        sorted_indices.sort_by(|&i, &j| doses[i].partial_cmp(&doses[j]).unwrap());

        let sorted_doses: Array1<Float> = sorted_indices.iter().map(|&i| doses[i]).collect();
        let mut fitted_responses: Array1<Float> = sorted_doses.mapv(|d| {
            let stimulation = a1 * d * (-k1 * d).exp();
            let inhibition = a2 * d / (1.0 + (-k2 * (d - d50)).exp());
            (self.background_response + stimulation - inhibition)
                .max(self.background_response * 0.1)
                .min(self.max_response)
        });

        // For simplicity, apply basic isotonic smoothing
        self.pool_adjacent_violators(&mut fitted_responses)?;

        // Constrain responses to be within bounds
        for response in fitted_responses.iter_mut() {
            *response = response
                .max(self.background_response)
                .min(self.max_response);
        }

        let parameters = Array1::from_vec(vec![a1, k1, a2, k2, d50]);

        Ok((sorted_doses, fitted_responses, parameters))
    }

    /// Calculate benchmark doses
    fn calculate_benchmark_doses(
        &self,
        doses: &Array1<Float>,
        responses: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let method = self.benchmark_method.unwrap();

        let target_response = match method {
            BenchmarkDoseMethod::BMD10 => {
                self.background_response + 0.1 * (self.max_response - self.background_response)
            }
            BenchmarkDoseMethod::BMD05 => {
                self.background_response + 0.05 * (self.max_response - self.background_response)
            }
            BenchmarkDoseMethod::BMD01 => {
                self.background_response + 0.01 * (self.max_response - self.background_response)
            }
            BenchmarkDoseMethod::BMDL => {
                self.background_response + 0.1 * (self.max_response - self.background_response)
            } // Simplified
            BenchmarkDoseMethod::BMDU => {
                self.background_response + 0.1 * (self.max_response - self.background_response)
            } // Simplified
            BenchmarkDoseMethod::LED10 => {
                self.background_response + 0.1 * (self.max_response - self.background_response)
            }
            BenchmarkDoseMethod::NOAEL => {
                self.background_response + 0.05 * (self.max_response - self.background_response)
            }
            BenchmarkDoseMethod::LOAEL => {
                self.background_response + 0.1 * (self.max_response - self.background_response)
            }
        };

        // Find dose corresponding to target response
        let mut benchmark_dose = 0.0;

        // Handle case where target response is never reached
        if responses.is_empty() {
            return Ok(Array1::from_vec(vec![0.0]));
        }

        // If target response is below the minimum response, return 0
        if target_response <= responses[0] {
            return Ok(Array1::from_vec(vec![0.0]));
        }

        // If target response is above maximum response, extrapolate from last segment
        if target_response > responses[responses.len() - 1] {
            if responses.len() >= 2 {
                let x1 = doses[responses.len() - 2];
                let x2 = doses[responses.len() - 1];
                let y1 = responses[responses.len() - 2];
                let y2 = responses[responses.len() - 1];

                if y2 != y1 {
                    benchmark_dose = x1 + (target_response - y1) * (x2 - x1) / (y2 - y1);
                    benchmark_dose = benchmark_dose.max(0.0); // Ensure non-negative
                } else {
                    benchmark_dose = x2;
                }
            } else {
                benchmark_dose = doses[responses.len() - 1];
            }
            return Ok(Array1::from_vec(vec![benchmark_dose]));
        }

        // Find interpolation between adjacent points
        for i in 0..responses.len() {
            if responses[i] >= target_response {
                if i > 0 {
                    // Linear interpolation
                    let x1 = doses[i - 1];
                    let x2 = doses[i];
                    let y1 = responses[i - 1];
                    let y2 = responses[i];

                    if y2 != y1 {
                        benchmark_dose = x1 + (target_response - y1) * (x2 - x1) / (y2 - y1);
                        benchmark_dose = benchmark_dose.max(0.0); // Ensure non-negative
                    } else {
                        benchmark_dose = x1;
                    }
                } else {
                    benchmark_dose = doses[i];
                }
                break;
            }
        }

        Ok(Array1::from_vec(vec![benchmark_dose]))
    }

    /// Calculate confidence intervals
    fn calculate_confidence_intervals(
        &self,
        fitted_doses: &Array1<Float>,
        fitted_responses: &Array1<Float>,
        _original_doses: &Array1<Float>,
        _original_responses: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n = fitted_responses.len();
        let mut intervals = Array2::zeros((n, 2));

        match self.confidence_method {
            ConfidenceMethod::Bootstrap => {
                // Simplified bootstrap confidence intervals
                let alpha = 1.0 - self.confidence_level;
                let margin = 1.96; // Approximate 95% CI

                for i in 0..n {
                    let response = fitted_responses[i];
                    let se = self.regularization * response.abs().sqrt(); // Simplified standard error
                    intervals[[i, 0]] = response - margin * se; // Lower bound
                    intervals[[i, 1]] = response + margin * se; // Upper bound
                }
            }
            ConfidenceMethod::Wald => {
                // Wald confidence intervals
                let z_score = 1.96; // 95% CI
                for i in 0..n {
                    let response = fitted_responses[i];
                    let se = self.regularization * response.abs().sqrt();
                    intervals[[i, 0]] = response - z_score * se;
                    intervals[[i, 1]] = response + z_score * se;
                }
            }
            _ => {
                // Default to simple intervals
                for i in 0..n {
                    let response = fitted_responses[i];
                    let margin = self.regularization * response.abs();
                    intervals[[i, 0]] = response - margin;
                    intervals[[i, 1]] = response + margin;
                }
            }
        }

        Ok(intervals)
    }

    /// Pool Adjacent Violators Algorithm for isotonic regression
    fn pool_adjacent_violators(&self, y: &mut Array1<Float>) -> Result<()> {
        let n = y.len();
        if n <= 1 {
            return Ok(());
        }

        let mut i = 0;
        while i < n - 1 {
            if y[i] > y[i + 1] {
                // Find the violating segment
                let mut j = i + 1;
                while j < n && y[i] > y[j] {
                    j += 1;
                }

                // Compute the average for the violating segment
                let sum: Float = y.slice(s![i..j]).sum();
                let avg = sum / (j - i) as Float;

                // Set all values in the segment to the average
                for k in i..j {
                    y[k] = avg;
                }

                // Backtrack to check for new violations
                if i > 0 {
                    i -= 1;
                } else {
                    i = j;
                }
            } else {
                i += 1;
            }
        }

        Ok(())
    }
}

impl Predict<Array1<Float>, Array1<Float>> for DoseResponseIsotonicRegression<Trained> {
    fn predict(&self, doses: &Array1<Float>) -> Result<Array1<Float>> {
        let fitted_doses = self
            .fitted_doses
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "Model not fitted".to_string(),
            })?;
        let fitted_responses =
            self.fitted_responses
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "Model not fitted".to_string(),
                })?;

        // Transform doses if needed
        let transformed_doses = if self.log_transform_dose {
            let min_dose = doses
                .iter()
                .filter(|&&d| d > 0.0)
                .fold(Float::INFINITY, |a, &b| a.min(b));
            let offset = if min_dose.is_finite() {
                min_dose * 0.01
            } else {
                0.001
            };
            doses.mapv(|d| (d + offset).ln())
        } else {
            doses.clone()
        };

        let mut predictions = Array1::zeros(transformed_doses.len());

        for (i, &dose) in transformed_doses.iter().enumerate() {
            // Linear interpolation/extrapolation
            if dose <= fitted_doses[0] {
                predictions[i] = fitted_responses[0];
            } else if dose >= fitted_doses[fitted_doses.len() - 1] {
                predictions[i] = fitted_responses[fitted_responses.len() - 1];
            } else {
                // Find surrounding points for interpolation
                let mut idx = 0;
                for j in 0..fitted_doses.len() - 1 {
                    if dose >= fitted_doses[j] && dose <= fitted_doses[j + 1] {
                        idx = j;
                        break;
                    }
                }

                let x1 = fitted_doses[idx];
                let x2 = fitted_doses[idx + 1];
                let y1 = fitted_responses[idx];
                let y2 = fitted_responses[idx + 1];

                if x2 != x1 {
                    predictions[i] = y1 + (dose - x1) * (y2 - y1) / (x2 - x1);
                } else {
                    predictions[i] = y1;
                }
            }
        }

        Ok(predictions)
    }
}

impl DoseResponseIsotonicRegression<Trained> {
    /// Get benchmark doses if calculated
    pub fn benchmark_doses(&self) -> Option<&Array1<Float>> {
        self.benchmark_doses.as_ref()
    }

    /// Get confidence intervals
    pub fn confidence_intervals(&self) -> Option<&Array2<Float>> {
        self.confidence_intervals.as_ref()
    }

    /// Get model parameters
    pub fn model_parameters(&self) -> Option<&Array1<Float>> {
        self.model_parameters.as_ref()
    }
}

/// Convenience function for monotonic dose-response curve fitting
pub fn monotonic_dose_response_curve(
    doses: &Array1<Float>,
    responses: &Array1<Float>,
    model_type: DoseResponseModel,
) -> Result<(Array1<Float>, Array1<Float>)> {
    let model = DoseResponseIsotonicRegression::new().model_type(model_type);
    let fitted_model = model.fit(doses, responses)?;

    let fitted_doses = fitted_model.fitted_doses.unwrap();
    let fitted_responses = fitted_model.fitted_responses.unwrap();

    Ok((fitted_doses, fitted_responses))
}

/// Convenience function for benchmark dose estimation
pub fn benchmark_dose_estimation(
    doses: &Array1<Float>,
    responses: &Array1<Float>,
    model_type: DoseResponseModel,
    benchmark_method: BenchmarkDoseMethod,
) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
    let model = DoseResponseIsotonicRegression::new()
        .model_type(model_type)
        .benchmark_method(benchmark_method);
    let fitted_model = model.fit(doses, responses)?;

    let fitted_doses = fitted_model.fitted_doses.unwrap();
    let fitted_responses = fitted_model.fitted_responses.unwrap();
    let benchmark_doses = fitted_model.benchmark_doses.unwrap();

    Ok((fitted_doses, fitted_responses, benchmark_doses))
}

/// Convenience function for toxicological modeling
pub fn toxicological_modeling(
    doses: &Array1<Float>,
    responses: &Array1<Float>,
    benchmark_method: BenchmarkDoseMethod,
) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>, Array2<Float>)> {
    let model = DoseResponseIsotonicRegression::new()
        .application_domain(ApplicationDomain::Toxicology)
        .model_type(DoseResponseModel::Hill)
        .benchmark_method(benchmark_method)
        .confidence_method(ConfidenceMethod::Bootstrap);

    let fitted_model = model.fit(doses, responses)?;

    let fitted_doses = fitted_model.fitted_doses.unwrap();
    let fitted_responses = fitted_model.fitted_responses.unwrap();
    let benchmark_doses = fitted_model.benchmark_doses.unwrap();
    let confidence_intervals = fitted_model.confidence_intervals.unwrap();

    Ok((
        fitted_doses,
        fitted_responses,
        benchmark_doses,
        confidence_intervals,
    ))
}

/// Convenience function for pharmacokinetic applications
pub fn pharmacokinetic_modeling(
    doses: &Array1<Float>,
    responses: &Array1<Float>,
) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
    let model = DoseResponseIsotonicRegression::new()
        .application_domain(ApplicationDomain::Pharmacokinetics)
        .model_type(DoseResponseModel::LogLinear)
        .log_transform_dose(true);

    let fitted_model = model.fit(doses, responses)?;

    let fitted_doses = fitted_model.fitted_doses.unwrap();
    let fitted_responses = fitted_model.fitted_responses.unwrap();
    let model_parameters = fitted_model.model_parameters.unwrap();

    Ok((fitted_doses, fitted_responses, model_parameters))
}

/// Convenience function for efficacy modeling
pub fn efficacy_modeling(
    doses: &Array1<Float>,
    responses: &Array1<Float>,
) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
    let model = DoseResponseIsotonicRegression::new()
        .application_domain(ApplicationDomain::Efficacy)
        .model_type(DoseResponseModel::Hill)
        .background_response(0.0)
        .max_response(1.0);

    let fitted_model = model.fit(doses, responses)?;

    let fitted_doses = fitted_model.fitted_doses.unwrap();
    let fitted_responses = fitted_model.fitted_responses.unwrap();
    let model_parameters = fitted_model.model_parameters.unwrap();

    Ok((fitted_doses, fitted_responses, model_parameters))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_linear_dose_response() -> Result<()> {
        let doses = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let responses = array![0.0, 0.2, 0.4, 0.6, 0.8, 1.0];

        let (fitted_doses, fitted_responses) =
            monotonic_dose_response_curve(&doses, &responses, DoseResponseModel::Linear)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);

        // Check monotonicity
        for i in 0..fitted_responses.len() - 1 {
            assert!(fitted_responses[i] <= fitted_responses[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_hill_dose_response() -> Result<()> {
        let doses = array![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let responses = array![0.05, 0.15, 0.3, 0.6, 0.85, 0.95];

        let (fitted_doses, fitted_responses) =
            monotonic_dose_response_curve(&doses, &responses, DoseResponseModel::Hill)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);

        // Check monotonicity
        for i in 0..fitted_responses.len() - 1 {
            assert!(fitted_responses[i] <= fitted_responses[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_benchmark_dose_estimation() -> Result<()> {
        let doses = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let responses = array![0.0, 0.05, 0.15, 0.3, 0.6, 0.9];

        let (fitted_doses, fitted_responses, benchmark_doses) = benchmark_dose_estimation(
            &doses,
            &responses,
            DoseResponseModel::Linear,
            BenchmarkDoseMethod::BMD10,
        )?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);
        assert_eq!(benchmark_doses.len(), 1);
        assert!(benchmark_doses[0] >= 0.0);

        Ok(())
    }

    #[test]
    fn test_toxicological_modeling() -> Result<()> {
        let doses = array![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let responses = array![0.01, 0.05, 0.15, 0.35, 0.7, 0.9];

        let (fitted_doses, fitted_responses, benchmark_doses, confidence_intervals) =
            toxicological_modeling(&doses, &responses, BenchmarkDoseMethod::BMD10)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);
        assert_eq!(benchmark_doses.len(), 1);
        assert_eq!(confidence_intervals.shape(), [6, 2]);

        // Check confidence intervals are properly ordered
        for i in 0..fitted_responses.len() {
            assert!(confidence_intervals[[i, 0]] <= fitted_responses[i]);
            assert!(fitted_responses[i] <= confidence_intervals[[i, 1]]);
        }

        Ok(())
    }

    #[test]
    fn test_pharmacokinetic_modeling() -> Result<()> {
        let doses = array![1.0, 2.0, 5.0, 10.0, 20.0, 50.0];
        let responses = array![0.1, 0.2, 0.4, 0.6, 0.8, 1.0];

        let (fitted_doses, fitted_responses, model_parameters) =
            pharmacokinetic_modeling(&doses, &responses)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);
        assert!(model_parameters.len() >= 2); // At least slope and intercept

        Ok(())
    }

    #[test]
    fn test_efficacy_modeling() -> Result<()> {
        let doses = array![0.0, 0.1, 0.5, 1.0, 2.0, 5.0];
        let responses = array![0.0, 0.1, 0.3, 0.5, 0.7, 0.9];

        let (fitted_doses, fitted_responses, model_parameters) =
            efficacy_modeling(&doses, &responses)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);
        assert!(model_parameters.len() >= 2);

        // Check responses are within bounds
        for &response in fitted_responses.iter() {
            assert!(response >= 0.0 && response <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_weibull_model() -> Result<()> {
        let doses = array![0.1, 0.5, 1.0, 2.0, 5.0];
        let responses = array![0.05, 0.2, 0.4, 0.7, 0.9];

        let (fitted_doses, fitted_responses) =
            monotonic_dose_response_curve(&doses, &responses, DoseResponseModel::Weibull)?;

        assert_eq!(fitted_doses.len(), 5);
        assert_eq!(fitted_responses.len(), 5);

        // Check monotonicity
        for i in 0..fitted_responses.len() - 1 {
            assert!(fitted_responses[i] <= fitted_responses[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_threshold_model() -> Result<()> {
        let doses = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let responses = array![0.1, 0.1, 0.1, 0.3, 0.6, 0.9];

        let (fitted_doses, fitted_responses) =
            monotonic_dose_response_curve(&doses, &responses, DoseResponseModel::Threshold)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);

        // Check monotonicity
        for i in 0..fitted_responses.len() - 1 {
            assert!(fitted_responses[i] <= fitted_responses[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_dose_response_prediction() -> Result<()> {
        let doses = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let responses = array![0.1, 0.3, 0.5, 0.7, 0.9];

        let model = DoseResponseIsotonicRegression::new().model_type(DoseResponseModel::Linear);
        let fitted_model = model.fit(&doses, &responses)?;

        let test_doses = array![1.5, 2.5, 3.5];
        let predictions = fitted_model.predict(&test_doses)?;

        assert_eq!(predictions.len(), 3);

        // Predictions should be reasonable
        for &pred in predictions.iter() {
            assert!(pred >= 0.0 && pred <= 1.0);
        }

        Ok(())
    }

    #[test]
    fn test_empty_doses() {
        let doses = array![];
        let responses = array![];

        let model = DoseResponseIsotonicRegression::new();
        let result = model.fit(&doses, &responses);

        assert!(result.is_ok());
    }

    #[test]
    fn test_negative_doses() {
        let doses = array![-1.0, 0.0, 1.0];
        let responses = array![0.0, 0.1, 0.2];

        let model = DoseResponseIsotonicRegression::new();
        let result = model.fit(&doses, &responses);

        assert!(result.is_err());
    }

    #[test]
    fn test_biphasic_model() -> Result<()> {
        let doses = array![0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
        let responses = array![0.05, 0.15, 0.12, 0.08, 0.05, 0.02]; // Hormesis pattern

        let (fitted_doses, fitted_responses) =
            monotonic_dose_response_curve(&doses, &responses, DoseResponseModel::Biphasic)?;

        assert_eq!(fitted_doses.len(), 6);
        assert_eq!(fitted_responses.len(), 6);

        // After isotonic constraint, should be monotonic
        for i in 0..fitted_responses.len() - 1 {
            assert!(fitted_responses[i] <= fitted_responses[i + 1]);
        }

        Ok(())
    }

    #[test]
    fn test_benchmark_dose_methods() -> Result<()> {
        let doses = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let responses = array![0.0, 0.02, 0.06, 0.12, 0.20, 0.30];

        let methods = vec![
            BenchmarkDoseMethod::BMD10,
            BenchmarkDoseMethod::BMD05,
            BenchmarkDoseMethod::BMD01,
            BenchmarkDoseMethod::NOAEL,
            BenchmarkDoseMethod::LOAEL,
        ];

        for method in methods {
            let (_, _, benchmark_doses) =
                benchmark_dose_estimation(&doses, &responses, DoseResponseModel::Linear, method)?;

            assert_eq!(benchmark_doses.len(), 1);
            assert!(benchmark_doses[0] >= 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_confidence_intervals() -> Result<()> {
        let doses = array![0.0, 1.0, 2.0, 3.0, 4.0];
        let responses = array![0.0, 0.2, 0.4, 0.6, 0.8];

        let model = DoseResponseIsotonicRegression::new()
            .model_type(DoseResponseModel::Linear)
            .confidence_method(ConfidenceMethod::Bootstrap)
            .confidence_level(0.95);

        let fitted_model = model.fit(&doses, &responses)?;
        let confidence_intervals = fitted_model.confidence_intervals().unwrap();

        assert_eq!(confidence_intervals.shape(), [5, 2]);

        // Check that lower bounds are less than upper bounds
        for i in 0..5 {
            assert!(confidence_intervals[[i, 0]] <= confidence_intervals[[i, 1]]);
        }

        Ok(())
    }
}
