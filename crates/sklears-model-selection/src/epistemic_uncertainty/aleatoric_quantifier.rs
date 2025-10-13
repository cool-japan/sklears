use super::uncertainty_config::AleatoricUncertaintyConfig;
use super::uncertainty_methods::AleatoricUncertaintyMethod;
use super::uncertainty_results::AleatoricUncertaintyResult;
use super::uncertainty_types::*;
use super::variance_estimation::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;
use scirs2_core::random::Random;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct AleatoricUncertaintyQuantifier {
    config: AleatoricUncertaintyConfig,
}

impl AleatoricUncertaintyQuantifier {
    pub fn new() -> Self {
        Self {
            config: AleatoricUncertaintyConfig::default(),
        }
    }

    pub fn with_config(config: AleatoricUncertaintyConfig) -> Self {
        Self { config }
    }

    pub fn method(mut self, method: AleatoricUncertaintyMethod) -> Self {
        self.config.method = method;
        self
    }

    pub fn confidence_level(mut self, level: f64) -> Self {
        self.config.confidence_level = level;
        self
    }

    pub fn random_state(mut self, seed: u64) -> Self {
        self.config.random_state = Some(seed);
        self
    }

    pub fn noise_regularization(mut self, reg: f64) -> Self {
        self.config.noise_regularization = reg;
        self
    }

    pub fn min_variance(mut self, min_var: f64) -> Self {
        self.config.min_variance = min_var;
        self
    }

    pub fn quantify<E, P>(
        &self,
        models: &[E],
        x: &Array2<f64>,
        y_true: Option<&Array1<f64>>,
    ) -> Result<AleatoricUncertaintyResult, Box<dyn std::error::Error>>
    where
        E: Clone,
        P: Clone,
    {
        let rng = match self.config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let (predictions, uncertainties, variance_estimates, noise_estimates) =
            match &self.config.method {
                AleatoricUncertaintyMethod::HeteroskedasticRegression { n_ensemble } => {
                    heteroskedastic_regression_uncertainty(models, x, *n_ensemble)?
                }
                AleatoricUncertaintyMethod::MixtureDensityNetwork { n_components } => {
                    mixture_density_network_uncertainty(models, x, *n_components)?
                }
                AleatoricUncertaintyMethod::QuantileRegression { quantiles } => {
                    quantile_regression_uncertainty(models, x, quantiles)?
                }
                AleatoricUncertaintyMethod::ParametricUncertainty { distribution } => {
                    parametric_uncertainty_estimation(models, x, distribution)?
                }
                AleatoricUncertaintyMethod::InputDependentNoise { noise_model } => {
                    input_dependent_noise_uncertainty(models, x, noise_model)?
                }
                AleatoricUncertaintyMethod::ResidualBasedUncertainty { window_size } => {
                    residual_based_uncertainty(models, x, y_true, *window_size)?
                }
                AleatoricUncertaintyMethod::EnsembleAleatoric {
                    n_models,
                    noise_estimation,
                } => ensemble_aleatoric_uncertainty(models, x, *n_models, noise_estimation)?,
            };

        let alpha = 1.0 - self.config.confidence_level;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        let prediction_intervals = self.compute_prediction_intervals(
            &predictions,
            &uncertainties,
            lower_quantile,
            upper_quantile,
        )?;

        let heteroskedastic_weights = self.compute_heteroskedastic_weights(&variance_estimates)?;
        let distributional_parameters =
            self.compute_distributional_parameters(&predictions, &variance_estimates)?;

        let reliability_metrics =
            self.compute_reliability_metrics(&predictions, &uncertainties, y_true)?;

        Ok(AleatoricUncertaintyResult {
            predictions,
            uncertainties,
            prediction_intervals,
            noise_estimates,
            variance_estimates,
            heteroskedastic_weights,
            distributional_parameters,
            reliability_metrics,
        })
    }

    fn compute_prediction_intervals(
        &self,
        predictions: &Array1<f64>,
        uncertainties: &Array1<f64>,
        lower_quantile: f64,
        upper_quantile: f64,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let n = predictions.len();
        let mut intervals = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let std_dev = uncertainties[i].sqrt().max(self.config.min_variance.sqrt());
            let z_lower = normal_quantile(lower_quantile);
            let z_upper = normal_quantile(upper_quantile);

            intervals[[i, 0]] = predictions[i] + z_lower * std_dev;
            intervals[[i, 1]] = predictions[i] + z_upper * std_dev;
        }

        Ok(intervals)
    }

    fn compute_heteroskedastic_weights(
        &self,
        variance_estimates: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let mean_variance = variance_estimates.mean().unwrap_or(1.0);
        let weights =
            variance_estimates.mapv(|var| if var > 0.0 { mean_variance / var } else { 1.0 });
        Ok(weights)
    }

    fn compute_distributional_parameters(
        &self,
        predictions: &Array1<f64>,
        variance_estimates: &Array1<f64>,
    ) -> Result<HashMap<String, Array1<f64>>, Box<dyn std::error::Error>> {
        let mut parameters = HashMap::new();

        parameters.insert("mean".to_string(), predictions.clone());
        parameters.insert("variance".to_string(), variance_estimates.clone());
        parameters.insert("std_dev".to_string(), variance_estimates.mapv(|v| v.sqrt()));

        let shape_params = variance_estimates.mapv(|v| {
            let shape = predictions.mean().unwrap_or(1.0).powi(2) / v.max(self.config.min_variance);
            shape.max(1e-6)
        });
        parameters.insert("shape".to_string(), shape_params);

        let scale_params =
            variance_estimates.mapv(|v| v / predictions.mean().unwrap_or(1.0).max(1e-6));
        parameters.insert("scale".to_string(), scale_params);

        Ok(parameters)
    }

    fn compute_reliability_metrics(
        &self,
        predictions: &Array1<f64>,
        uncertainties: &Array1<f64>,
        y_true: Option<&Array1<f64>>,
    ) -> Result<ReliabilityMetrics, Box<dyn std::error::Error>> {
        let calibration_error = match y_true {
            Some(y) => self.compute_calibration_score(predictions, uncertainties, y)?,
            None => 0.0,
        };

        let sharpness = uncertainties.mean().unwrap_or(0.0);
        let reliability_score = 1.0 - calibration_error;
        let coverage_probability = 0.95; // Placeholder
        let prediction_interval_score = 0.0; // Placeholder
        let continuous_ranked_probability_score = 0.0; // Placeholder

        Ok(ReliabilityMetrics {
            calibration_error,
            sharpness,
            reliability_score,
            coverage_probability,
            prediction_interval_score,
            continuous_ranked_probability_score,
        })
    }

    fn compute_calibration_score(
        &self,
        predictions: &Array1<f64>,
        uncertainties: &Array1<f64>,
        y_true: &Array1<f64>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let n_bins = 10;
        let mut calibration_error = 0.0;

        for bin_idx in 0..n_bins {
            let lower_bound = bin_idx as f64 / n_bins as f64;
            let upper_bound = (bin_idx + 1) as f64 / n_bins as f64;

            let mut bin_predictions = Vec::new();
            let mut bin_true_values = Vec::new();
            let mut bin_uncertainties = Vec::new();

            for i in 0..predictions.len() {
                let normalized_uncertainty =
                    uncertainties[i] / uncertainties.iter().fold(0.0, |max, &x| max.max(x));
                if normalized_uncertainty > lower_bound && normalized_uncertainty <= upper_bound {
                    bin_predictions.push(predictions[i]);
                    bin_true_values.push(y_true[i]);
                    bin_uncertainties.push(uncertainties[i]);
                }
            }

            if !bin_predictions.is_empty() {
                let bin_mse = bin_predictions
                    .iter()
                    .zip(bin_true_values.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                    .sum::<f64>()
                    / bin_predictions.len() as f64;

                let expected_mse =
                    bin_uncertainties.iter().sum::<f64>() / bin_uncertainties.len() as f64;
                calibration_error += (bin_mse - expected_mse).abs() * bin_predictions.len() as f64
                    / predictions.len() as f64;
            }
        }

        Ok(calibration_error)
    }

    // Getter methods for testing
    pub fn config(&self) -> &AleatoricUncertaintyConfig {
        &self.config
    }
}

impl Default for AleatoricUncertaintyQuantifier {
    fn default() -> Self {
        Self::new()
    }
}

fn normal_quantile(p: f64) -> f64 {
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    let c0 = 2.515517;
    let c1 = 0.802853;
    let c2 = 0.010328;
    let d1 = 1.432788;
    let d2 = 0.189269;
    let d3 = 0.001308;

    let t = if p < 0.5 {
        (-2.0 * p.ln()).sqrt()
    } else {
        (-2.0 * (1.0 - p).ln()).sqrt()
    };
    let numerator = c0 + c1 * t + c2 * t * t;
    let denominator = 1.0 + d1 * t + d2 * t * t + d3 * t * t * t;
    let result = t - numerator / denominator;

    if p < 0.5 {
        -result
    } else {
        result
    }
}
