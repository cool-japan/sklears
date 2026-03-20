use super::bayesian_methods::*;
use super::calibration::CalibrationMethod;
use super::ensemble_methods::*;
use super::monte_carlo_methods::*;
use super::uncertainty_config::EpistemicUncertaintyConfig;
use super::uncertainty_methods::EpistemicUncertaintyMethod;
use super::uncertainty_results::EpistemicUncertaintyResult;
use super::uncertainty_types::*;
use scirs2_core::ndarray::{Array1, Array2};
// use scirs2_core::numeric::Float;
use scirs2_core::random::Random;

#[derive(Debug, Clone)]
pub struct EpistemicUncertaintyQuantifier {
    config: EpistemicUncertaintyConfig,
}

impl EpistemicUncertaintyQuantifier {
    pub fn new() -> Self {
        Self {
            config: EpistemicUncertaintyConfig::default(),
        }
    }

    pub fn with_config(config: EpistemicUncertaintyConfig) -> Self {
        Self { config }
    }

    pub fn method(mut self, method: EpistemicUncertaintyMethod) -> Self {
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

    pub fn calibration_method(mut self, method: CalibrationMethod) -> Self {
        self.config.calibration_method = method;
        self
    }

    pub fn temperature_scaling(mut self, enable: bool) -> Self {
        self.config.temperature_scaling = enable;
        self
    }

    pub fn quantify<E, P>(
        &self,
        models: &[E],
        x: &Array2<f64>,
        y_true: Option<&Array1<f64>>,
    ) -> Result<EpistemicUncertaintyResult, Box<dyn std::error::Error>>
    where
        E: Clone,
        P: Clone,
    {
        let mut rng = match self.config.random_state {
            Some(seed) => Random::seed(seed),
            None => Random::seed(42),
        };

        let (predictions, uncertainties) = match &self.config.method {
            EpistemicUncertaintyMethod::MonteCarloDropout {
                dropout_rate,
                n_samples,
            } => monte_carlo_dropout_uncertainty(models, x, *dropout_rate, *n_samples, &mut rng)?,
            EpistemicUncertaintyMethod::DeepEnsembles { n_models } => {
                deep_ensemble_uncertainty(models, x, *n_models)?
            }
            EpistemicUncertaintyMethod::BayesianNeuralNetwork { n_samples } => {
                bayesian_neural_network_uncertainty(models, x, *n_samples, &mut rng)?
            }
            EpistemicUncertaintyMethod::Bootstrap {
                n_bootstrap,
                sample_ratio,
            } => bootstrap_uncertainty(models, x, *n_bootstrap, *sample_ratio, &mut rng)?,
            EpistemicUncertaintyMethod::GaussianProcess { kernel_type } => {
                gaussian_process_uncertainty(models, x, kernel_type)?
            }
            EpistemicUncertaintyMethod::VariationalInference { n_samples } => {
                variational_inference_uncertainty(models, x, *n_samples, &mut rng)?
            }
            EpistemicUncertaintyMethod::LaplaceApproximation { hessian_method } => {
                laplace_approximation_uncertainty(models, x, hessian_method)?
            }
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

        let entropy = self.compute_entropy(&predictions)?;
        let mutual_information = self.compute_mutual_information(&predictions)?;

        let epistemic_uncertainty_components = UncertaintyComponents {
            model_uncertainty: uncertainties.clone(),
            data_uncertainty: Array1::zeros(uncertainties.len()),
            parameter_uncertainty: uncertainties.clone(),
            structural_uncertainty: Array1::zeros(uncertainties.len()),
            approximation_uncertainty: Array1::zeros(uncertainties.len()),
        };

        let calibration_score = match y_true {
            Some(y) => self.compute_calibration_score(&predictions, &uncertainties, y)?,
            None => 0.0,
        };

        let reliability_metrics =
            self.compute_reliability_metrics(&predictions, &uncertainties, y_true)?;

        Ok(EpistemicUncertaintyResult {
            predictions,
            uncertainties,
            prediction_intervals,
            calibration_score,
            entropy,
            mutual_information,
            epistemic_uncertainty_components,
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
            let std_dev = uncertainties[i].sqrt();
            let z_lower = normal_quantile(lower_quantile);
            let z_upper = normal_quantile(upper_quantile);

            intervals[[i, 0]] = predictions[i] + z_lower * std_dev;
            intervals[[i, 1]] = predictions[i] + z_upper * std_dev;
        }

        Ok(intervals)
    }

    fn compute_entropy(
        &self,
        predictions: &Array1<f64>,
    ) -> Result<Array1<f64>, Box<dyn std::error::Error>> {
        let entropy = predictions.mapv(|p| {
            if p > 0.0 && p < 1.0 {
                -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
            } else {
                0.0
            }
        });
        Ok(entropy)
    }

    fn compute_mutual_information(
        &self,
        predictions: &Array1<f64>,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let mean_entropy = predictions
            .iter()
            .map(|&p| {
                if p > 0.0 && p < 1.0 {
                    -p * p.ln() - (1.0 - p) * (1.0 - p).ln()
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / predictions.len() as f64;

        let mean_prediction = predictions.mean().unwrap_or(0.0);
        let entropy_of_mean = if mean_prediction > 0.0 && mean_prediction < 1.0 {
            -mean_prediction * mean_prediction.ln()
                - (1.0 - mean_prediction) * (1.0 - mean_prediction).ln()
        } else {
            0.0
        };

        Ok(entropy_of_mean - mean_entropy)
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

            for i in 0..predictions.len() {
                let confidence = 1.0 - uncertainties[i];
                if confidence > lower_bound && confidence <= upper_bound {
                    bin_predictions.push(predictions[i]);
                    bin_true_values.push(y_true[i]);
                }
            }

            if !bin_predictions.is_empty() {
                let bin_accuracy = bin_predictions
                    .iter()
                    .zip(bin_true_values.iter())
                    .map(|(&pred, &true_val)| {
                        if (pred - true_val).abs() < 0.1 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<f64>()
                    / bin_predictions.len() as f64;

                let bin_confidence = (lower_bound + upper_bound) / 2.0;
                calibration_error += (bin_accuracy - bin_confidence).abs()
                    * bin_predictions.len() as f64
                    / predictions.len() as f64;
            }
        }

        Ok(calibration_error)
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

    // Getter methods for testing
    pub fn config(&self) -> &EpistemicUncertaintyConfig {
        &self.config
    }
}

impl Default for EpistemicUncertaintyQuantifier {
    fn default() -> Self {
        Self::new()
    }
}

fn normal_quantile(p: f64) -> f64 {
    // Simplified normal quantile approximation
    if p <= 0.0 {
        return f64::NEG_INFINITY;
    }
    if p >= 1.0 {
        return f64::INFINITY;
    }
    if p == 0.5 {
        return 0.0;
    }

    // Box-Muller approximation for normal quantile
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
