use super::aleatoric_quantifier::AleatoricUncertaintyQuantifier;
use super::epistemic_quantifier::EpistemicUncertaintyQuantifier;
use super::uncertainty_config::*;
use super::uncertainty_decomposition::*;
use super::uncertainty_results::*;
use super::uncertainty_types::*;
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::numeric::Float;

#[derive(Debug, Clone)]
pub struct UncertaintyQuantifier {
    config: UncertaintyQuantificationConfig,
}

impl UncertaintyQuantifier {
    pub fn new() -> Self {
        Self {
            config: UncertaintyQuantificationConfig::default(),
        }
    }

    pub fn with_config(config: UncertaintyQuantificationConfig) -> Self {
        Self { config }
    }

    pub fn epistemic_config(mut self, config: EpistemicUncertaintyConfig) -> Self {
        self.config.epistemic_config = config;
        self
    }

    pub fn aleatoric_config(mut self, config: AleatoricUncertaintyConfig) -> Self {
        self.config.aleatoric_config = config;
        self
    }

    pub fn decomposition_method(mut self, method: UncertaintyDecompositionMethod) -> Self {
        self.config.decomposition_method = method;
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

    pub fn quantify<E, P>(
        &self,
        models: &[E],
        x: &Array2<f64>,
        y_true: Option<&Array1<f64>>,
    ) -> Result<UncertaintyQuantificationResult, Box<dyn std::error::Error>>
    where
        E: Clone,
        P: Clone,
    {
        let epistemic_quantifier =
            EpistemicUncertaintyQuantifier::with_config(self.config.epistemic_config.clone());
        let aleatoric_quantifier =
            AleatoricUncertaintyQuantifier::with_config(self.config.aleatoric_config.clone());

        let epistemic_result = epistemic_quantifier.quantify::<E, P>(models, x, y_true)?;
        let aleatoric_result = aleatoric_quantifier.quantify::<E, P>(models, x, y_true)?;

        let total_uncertainty =
            epistemic_result.uncertainties.clone() + &aleatoric_result.uncertainties;

        let uncertainty_decomposition = self.decompose_uncertainty(
            &epistemic_result.uncertainties,
            &aleatoric_result.uncertainties,
        )?;

        let alpha = 1.0 - self.config.confidence_level;
        let lower_quantile = alpha / 2.0;
        let upper_quantile = 1.0 - alpha / 2.0;

        let prediction_intervals = self.compute_combined_prediction_intervals(
            &epistemic_result.predictions,
            &total_uncertainty,
            lower_quantile,
            upper_quantile,
        )?;

        let calibration_score = (epistemic_result.calibration_score
            + aleatoric_result.reliability_metrics.calibration_error)
            / 2.0;

        let reliability_metrics = ReliabilityMetrics {
            calibration_error: calibration_score,
            sharpness: total_uncertainty.mean().unwrap_or(0.0),
            reliability_score: 1.0 - calibration_score,
            coverage_probability: 0.95,
            prediction_interval_score: 0.0,
            continuous_ranked_probability_score: 0.0,
        };

        Ok(UncertaintyQuantificationResult {
            predictions: epistemic_result.predictions.clone(),
            total_uncertainty,
            epistemic_uncertainty: epistemic_result.uncertainties.clone(),
            aleatoric_uncertainty: aleatoric_result.uncertainties.clone(),
            prediction_intervals,
            uncertainty_decomposition,
            calibration_score,
            reliability_metrics,
            epistemic_result,
            aleatoric_result,
        })
    }

    fn decompose_uncertainty(
        &self,
        epistemic_uncertainty: &Array1<f64>,
        aleatoric_uncertainty: &Array1<f64>,
    ) -> Result<UncertaintyDecomposition, Box<dyn std::error::Error>> {
        let total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty;
        let n = total_uncertainty.len();

        let entropy_components = std::collections::HashMap::new();
        let mutual_information = epistemic_uncertainty.mean().unwrap_or(0.0);
        let explained_variance_ratio = epistemic_uncertainty.sum() / total_uncertainty.sum();
        let uncertainty_ratios = epistemic_uncertainty / &total_uncertainty;

        Ok(UncertaintyDecomposition {
            total_uncertainty,
            epistemic_uncertainty: epistemic_uncertainty.clone(),
            aleatoric_uncertainty: aleatoric_uncertainty.clone(),
            decomposition_method: format!("{:?}", self.config.decomposition_method),
            entropy_components,
            mutual_information,
            explained_variance_ratio,
            uncertainty_ratios,
        })
    }

    fn compute_combined_prediction_intervals(
        &self,
        predictions: &Array1<f64>,
        total_uncertainty: &Array1<f64>,
        lower_quantile: f64,
        upper_quantile: f64,
    ) -> Result<Array2<f64>, Box<dyn std::error::Error>> {
        let n = predictions.len();
        let mut intervals = Array2::<f64>::zeros((n, 2));

        for i in 0..n {
            let std_dev = total_uncertainty[i].sqrt();
            let z_lower = normal_quantile(lower_quantile);
            let z_upper = normal_quantile(upper_quantile);

            intervals[[i, 0]] = predictions[i] + z_lower * std_dev;
            intervals[[i, 1]] = predictions[i] + z_upper * std_dev;
        }

        Ok(intervals)
    }

    // Getter methods for testing
    pub fn config(&self) -> &UncertaintyQuantificationConfig {
        &self.config
    }
}

impl Default for UncertaintyQuantifier {
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
