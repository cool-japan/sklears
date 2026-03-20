use super::uncertainty_types::*;
use scirs2_core::ndarray::{Array1, Array2};
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct EpistemicUncertaintyResult {
    pub predictions: Array1<f64>,
    pub uncertainties: Array1<f64>,
    pub prediction_intervals: Array2<f64>,
    pub calibration_score: f64,
    pub entropy: Array1<f64>,
    pub mutual_information: f64,
    pub epistemic_uncertainty_components: UncertaintyComponents,
    pub reliability_metrics: ReliabilityMetrics,
}

#[derive(Debug, Clone)]
pub struct AleatoricUncertaintyResult {
    pub predictions: Array1<f64>,
    pub uncertainties: Array1<f64>,
    pub prediction_intervals: Array2<f64>,
    pub noise_estimates: Array1<f64>,
    pub variance_estimates: Array1<f64>,
    pub heteroskedastic_weights: Array1<f64>,
    pub distributional_parameters: HashMap<String, Array1<f64>>,
    pub reliability_metrics: ReliabilityMetrics,
}

#[derive(Debug, Clone)]
pub struct UncertaintyQuantificationResult {
    pub predictions: Array1<f64>,
    pub total_uncertainty: Array1<f64>,
    pub epistemic_uncertainty: Array1<f64>,
    pub aleatoric_uncertainty: Array1<f64>,
    pub prediction_intervals: Array2<f64>,
    pub uncertainty_decomposition: UncertaintyDecomposition,
    pub calibration_score: f64,
    pub reliability_metrics: ReliabilityMetrics,
    pub epistemic_result: EpistemicUncertaintyResult,
    pub aleatoric_result: AleatoricUncertaintyResult,
}
