use scirs2_core::ndarray::Array1;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct UncertaintyDecomposition {
    pub total_uncertainty: Array1<f64>,
    pub epistemic_uncertainty: Array1<f64>,
    pub aleatoric_uncertainty: Array1<f64>,
    pub decomposition_method: String,
    pub entropy_components: HashMap<String, Array1<f64>>,
    pub mutual_information: f64,
    pub explained_variance_ratio: f64,
    pub uncertainty_ratios: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct UncertaintyComponents {
    pub model_uncertainty: Array1<f64>,
    pub data_uncertainty: Array1<f64>,
    pub parameter_uncertainty: Array1<f64>,
    pub structural_uncertainty: Array1<f64>,
    pub approximation_uncertainty: Array1<f64>,
}

#[derive(Debug, Clone)]
pub struct ReliabilityMetrics {
    pub calibration_error: f64,
    pub sharpness: f64,
    pub reliability_score: f64,
    pub coverage_probability: f64,
    pub prediction_interval_score: f64,
    pub continuous_ranked_probability_score: f64,
}

#[derive(Debug, Clone)]
pub struct ReliabilityDiagram {
    pub bin_boundaries: Array1<f64>,
    pub bin_accuracies: Array1<f64>,
    pub bin_confidences: Array1<f64>,
    pub bin_counts: Array1<usize>,
    pub expected_calibration_error: f64,
    pub maximum_calibration_error: f64,
}
