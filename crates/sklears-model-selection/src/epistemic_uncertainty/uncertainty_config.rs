use super::calibration::CalibrationMethod;
use super::uncertainty_decomposition::UncertaintyDecompositionMethod;
use super::uncertainty_methods::*;

#[derive(Debug, Clone)]
pub struct EpistemicUncertaintyConfig {
    pub method: EpistemicUncertaintyMethod,
    pub confidence_level: f64,
    pub random_state: Option<u64>,
    pub calibration_method: CalibrationMethod,
    pub temperature_scaling: bool,
}

#[derive(Debug, Clone)]
pub struct AleatoricUncertaintyConfig {
    pub method: AleatoricUncertaintyMethod,
    pub confidence_level: f64,
    pub random_state: Option<u64>,
    pub noise_regularization: f64,
    pub min_variance: f64,
}

#[derive(Debug, Clone)]
pub struct UncertaintyQuantificationConfig {
    pub epistemic_config: EpistemicUncertaintyConfig,
    pub aleatoric_config: AleatoricUncertaintyConfig,
    pub decomposition_method: UncertaintyDecompositionMethod,
    pub confidence_level: f64,
    pub random_state: Option<u64>,
}

impl Default for EpistemicUncertaintyConfig {
    fn default() -> Self {
        Self {
            method: EpistemicUncertaintyMethod::Bootstrap {
                n_bootstrap: 100,
                sample_ratio: 0.8,
            },
            confidence_level: 0.95,
            random_state: None,
            calibration_method: CalibrationMethod::TemperatureScaling,
            temperature_scaling: true,
        }
    }
}

impl Default for AleatoricUncertaintyConfig {
    fn default() -> Self {
        Self {
            method: AleatoricUncertaintyMethod::ResidualBasedUncertainty { window_size: 10 },
            confidence_level: 0.95,
            random_state: None,
            noise_regularization: 0.01,
            min_variance: 1e-6,
        }
    }
}

impl Default for UncertaintyQuantificationConfig {
    fn default() -> Self {
        Self {
            epistemic_config: EpistemicUncertaintyConfig::default(),
            aleatoric_config: AleatoricUncertaintyConfig::default(),
            decomposition_method: UncertaintyDecompositionMethod::VarianceDecomposition,
            confidence_level: 0.95,
            random_state: None,
        }
    }
}
