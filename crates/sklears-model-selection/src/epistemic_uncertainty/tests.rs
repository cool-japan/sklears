use super::*;
use scirs2_core::ndarray::{Array1, Array2};

#[test]
fn test_epistemic_uncertainty_quantifier_creation() {
    let quantifier = EpistemicUncertaintyQuantifier::new();
    assert!(matches!(
        quantifier.config().method,
        EpistemicUncertaintyMethod::Bootstrap { .. }
    ));
    assert_eq!(quantifier.config().confidence_level, 0.95);
}

#[test]
fn test_epistemic_uncertainty_with_config() {
    let config = EpistemicUncertaintyConfig {
        method: EpistemicUncertaintyMethod::MonteCarloDropout {
            dropout_rate: 0.2,
            n_samples: 50,
        },
        confidence_level: 0.99,
        random_state: Some(42),
        calibration_method: CalibrationMethod::PlattScaling,
        temperature_scaling: false,
    };

    let quantifier = EpistemicUncertaintyQuantifier::with_config(config.clone());
    assert_eq!(quantifier.config().confidence_level, 0.99);
    assert_eq!(quantifier.config().random_state, Some(42));
    assert!(!quantifier.config().temperature_scaling);
}

#[test]
fn test_aleatoric_uncertainty_quantifier_creation() {
    let quantifier = AleatoricUncertaintyQuantifier::new();
    assert!(matches!(
        quantifier.config().method,
        AleatoricUncertaintyMethod::ResidualBasedUncertainty { .. }
    ));
    assert_eq!(quantifier.config().confidence_level, 0.95);
}

#[test]
fn test_uncertainty_quantifier_creation() {
    let quantifier = UncertaintyQuantifier::new();
    assert_eq!(quantifier.config().confidence_level, 0.95);
    assert!(matches!(
        quantifier.config().decomposition_method,
        UncertaintyDecompositionMethod::VarianceDecomposition
    ));
}

#[test]
fn test_calibration_method_variants() {
    let methods = vec![
        CalibrationMethod::PlattScaling,
        CalibrationMethod::IsotonicRegression,
        CalibrationMethod::TemperatureScaling,
        CalibrationMethod::HistogramBinning { n_bins: 10 },
        CalibrationMethod::None,
    ];

    for method in methods {
        let config = EpistemicUncertaintyConfig {
            method: EpistemicUncertaintyMethod::Bootstrap {
                n_bootstrap: 100,
                sample_ratio: 0.8,
            },
            confidence_level: 0.95,
            random_state: None,
            calibration_method: method,
            temperature_scaling: true,
        };

        let quantifier = EpistemicUncertaintyQuantifier::with_config(config);
        // Test creation succeeds
        assert_eq!(quantifier.config().confidence_level, 0.95);
    }
}

#[test]
fn test_uncertainty_decomposition_methods() {
    let methods = vec![
        UncertaintyDecompositionMethod::VarianceDecomposition,
        UncertaintyDecompositionMethod::InformationTheoretic,
        UncertaintyDecompositionMethod::BayesianDecomposition,
        UncertaintyDecompositionMethod::PredictiveEntropy,
        UncertaintyDecompositionMethod::MutualInformation,
    ];

    for method in methods {
        let config = UncertaintyQuantificationConfig {
            epistemic_config: EpistemicUncertaintyConfig::default(),
            aleatoric_config: AleatoricUncertaintyConfig::default(),
            decomposition_method: method,
            confidence_level: 0.95,
            random_state: None,
        };

        let quantifier = UncertaintyQuantifier::with_config(config);
        assert_eq!(quantifier.config().confidence_level, 0.95);
    }
}

#[test]
fn test_epistemic_uncertainty_methods() {
    let methods = vec![
        EpistemicUncertaintyMethod::MonteCarloDropout {
            dropout_rate: 0.1,
            n_samples: 10,
        },
        EpistemicUncertaintyMethod::DeepEnsembles { n_models: 5 },
        EpistemicUncertaintyMethod::BayesianNeuralNetwork { n_samples: 10 },
        EpistemicUncertaintyMethod::Bootstrap {
            n_bootstrap: 10,
            sample_ratio: 0.8,
        },
        EpistemicUncertaintyMethod::GaussianProcess {
            kernel_type: "rbf".to_string(),
        },
        EpistemicUncertaintyMethod::VariationalInference { n_samples: 10 },
        EpistemicUncertaintyMethod::LaplaceApproximation {
            hessian_method: "diagonal".to_string(),
        },
    ];

    for method in methods {
        let config = EpistemicUncertaintyConfig {
            method,
            confidence_level: 0.95,
            random_state: Some(42),
            calibration_method: CalibrationMethod::None,
            temperature_scaling: false,
        };

        let quantifier = EpistemicUncertaintyQuantifier::with_config(config);
        assert_eq!(quantifier.config().confidence_level, 0.95);
    }
}

#[test]
fn test_aleatoric_uncertainty_methods() {
    let methods = vec![
        AleatoricUncertaintyMethod::HeteroskedasticRegression { n_ensemble: 5 },
        AleatoricUncertaintyMethod::MixtureDensityNetwork { n_components: 3 },
        AleatoricUncertaintyMethod::QuantileRegression {
            quantiles: vec![0.1, 0.5, 0.9],
        },
        AleatoricUncertaintyMethod::ParametricUncertainty {
            distribution: "normal".to_string(),
        },
        AleatoricUncertaintyMethod::InputDependentNoise {
            noise_model: "linear".to_string(),
        },
        AleatoricUncertaintyMethod::ResidualBasedUncertainty { window_size: 10 },
        AleatoricUncertaintyMethod::EnsembleAleatoric {
            n_models: 5,
            noise_estimation: "average".to_string(),
        },
    ];

    for method in methods {
        let config = AleatoricUncertaintyConfig {
            method,
            confidence_level: 0.95,
            random_state: Some(42),
            noise_regularization: 0.01,
            min_variance: 1e-6,
        };

        let quantifier = AleatoricUncertaintyQuantifier::with_config(config);
        assert_eq!(quantifier.config().confidence_level, 0.95);
    }
}

#[test]
fn test_builder_pattern() {
    let quantifier = EpistemicUncertaintyQuantifier::new()
        .method(EpistemicUncertaintyMethod::MonteCarloDropout {
            dropout_rate: 0.3,
            n_samples: 20,
        })
        .confidence_level(0.99)
        .random_state(123)
        .calibration_method(CalibrationMethod::TemperatureScaling)
        .temperature_scaling(true);

    assert_eq!(quantifier.config().confidence_level, 0.99);
    assert_eq!(quantifier.config().random_state, Some(123));
    assert!(quantifier.config().temperature_scaling);
}

#[test]
fn test_default_configs() {
    let epistemic_config = EpistemicUncertaintyConfig::default();
    assert_eq!(epistemic_config.confidence_level, 0.95);
    assert!(epistemic_config.temperature_scaling);

    let aleatoric_config = AleatoricUncertaintyConfig::default();
    assert_eq!(aleatoric_config.confidence_level, 0.95);
    assert_eq!(aleatoric_config.min_variance, 1e-6);

    let combined_config = UncertaintyQuantificationConfig::default();
    assert_eq!(combined_config.confidence_level, 0.95);
}

#[test]
fn test_uncertainty_result_structures() {
    // Test that result structures can be created
    let n_samples = 10;
    let predictions = Array1::<f64>::zeros(n_samples);
    let uncertainties = Array1::<f64>::ones(n_samples);
    let prediction_intervals = Array2::<f64>::zeros((n_samples, 2));

    let reliability_metrics = ReliabilityMetrics {
        calibration_error: 0.05,
        sharpness: 0.1,
        reliability_score: 0.95,
        coverage_probability: 0.95,
        prediction_interval_score: 0.2,
        continuous_ranked_probability_score: 0.15,
    };

    let uncertainty_components = UncertaintyComponents {
        model_uncertainty: uncertainties.clone(),
        data_uncertainty: Array1::<f64>::zeros(n_samples),
        parameter_uncertainty: uncertainties.clone(),
        structural_uncertainty: Array1::<f64>::zeros(n_samples),
        approximation_uncertainty: Array1::<f64>::zeros(n_samples),
    };

    let epistemic_result = EpistemicUncertaintyResult {
        predictions: predictions.clone(),
        uncertainties: uncertainties.clone(),
        prediction_intervals: prediction_intervals.clone(),
        calibration_score: 0.05,
        entropy: Array1::<f64>::zeros(n_samples),
        mutual_information: 0.1,
        epistemic_uncertainty_components: uncertainty_components,
        reliability_metrics: reliability_metrics.clone(),
    };

    assert_eq!(epistemic_result.predictions.len(), n_samples);
    assert_eq!(epistemic_result.uncertainties.len(), n_samples);
    assert_eq!(epistemic_result.calibration_score, 0.05);
}
