//! Fluent API for calibration configuration
//!
//! This module provides a builder-style fluent API for configuring calibration
//! methods with a more intuitive and readable syntax.

use crate::{CalibratedClassifierCV, CalibrationMethod};
use sklears_core::types::Float;

/// Fluent builder for calibration configuration
#[derive(Debug, Clone)]
pub struct CalibrationBuilder {
    method: Option<CalibrationMethod>,
    cv_folds: Option<usize>,
    ensemble_size: Option<usize>,
    custom_params: std::collections::HashMap<String, CalibrationParameter>,
}

/// Parameter types for calibration configuration
#[derive(Debug, Clone)]
pub enum CalibrationParameter {
    /// Float
    Float(Float),
    /// Int
    Int(usize),
    /// Bool
    Bool(bool),
    /// String
    String(String),
    /// FloatArray
    FloatArray(Vec<Float>),
    /// IntArray
    IntArray(Vec<usize>),
}

impl CalibrationBuilder {
    /// Create a new calibration builder
    pub fn new() -> Self {
        Self {
            method: None,
            cv_folds: None,
            ensemble_size: None,
            custom_params: std::collections::HashMap::new(),
        }
    }

    /// Set the calibration method to sigmoid (Platt scaling)
    pub fn sigmoid(mut self) -> Self {
        self.method = Some(CalibrationMethod::Sigmoid);
        self
    }

    /// Set the calibration method to isotonic regression
    pub fn isotonic(mut self) -> Self {
        self.method = Some(CalibrationMethod::Isotonic);
        self
    }

    /// Set the calibration method to temperature scaling
    pub fn temperature(mut self) -> Self {
        self.method = Some(CalibrationMethod::Temperature);
        self
    }

    /// Set histogram binning calibration with specified number of bins
    pub fn histogram_binning(mut self, n_bins: usize) -> Self {
        self.method = Some(CalibrationMethod::HistogramBinning { n_bins });
        self
    }

    /// Set Bayesian binning into quantiles (BBQ) calibration
    pub fn bbq(mut self, min_bins: usize, max_bins: usize) -> Self {
        self.method = Some(CalibrationMethod::BBQ { min_bins, max_bins });
        self
    }

    /// Set beta calibration
    pub fn beta(mut self) -> Self {
        self.method = Some(CalibrationMethod::Beta);
        self
    }

    /// Set ensemble temperature scaling
    pub fn ensemble_temperature(mut self, n_estimators: usize) -> Self {
        self.method = Some(CalibrationMethod::EnsembleTemperature { n_estimators });
        self
    }

    /// Set one-vs-one multiclass calibration
    pub fn one_vs_one(mut self) -> Self {
        self.method = Some(CalibrationMethod::OneVsOne);
        self
    }

    /// Set multiclass temperature scaling
    pub fn multiclass_temperature(mut self) -> Self {
        self.method = Some(CalibrationMethod::MulticlassTemperature);
        self
    }

    /// Set matrix scaling for multiclass
    pub fn matrix_scaling(mut self) -> Self {
        self.method = Some(CalibrationMethod::MatrixScaling);
        self
    }

    /// Set Dirichlet calibration for multiclass
    pub fn dirichlet(mut self, concentration: Float) -> Self {
        self.method = Some(CalibrationMethod::Dirichlet { concentration });
        self
    }

    /// Set local k-NN calibration
    pub fn local_knn(mut self, k: usize) -> Self {
        self.method = Some(CalibrationMethod::LocalKNN { k });
        self
    }

    /// Set local binning calibration
    pub fn local_binning(mut self, n_bins: usize) -> Self {
        self.method = Some(CalibrationMethod::LocalBinning { n_bins });
        self
    }

    /// Set kernel density estimation calibration
    pub fn kde(mut self) -> Self {
        self.method = Some(CalibrationMethod::KDE);
        self
    }

    /// Set adaptive KDE calibration
    pub fn adaptive_kde(mut self, adaptation_factor: Float) -> Self {
        self.method = Some(CalibrationMethod::AdaptiveKDE { adaptation_factor });
        self
    }

    /// Set Gaussian process calibration
    pub fn gaussian_process(mut self) -> Self {
        self.method = Some(CalibrationMethod::GaussianProcess);
        self
    }

    /// Set variational Gaussian process calibration
    pub fn variational_gp(mut self, n_inducing: usize) -> Self {
        self.method = Some(CalibrationMethod::VariationalGP { n_inducing });
        self
    }

    /// Set split conformal prediction
    pub fn conformal_split(mut self, alpha: Float) -> Self {
        self.method = Some(CalibrationMethod::ConformalSplit { alpha });
        self
    }

    /// Set cross-conformal prediction
    pub fn conformal_cross(mut self, alpha: Float, n_folds: usize) -> Self {
        self.method = Some(CalibrationMethod::ConformalCross { alpha, n_folds });
        self
    }

    /// Set Jackknife+ conformal prediction
    pub fn conformal_jackknife(mut self, alpha: Float) -> Self {
        self.method = Some(CalibrationMethod::ConformalJackknife { alpha });
        self
    }

    /// Set Bayesian model averaging calibration
    pub fn bayesian_model_averaging(mut self, n_models: usize) -> Self {
        self.method = Some(CalibrationMethod::BayesianModelAveraging { n_models });
        self
    }

    /// Set variational inference calibration
    pub fn variational_inference(
        mut self,
        learning_rate: Float,
        n_samples: usize,
        max_iter: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::VariationalInference {
            learning_rate,
            n_samples,
            max_iter,
        });
        self
    }

    /// Set MCMC-based calibration
    pub fn mcmc(mut self, n_samples: usize, burn_in: usize, step_size: Float) -> Self {
        self.method = Some(CalibrationMethod::MCMC {
            n_samples,
            burn_in,
            step_size,
        });
        self
    }

    /// Set hierarchical Bayesian calibration
    pub fn hierarchical_bayesian(mut self) -> Self {
        self.method = Some(CalibrationMethod::HierarchicalBayesian);
        self
    }

    /// Set Dirichlet Process calibration
    pub fn dirichlet_process(mut self, concentration: Float, max_clusters: usize) -> Self {
        self.method = Some(CalibrationMethod::DirichletProcess {
            concentration,
            max_clusters,
        });
        self
    }

    /// Set non-parametric Gaussian Process calibration
    pub fn nonparametric_gp(mut self, kernel_type: String, n_inducing: usize) -> Self {
        self.method = Some(CalibrationMethod::NonParametricGP {
            kernel_type,
            n_inducing,
        });
        self
    }

    /// Set time series calibration
    pub fn time_series(mut self, window_size: usize, temporal_decay: Float) -> Self {
        self.method = Some(CalibrationMethod::TimeSeries {
            window_size,
            temporal_decay,
        });
        self
    }

    /// Set regression calibration
    pub fn regression(mut self, distributional: bool) -> Self {
        self.method = Some(CalibrationMethod::Regression { distributional });
        self
    }

    /// Set ranking calibration
    pub fn ranking(mut self, ranking_weight: Float, listwise: bool) -> Self {
        self.method = Some(CalibrationMethod::Ranking {
            ranking_weight,
            listwise,
        });
        self
    }

    /// Set survival analysis calibration
    pub fn survival(mut self, time_points: Vec<Float>, handle_censoring: bool) -> Self {
        self.method = Some(CalibrationMethod::Survival {
            time_points,
            handle_censoring,
        });
        self
    }

    /// Set neural network calibration
    pub fn neural_calibration(
        mut self,
        hidden_dims: Vec<usize>,
        activation: String,
        learning_rate: Float,
        epochs: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::NeuralCalibration {
            hidden_dims,
            activation,
            learning_rate,
            epochs,
        });
        self
    }

    /// Set mixup calibration
    pub fn mixup_calibration(
        mut self,
        base_method: String,
        alpha: Float,
        num_mixup_samples: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::MixupCalibration {
            base_method,
            alpha,
            num_mixup_samples,
        });
        self
    }

    /// Set dropout-based uncertainty calibration
    pub fn dropout_calibration(
        mut self,
        hidden_dims: Vec<usize>,
        dropout_prob: Float,
        mc_samples: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::DropoutCalibration {
            hidden_dims,
            dropout_prob,
            mc_samples,
        });
        self
    }

    /// Set ensemble neural calibration
    pub fn ensemble_neural_calibration(
        mut self,
        n_estimators: usize,
        hidden_dims: Vec<usize>,
    ) -> Self {
        self.method = Some(CalibrationMethod::EnsembleNeuralCalibration {
            n_estimators,
            hidden_dims,
        });
        self
    }

    /// Set structured prediction calibration
    pub fn structured_prediction(
        mut self,
        structure_type: String,
        use_mrf: bool,
        temperature: Float,
    ) -> Self {
        self.method = Some(CalibrationMethod::StructuredPrediction {
            structure_type,
            use_mrf,
            temperature,
        });
        self
    }

    /// Set online sigmoid calibration
    pub fn online_sigmoid(
        mut self,
        learning_rate: Float,
        use_momentum: bool,
        momentum: Float,
    ) -> Self {
        self.method = Some(CalibrationMethod::OnlineSigmoid {
            learning_rate,
            use_momentum,
            momentum,
        });
        self
    }

    /// Set adaptive online calibration
    pub fn adaptive_online(
        mut self,
        window_size: usize,
        retrain_frequency: usize,
        drift_threshold: Float,
    ) -> Self {
        self.method = Some(CalibrationMethod::AdaptiveOnline {
            window_size,
            retrain_frequency,
            drift_threshold,
        });
        self
    }

    /// Set incremental calibration updates
    pub fn incremental_update(
        mut self,
        update_frequency: usize,
        learning_rate: Float,
        use_smoothing: bool,
    ) -> Self {
        self.method = Some(CalibrationMethod::IncrementalUpdate {
            update_frequency,
            learning_rate,
            use_smoothing,
        });
        self
    }

    /// Set calibration-aware focal loss training
    pub fn calibration_aware_focal(
        mut self,
        gamma: Float,
        temperature: Float,
        learning_rate: Float,
        max_epochs: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::CalibrationAwareFocal {
            gamma,
            temperature,
            learning_rate,
            max_epochs,
        });
        self
    }

    /// Set calibration-aware cross-entropy training
    pub fn calibration_aware_cross_entropy(
        mut self,
        lambda: Float,
        learning_rate: Float,
        max_epochs: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::CalibrationAwareCrossEntropy {
            lambda,
            learning_rate,
            max_epochs,
        });
        self
    }

    /// Set calibration-aware Brier score training
    pub fn calibration_aware_brier(mut self, learning_rate: Float, max_epochs: usize) -> Self {
        self.method = Some(CalibrationMethod::CalibrationAwareBrier {
            learning_rate,
            max_epochs,
        });
        self
    }

    /// Set calibration-aware ECE training
    pub fn calibration_aware_ece(
        mut self,
        n_bins: usize,
        learning_rate: Float,
        max_epochs: usize,
    ) -> Self {
        self.method = Some(CalibrationMethod::CalibrationAwareECE {
            n_bins,
            learning_rate,
            max_epochs,
        });
        self
    }

    /// Set number of cross-validation folds
    pub fn cv_folds(mut self, folds: usize) -> Self {
        self.cv_folds = Some(folds);
        self
    }

    /// Set ensemble size for ensemble methods
    pub fn ensemble_size(mut self, size: usize) -> Self {
        self.ensemble_size = Some(size);
        self
    }

    /// Add a custom float parameter
    pub fn with_float_param(mut self, name: String, value: Float) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::Float(value));
        self
    }

    /// Add a custom integer parameter
    pub fn with_int_param(mut self, name: String, value: usize) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::Int(value));
        self
    }

    /// Add a custom boolean parameter
    pub fn with_bool_param(mut self, name: String, value: bool) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::Bool(value));
        self
    }

    /// Add a custom string parameter
    pub fn with_string_param(mut self, name: String, value: String) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::String(value));
        self
    }

    /// Add a custom float array parameter
    pub fn with_float_array_param(mut self, name: String, value: Vec<Float>) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::FloatArray(value));
        self
    }

    /// Add a custom integer array parameter
    pub fn with_int_array_param(mut self, name: String, value: Vec<usize>) -> Self {
        self.custom_params
            .insert(name, CalibrationParameter::IntArray(value));
        self
    }

    /// Build the CalibratedClassifierCV with the configured parameters
    pub fn build(self) -> CalibratedClassifierCV {
        let mut calibrator = CalibratedClassifierCV::new();

        if let Some(method) = self.method {
            calibrator = calibrator.method(method);
        }

        if let Some(cv) = self.cv_folds {
            calibrator = calibrator.cv(cv);
        }

        calibrator
    }

    /// Get the current method
    pub fn get_method(&self) -> Option<&CalibrationMethod> {
        self.method.as_ref()
    }

    /// Get the current CV folds
    pub fn get_cv_folds(&self) -> Option<usize> {
        self.cv_folds
    }

    /// Get a custom parameter
    pub fn get_custom_param(&self, name: &str) -> Option<&CalibrationParameter> {
        self.custom_params.get(name)
    }

    /// List all custom parameter names
    pub fn list_custom_params(&self) -> Vec<&String> {
        self.custom_params.keys().collect()
    }
}

impl Default for CalibrationBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Preset configurations for common calibration scenarios
pub struct CalibrationPresets;

impl CalibrationPresets {
    /// Fast calibration for real-time applications
    pub fn fast() -> CalibrationBuilder {
        CalibrationBuilder::new().sigmoid().cv_folds(2)
    }

    /// Accurate calibration for high-quality results
    pub fn accurate() -> CalibrationBuilder {
        CalibrationBuilder::new().isotonic().cv_folds(5)
    }

    /// Robust calibration for noisy data
    pub fn robust() -> CalibrationBuilder {
        CalibrationBuilder::new().bbq(5, 15).cv_folds(3)
    }

    /// Deep learning calibration for neural networks
    pub fn deep_learning() -> CalibrationBuilder {
        CalibrationBuilder::new().temperature().cv_folds(3)
    }

    /// Advanced calibration with uncertainty quantification
    pub fn advanced() -> CalibrationBuilder {
        CalibrationBuilder::new().gaussian_process().cv_folds(5)
    }

    /// Multiclass calibration for multi-label problems
    pub fn multiclass() -> CalibrationBuilder {
        CalibrationBuilder::new()
            .multiclass_temperature()
            .cv_folds(3)
    }

    /// Bayesian calibration for uncertainty-aware predictions
    pub fn bayesian() -> CalibrationBuilder {
        CalibrationBuilder::new()
            .bayesian_model_averaging(5)
            .cv_folds(3)
    }

    /// Online calibration for streaming data
    pub fn online() -> CalibrationBuilder {
        CalibrationBuilder::new().online_sigmoid(0.01, true, 0.9)
    }

    /// Ensemble calibration for improved robustness
    pub fn ensemble() -> CalibrationBuilder {
        CalibrationBuilder::new()
            .ensemble_temperature(5)
            .cv_folds(3)
    }

    /// Conformal prediction for distribution-free uncertainty
    pub fn conformal() -> CalibrationBuilder {
        CalibrationBuilder::new().conformal_split(0.1).cv_folds(3)
    }
}

/// Chaining builder for combining multiple calibration methods
pub struct CalibrationChain {
    methods: Vec<CalibrationMethod>,
    weights: Vec<Float>,
}

impl CalibrationChain {
    /// Create a new calibration chain
    pub fn new() -> Self {
        Self {
            methods: Vec::new(),
            weights: Vec::new(),
        }
    }

    /// Add a calibration method to the chain
    pub fn add_method(mut self, method: CalibrationMethod, weight: Float) -> Self {
        self.methods.push(method);
        self.weights.push(weight);
        self
    }

    /// Add a method with equal weight
    pub fn add_equal_weight(self, method: CalibrationMethod) -> Self {
        let weight = 1.0 / (self.methods.len() + 1) as Float;
        self.add_method(method, weight)
    }

    /// Build an ensemble calibration method
    pub fn build_ensemble(self) -> CalibrationBuilder {
        // For now, use ensemble temperature scaling as the representative ensemble method
        // In a full implementation, this would create a custom ensemble method
        CalibrationBuilder::new().ensemble_temperature(self.methods.len())
    }

    /// Get the methods in the chain
    pub fn get_methods(&self) -> &[CalibrationMethod] {
        &self.methods
    }

    /// Get the weights in the chain
    pub fn get_weights(&self) -> &[Float] {
        &self.weights
    }
}

impl Default for CalibrationChain {
    fn default() -> Self {
        Self::new()
    }
}

/// Calibration configuration validator
pub struct CalibrationValidator;

impl CalibrationValidator {
    /// Validate a calibration configuration
    pub fn validate(builder: &CalibrationBuilder) -> Result<(), String> {
        if builder.method.is_none() {
            return Err("Calibration method must be specified".to_string());
        }

        if let Some(cv_folds) = builder.cv_folds {
            if cv_folds < 2 {
                return Err("CV folds must be at least 2".to_string());
            }
        }

        // Validate method-specific constraints
        if let Some(method) = &builder.method {
            match method {
                CalibrationMethod::HistogramBinning { n_bins } => {
                    if *n_bins < 1 {
                        return Err("Number of bins must be positive".to_string());
                    }
                }
                CalibrationMethod::BBQ { min_bins, max_bins } => {
                    if min_bins >= max_bins {
                        return Err("min_bins must be less than max_bins".to_string());
                    }
                }
                CalibrationMethod::LocalKNN { k } => {
                    if *k < 1 {
                        return Err("k must be positive for k-NN".to_string());
                    }
                }
                CalibrationMethod::Dirichlet { concentration } => {
                    if *concentration <= 0.0 {
                        return Err("Dirichlet concentration must be positive".to_string());
                    }
                }
                CalibrationMethod::ConformalSplit { alpha }
                | CalibrationMethod::ConformalCross { alpha, .. }
                | CalibrationMethod::ConformalJackknife { alpha } => {
                    if *alpha <= 0.0 || *alpha >= 1.0 {
                        return Err("Conformal alpha must be in (0, 1)".to_string());
                    }
                }
                _ => {} // Other methods have internal validation
            }
        }

        Ok(())
    }

    /// Check if a configuration is suitable for the given dataset size
    pub fn check_dataset_compatibility(
        builder: &CalibrationBuilder,
        n_samples: usize,
    ) -> Result<(), String> {
        if let Some(method) = &builder.method {
            match method {
                CalibrationMethod::GaussianProcess => {
                    if n_samples > 5000 {
                        return Err(
                            "Gaussian Process calibration may be slow for large datasets"
                                .to_string(),
                        );
                    }
                }
                CalibrationMethod::LocalKNN { k } => {
                    if *k >= n_samples {
                        return Err("k must be less than number of samples".to_string());
                    }
                }
                CalibrationMethod::HistogramBinning { n_bins } => {
                    if *n_bins > n_samples / 2 {
                        return Err("Too many bins for dataset size".to_string());
                    }
                }
                _ => {}
            }
        }

        if let Some(cv_folds) = builder.cv_folds {
            if cv_folds > n_samples {
                return Err("More CV folds than samples".to_string());
            }
        }

        Ok(())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_calibration_builder_basic() {
        let builder = CalibrationBuilder::new().sigmoid().cv_folds(5);

        assert!(matches!(
            builder.get_method(),
            Some(CalibrationMethod::Sigmoid)
        ));
        assert_eq!(builder.get_cv_folds(), Some(5));
    }

    #[test]
    fn test_calibration_builder_complex() {
        let builder = CalibrationBuilder::new()
            .gaussian_process()
            .cv_folds(3)
            .with_float_param("learning_rate".to_string(), 0.01)
            .with_int_param("max_iter".to_string(), 100);

        assert!(matches!(
            builder.get_method(),
            Some(CalibrationMethod::GaussianProcess)
        ));
        assert_eq!(builder.get_cv_folds(), Some(3));
        assert!(builder.get_custom_param("learning_rate").is_some());
        assert!(builder.get_custom_param("max_iter").is_some());
    }

    #[test]
    fn test_calibration_presets() {
        let fast = CalibrationPresets::fast();
        assert!(matches!(
            fast.get_method(),
            Some(CalibrationMethod::Sigmoid)
        ));

        let accurate = CalibrationPresets::accurate();
        assert!(matches!(
            accurate.get_method(),
            Some(CalibrationMethod::Isotonic)
        ));

        let deep_learning = CalibrationPresets::deep_learning();
        assert!(matches!(
            deep_learning.get_method(),
            Some(CalibrationMethod::Temperature)
        ));
    }

    #[test]
    fn test_calibration_chain() {
        let chain = CalibrationChain::new()
            .add_method(CalibrationMethod::Sigmoid, 0.5)
            .add_method(CalibrationMethod::Isotonic, 0.5);

        assert_eq!(chain.get_methods().len(), 2);
        assert_eq!(chain.get_weights().len(), 2);
        assert_eq!(chain.get_weights()[0], 0.5);
        assert_eq!(chain.get_weights()[1], 0.5);
    }

    #[test]
    fn test_calibration_validator_valid() {
        let builder = CalibrationBuilder::new().sigmoid().cv_folds(3);

        assert!(CalibrationValidator::validate(&builder).is_ok());
    }

    #[test]
    fn test_calibration_validator_invalid() {
        let builder = CalibrationBuilder::new(); // No method specified
        assert!(CalibrationValidator::validate(&builder).is_err());

        let builder = CalibrationBuilder::new().sigmoid().cv_folds(1); // Too few folds
        assert!(CalibrationValidator::validate(&builder).is_err());
    }

    #[test]
    fn test_dataset_compatibility() {
        let builder = CalibrationBuilder::new().local_knn(50).cv_folds(3);

        // Should be ok for large dataset
        assert!(CalibrationValidator::check_dataset_compatibility(&builder, 1000).is_ok());

        // Should fail for small dataset
        assert!(CalibrationValidator::check_dataset_compatibility(&builder, 30).is_err());
    }

    #[test]
    fn test_all_calibration_methods() {
        // Test that all method builders work
        let methods = vec![
            CalibrationBuilder::new().sigmoid().build(),
            CalibrationBuilder::new().isotonic().build(),
            CalibrationBuilder::new().temperature().build(),
            CalibrationBuilder::new().histogram_binning(10).build(),
            CalibrationBuilder::new().bbq(5, 15).build(),
            CalibrationBuilder::new().beta().build(),
            CalibrationBuilder::new().local_knn(5).build(),
            CalibrationBuilder::new().kde().build(),
            CalibrationBuilder::new().gaussian_process().build(),
        ];

        // All should build successfully
        assert_eq!(methods.len(), 9);
    }
}
