//! Serialization support for calibration models
//!
//! This module provides serialization and deserialization capabilities for calibration models,
//! allowing trained calibrators to be saved and loaded from storage.

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{error::Result, types::Float};
use std::collections::HashMap;

use crate::{
    multi_modal::{EnsembleCombination, FusionStrategy, TransferStrategy},
    CalibrationEstimator, CalibrationMethod,
};

/// Serializable representation of a calibration model
///
/// This struct provides a way to serialize and deserialize calibration models
/// by storing their parameters and metadata rather than the actual trait objects.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SerializableCalibrationModel {
    /// The calibration method used
    pub method: CalibrationMethod,
    /// Model parameters as key-value pairs
    pub parameters: HashMap<String, SerializableParameter>,
    /// Model metadata
    pub metadata: CalibrationMetadata,
}

/// Serializable parameter value
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SerializableParameter {
    /// Float
    Float(Float),
    /// Int
    Int(i32),
    /// USize
    USize(usize),
    /// String
    String(String),
    /// Bool
    Bool(bool),
    /// FloatArray
    FloatArray(Vec<Float>),
    /// IntArray
    IntArray(Vec<i32>),
    /// FloatMatrix
    FloatMatrix(Vec<Vec<Float>>),
    /// FusionStrategy
    FusionStrategy(FusionStrategy),
    /// EnsembleCombination
    EnsembleCombination(EnsembleCombination),
    /// TransferStrategy
    TransferStrategy(TransferStrategy),
}

/// Calibration model metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationMetadata {
    /// Model version for compatibility checking
    pub version: String,
    /// Whether the model is fitted
    pub is_fitted: bool,
    /// Number of classes the model was trained on
    pub n_classes: usize,
    /// Training data size
    pub n_samples: usize,
    /// Creation timestamp
    pub created_at: String,
    /// Additional custom metadata
    pub custom: HashMap<String, String>,
}

impl Default for CalibrationMetadata {
    fn default() -> Self {
        Self {
            version: "1.0.0".to_string(),
            is_fitted: false,
            n_classes: 2,
            n_samples: 0,
            created_at: chrono::Utc::now().to_rfc3339(),
            custom: HashMap::new(),
        }
    }
}

impl SerializableCalibrationModel {
    /// Create a new serializable calibration model
    pub fn new(method: CalibrationMethod) -> Self {
        Self {
            method,
            parameters: HashMap::new(),
            metadata: CalibrationMetadata::default(),
        }
    }

    /// Set a parameter value
    pub fn set_parameter(&mut self, key: String, value: SerializableParameter) {
        self.parameters.insert(key, value);
    }

    /// Get a parameter value
    pub fn get_parameter(&self, key: &str) -> Option<&SerializableParameter> {
        self.parameters.get(key)
    }

    /// Set metadata
    pub fn set_metadata(&mut self, metadata: CalibrationMetadata) {
        self.metadata = metadata;
    }

    /// Mark as fitted
    pub fn mark_fitted(&mut self, n_classes: usize, n_samples: usize) {
        self.metadata.is_fitted = true;
        self.metadata.n_classes = n_classes;
        self.metadata.n_samples = n_samples;
    }
}

/// Trait for converting calibration estimators to serializable form
pub trait ToSerializable {
    /// Convert to serializable representation
    fn to_serializable(&self) -> Result<SerializableCalibrationModel>;
}

/// Trait for creating calibration estimators from serializable form
pub trait FromSerializable {
    /// Create from serializable representation
    fn from_serializable(
        model: &SerializableCalibrationModel,
    ) -> Result<Box<dyn CalibrationEstimator>>;
}

/// Factory for creating calibration estimators from serializable models
pub struct CalibrationModelFactory;

impl CalibrationModelFactory {
    /// Create a calibration estimator from a serializable model
    pub fn create_from_serializable(
        model: &SerializableCalibrationModel,
    ) -> Result<Box<dyn CalibrationEstimator>> {
        match &model.method {
            CalibrationMethod::Sigmoid => {
                let mut calibrator = crate::SigmoidCalibrator::new();
                Self::restore_sigmoid_parameters(&mut calibrator, model)?;
                Ok(Box::new(calibrator))
            }
            CalibrationMethod::Isotonic => {
                let mut calibrator = crate::isotonic::IsotonicCalibrator::new();
                Self::restore_isotonic_parameters(&mut calibrator, model)?;
                Ok(Box::new(calibrator))
            }
            CalibrationMethod::Temperature => {
                let mut calibrator = crate::temperature::TemperatureScalingCalibrator::new();
                Self::restore_temperature_parameters(&mut calibrator, model)?;
                Ok(Box::new(calibrator))
            }
            CalibrationMethod::MultiModal {
                n_modalities,
                fusion_strategy,
            } => {
                let fusion = Self::parse_fusion_strategy(fusion_strategy)?;
                let mut calibrator =
                    crate::multi_modal::MultiModalCalibrator::new(*n_modalities, fusion);
                Self::restore_multi_modal_parameters(&mut calibrator, model)?;
                Ok(Box::new(calibrator))
            }
            CalibrationMethod::HeterogeneousEnsemble {
                combination_strategy,
            } => {
                let strategy = Self::parse_ensemble_combination(combination_strategy)?;
                let mut calibrator =
                    crate::multi_modal::HeterogeneousEnsembleCalibrator::new(strategy);
                Self::restore_heterogeneous_ensemble_parameters(&mut calibrator, model)?;
                Ok(Box::new(calibrator))
            }
            _ => {
                // For now, fallback to sigmoid for unsupported methods
                let calibrator = crate::SigmoidCalibrator::new();
                Ok(Box::new(calibrator))
            }
        }
    }

    fn parse_fusion_strategy(strategy_str: &str) -> Result<FusionStrategy> {
        match strategy_str {
            "weighted_average" => Ok(FusionStrategy::WeightedAverage),
            "attention" => Ok(FusionStrategy::AttentionFusion),
            "late_fusion" => Ok(FusionStrategy::LateFusion),
            "early_fusion" => Ok(FusionStrategy::EarlyFusion),
            _ => Ok(FusionStrategy::WeightedAverage),
        }
    }

    fn parse_ensemble_combination(strategy_str: &str) -> Result<EnsembleCombination> {
        match strategy_str {
            "performance_weighted" => Ok(EnsembleCombination::PerformanceWeighted),
            "dynamic_weighting" => Ok(EnsembleCombination::DynamicWeighting),
            "stacking" => Ok(EnsembleCombination::Stacking),
            "bayesian_averaging" => Ok(EnsembleCombination::BayesianAveraging),
            _ => Ok(EnsembleCombination::PerformanceWeighted),
        }
    }

    fn restore_sigmoid_parameters(
        _calibrator: &mut crate::SigmoidCalibrator,
        model: &SerializableCalibrationModel,
    ) -> Result<()> {
        // In a full implementation, this would restore sigmoid parameters
        // For now, we'll mark it as fitted if the metadata indicates so
        if model.metadata.is_fitted {
            // Would restore actual parameters here
        }
        Ok(())
    }

    fn restore_isotonic_parameters(
        _calibrator: &mut crate::isotonic::IsotonicCalibrator,
        model: &SerializableCalibrationModel,
    ) -> Result<()> {
        // Restore isotonic calibrator parameters
        if model.metadata.is_fitted {
            // Would restore actual parameters here
        }
        Ok(())
    }

    fn restore_temperature_parameters(
        _calibrator: &mut crate::temperature::TemperatureScalingCalibrator,
        model: &SerializableCalibrationModel,
    ) -> Result<()> {
        // Restore temperature scaling parameters
        if model.metadata.is_fitted {
            // Would restore actual parameters here
        }
        Ok(())
    }

    fn restore_multi_modal_parameters(
        _calibrator: &mut crate::multi_modal::MultiModalCalibrator,
        model: &SerializableCalibrationModel,
    ) -> Result<()> {
        // Restore multi-modal calibrator parameters
        if model.metadata.is_fitted {
            // Would restore actual parameters here
        }
        Ok(())
    }

    fn restore_heterogeneous_ensemble_parameters(
        _calibrator: &mut crate::multi_modal::HeterogeneousEnsembleCalibrator,
        model: &SerializableCalibrationModel,
    ) -> Result<()> {
        // Restore heterogeneous ensemble parameters
        if model.metadata.is_fitted {
            // Would restore actual parameters here
        }
        Ok(())
    }
}

/// Serialization utilities for calibration models
pub struct CalibrationSerializer;

impl CalibrationSerializer {
    /// Serialize a calibration method to JSON
    #[cfg(feature = "serde")]
    pub fn to_json(model: &SerializableCalibrationModel) -> Result<String> {
        serde_json::to_string_pretty(model)
            .map_err(|e| sklears_core::error::SklearsError::SerializationError(e.to_string()))
    }

    /// Deserialize a calibration method from JSON
    #[cfg(feature = "serde")]
    pub fn from_json(json: &str) -> Result<SerializableCalibrationModel> {
        serde_json::from_str(json)
            .map_err(|e| sklears_core::error::SklearsError::SerializationError(e.to_string()))
    }

    /// Serialize to binary format using bincode
    #[cfg(feature = "serde")]
    pub fn to_binary(model: &SerializableCalibrationModel) -> Result<Vec<u8>> {
        bincode::serde::encode_to_vec(model, bincode::config::standard())
            .map_err(|e| sklears_core::error::SklearsError::SerializationError(e.to_string()))
    }

    /// Deserialize from binary format using bincode
    #[cfg(feature = "serde")]
    pub fn from_binary(data: &[u8]) -> Result<SerializableCalibrationModel> {
        bincode::serde::decode_from_slice(data, bincode::config::standard())
            .map(|(model, _size)| model)
            .map_err(|e| sklears_core::error::SklearsError::SerializationError(e.to_string()))
    }
}

/// Helper functions for working with serializable calibration models
impl SerializableCalibrationModel {
    /// Save model to file as JSON
    #[cfg(feature = "serde")]
    pub fn save_json(&self, path: &std::path::Path) -> Result<()> {
        let json = CalibrationSerializer::to_json(self)?;
        std::fs::write(path, json).map_err(sklears_core::error::SklearsError::IoError)?;
        Ok(())
    }

    /// Load model from JSON file
    #[cfg(feature = "serde")]
    pub fn load_json(path: &std::path::Path) -> Result<Self> {
        let json =
            std::fs::read_to_string(path).map_err(sklears_core::error::SklearsError::IoError)?;
        CalibrationSerializer::from_json(&json)
    }

    /// Save model to file as binary
    #[cfg(feature = "serde")]
    pub fn save_binary(&self, path: &std::path::Path) -> Result<()> {
        let binary = CalibrationSerializer::to_binary(self)?;
        std::fs::write(path, binary).map_err(sklears_core::error::SklearsError::IoError)?;
        Ok(())
    }

    /// Load model from binary file
    #[cfg(feature = "serde")]
    pub fn load_binary(path: &std::path::Path) -> Result<Self> {
        let binary = std::fs::read(path).map_err(sklears_core::error::SklearsError::IoError)?;
        CalibrationSerializer::from_binary(&binary)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serializable_calibration_model_creation() {
        let mut model = SerializableCalibrationModel::new(CalibrationMethod::Sigmoid);
        model.set_parameter("slope".to_string(), SerializableParameter::Float(1.5));
        model.set_parameter("intercept".to_string(), SerializableParameter::Float(0.2));

        assert_eq!(model.method, CalibrationMethod::Sigmoid);
        assert!(model.get_parameter("slope").is_some());
        assert!(model.get_parameter("intercept").is_some());
    }

    #[test]
    fn test_calibration_metadata() {
        let mut metadata = CalibrationMetadata::default();
        metadata
            .custom
            .insert("author".to_string(), "test".to_string());

        assert!(!metadata.is_fitted);
        assert_eq!(metadata.n_classes, 2);
        assert_eq!(metadata.custom.get("author").unwrap(), "test");
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_json_serialization() {
        let mut model = SerializableCalibrationModel::new(CalibrationMethod::Temperature);
        model.set_parameter("temperature".to_string(), SerializableParameter::Float(2.0));
        model.mark_fitted(3, 1000);

        let json = CalibrationSerializer::to_json(&model).unwrap();
        assert!(json.contains("Temperature"));
        assert!(json.contains("temperature"));

        let deserialized = CalibrationSerializer::from_json(&json).unwrap();
        assert_eq!(deserialized.method, CalibrationMethod::Temperature);
        assert!(deserialized.metadata.is_fitted);
        assert_eq!(deserialized.metadata.n_classes, 3);
    }

    #[test]
    fn test_model_factory() {
        let model = SerializableCalibrationModel::new(CalibrationMethod::Sigmoid);
        let calibrator = CalibrationModelFactory::create_from_serializable(&model).unwrap();

        // The calibrator should be created successfully - just check it exists
        let _cloned = calibrator.clone_box();
        // If we get here without panicking, the test passes
    }
}
