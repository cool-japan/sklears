//! Serialization and deserialization for isotonic regression models
//!
//! This module provides functionality to serialize and deserialize trained isotonic
//! regression models, enabling model persistence and sharing across different
//! environments and applications.

use crate::core::{IsotonicRegression, LossFunction, MonotonicityConstraint};
use scirs2_core::ndarray::Array1;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Predict, Trained},
};

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Serializable isotonic regression model
///
/// This structure contains all the necessary information to reconstruct
/// a trained isotonic regression model, including the fitted values,
/// X values, and configuration parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
/// SerializableIsotonicRegression
pub struct SerializableIsotonicRegression {
    /// The fitted isotonic values
    pub fitted_values: Vec<f64>,
    /// The X values used for fitting
    pub x_values: Vec<f64>,
    /// Monotonicity constraint
    pub constraint: MonotonicityConstraint,
    /// Loss function used
    pub loss_function: LossFunction,
    /// Lower bound for fitted values
    pub y_min: Option<f64>,
    /// Upper bound for fitted values
    pub y_max: Option<f64>,
    /// Whether the model has been fitted
    pub is_fitted: bool,
}

impl SerializableIsotonicRegression {
    /// Create a new serializable isotonic regression model
    pub fn new() -> Self {
        Self {
            fitted_values: Vec::new(),
            x_values: Vec::new(),
            constraint: MonotonicityConstraint::Global { increasing: true },
            loss_function: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            is_fitted: false,
        }
    }

    /// Create from a trained IsotonicRegression model
    pub fn from_isotonic_regression(
        model: &IsotonicRegression<Trained>,
        x_values: &Array1<f64>,
    ) -> Result<Self> {
        // Get fitted values by predicting on the training X values
        let fitted_values = model.predict(x_values)?;

        Ok(Self {
            fitted_values: fitted_values.to_vec(),
            x_values: x_values.to_vec(),
            constraint: model.constraint.clone(),
            loss_function: model.loss,
            y_min: model.y_min,
            y_max: model.y_max,
            is_fitted: true,
        })
    }

    /// Convert to IsotonicRegression model
    pub fn to_isotonic_regression(&self) -> Result<IsotonicRegression<Trained>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Model is not fitted".to_string(),
            ));
        }

        let mut model = IsotonicRegression::new()
            .constraint(self.constraint.clone())
            .loss(self.loss_function);

        if let Some(y_min) = self.y_min {
            model = model.y_min(y_min);
        }

        if let Some(y_max) = self.y_max {
            model = model.y_max(y_max);
        }

        // Reconstruct the fitted model by setting internal state
        // This is a simplified approach - in practice, you might want to
        // store more internal state or use a different approach
        let x_array = Array1::from_vec(self.x_values.clone());
        let y_array = Array1::from_vec(self.fitted_values.clone());

        // Fit the model to reconstruct the internal state
        model.fit(&x_array, &y_array)
    }

    /// Predict using the serialized model
    pub fn predict(&self, x: &Array1<f64>) -> Result<Array1<f64>> {
        if !self.is_fitted {
            return Err(SklearsError::InvalidInput(
                "Model is not fitted".to_string(),
            ));
        }

        // Perform linear interpolation based on stored fitted values
        let mut predictions = Vec::with_capacity(x.len());

        for &x_val in x.iter() {
            let prediction = self.interpolate(x_val);
            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }

    /// Interpolate a single value using the fitted model
    fn interpolate(&self, x_val: f64) -> f64 {
        if self.x_values.is_empty() || self.fitted_values.is_empty() {
            return 0.0;
        }

        // Handle extrapolation for values outside the range
        if x_val <= self.x_values[0] {
            return self.fitted_values[0];
        }

        if x_val >= self.x_values[self.x_values.len() - 1] {
            return self.fitted_values[self.fitted_values.len() - 1];
        }

        // Find the interval for interpolation
        for i in 0..self.x_values.len() - 1 {
            if x_val >= self.x_values[i] && x_val <= self.x_values[i + 1] {
                let x0 = self.x_values[i];
                let x1 = self.x_values[i + 1];
                let y0 = self.fitted_values[i];
                let y1 = self.fitted_values[i + 1];

                // Linear interpolation
                let t = (x_val - x0) / (x1 - x0);
                return y0 + t * (y1 - y0);
            }
        }

        // Fallback (should not reach here)
        self.fitted_values[self.fitted_values.len() - 1]
    }
}

impl Default for SerializableIsotonicRegression {
    fn default() -> Self {
        Self::new()
    }
}

/// Serialize an isotonic regression model to JSON
#[cfg(feature = "serde")]
pub fn serialize_to_json(
    model: &IsotonicRegression<Trained>,
    x_values: &Array1<f64>,
) -> Result<String> {
    let serializable = SerializableIsotonicRegression::from_isotonic_regression(model, x_values)?;
    serde_json::to_string(&serializable)
        .map_err(|e| SklearsError::InvalidInput(format!("Serialization failed: {}", e)))
}

/// Deserialize an isotonic regression model from JSON
#[cfg(feature = "serde")]
pub fn deserialize_from_json(json: &str) -> Result<SerializableIsotonicRegression> {
    serde_json::from_str(json)
        .map_err(|e| SklearsError::InvalidInput(format!("Deserialization failed: {}", e)))
}

/// Serialize an isotonic regression model to compact JSON format
#[cfg(feature = "serde")]
pub fn serialize_to_compact_json(
    model: &IsotonicRegression<Trained>,
    x_values: &Array1<f64>,
) -> Result<String> {
    let serializable = SerializableIsotonicRegression::from_isotonic_regression(model, x_values)?;
    serde_json::to_string(&serializable).map_err(|e| {
        SklearsError::InvalidInput(format!("Compact JSON serialization failed: {}", e))
    })
}

/// Serialize an isotonic regression model to pretty JSON format
#[cfg(feature = "serde")]
pub fn serialize_to_pretty_json(
    model: &IsotonicRegression<Trained>,
    x_values: &Array1<f64>,
) -> Result<String> {
    let serializable = SerializableIsotonicRegression::from_isotonic_regression(model, x_values)?;
    serde_json::to_string_pretty(&serializable)
        .map_err(|e| SklearsError::InvalidInput(format!("Pretty JSON serialization failed: {}", e)))
}

/// Model persistence utilities
pub struct ModelPersistence;

impl ModelPersistence {
    /// Save model to file (JSON format)
    #[cfg(feature = "serde")]
    pub fn save_to_file(
        model: &IsotonicRegression<Trained>,
        x_values: &Array1<f64>,
        path: &str,
    ) -> Result<()> {
        let json = serialize_to_json(model, x_values)?;
        std::fs::write(path, json)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to write file: {}", e)))
    }

    /// Load model from file (JSON format)
    #[cfg(feature = "serde")]
    pub fn load_from_file(path: &str) -> Result<SerializableIsotonicRegression> {
        let json = std::fs::read_to_string(path)
            .map_err(|e| SklearsError::InvalidInput(format!("Failed to read file: {}", e)))?;
        deserialize_from_json(&json)
    }

    /// Save model to pretty JSON file
    #[cfg(feature = "serde")]
    pub fn save_to_pretty_file(
        model: &IsotonicRegression<Trained>,
        x_values: &Array1<f64>,
        path: &str,
    ) -> Result<()> {
        let json = serialize_to_pretty_json(model, x_values)?;
        std::fs::write(path, json).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to write pretty JSON file: {}", e))
        })
    }

    /// Save model to compact JSON file
    #[cfg(feature = "serde")]
    pub fn save_to_compact_file(
        model: &IsotonicRegression<Trained>,
        x_values: &Array1<f64>,
        path: &str,
    ) -> Result<()> {
        let json = serialize_to_compact_json(model, x_values)?;
        std::fs::write(path, json).map_err(|e| {
            SklearsError::InvalidInput(format!("Failed to write compact JSON file: {}", e))
        })
    }
}

/// Convenience function for serializing an isotonic regression model
pub fn serialize_isotonic_model(
    model: &IsotonicRegression<Trained>,
    x_values: &Array1<f64>,
) -> Result<SerializableIsotonicRegression> {
    SerializableIsotonicRegression::from_isotonic_regression(model, x_values)
}

/// Convenience function for deserializing an isotonic regression model
pub fn deserialize_isotonic_model(
    serializable: &SerializableIsotonicRegression,
) -> Result<IsotonicRegression<Trained>> {
    serializable.to_isotonic_regression()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_serializable_isotonic_regression() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();
        let original_predictions = fitted.predict(&x).unwrap();

        // Create serializable model
        let serializable =
            SerializableIsotonicRegression::from_isotonic_regression(&fitted, &x).unwrap();

        // Test prediction with serializable model
        let serializable_predictions = serializable.predict(&x).unwrap();

        // Predictions should be close (may not be exactly equal due to reconstruction)
        for i in 0..original_predictions.len() {
            assert_abs_diff_eq!(
                original_predictions[i],
                serializable_predictions[i],
                epsilon = 1e-6
            );
        }
    }

    #[test]
    fn test_serializable_interpolation() {
        let serializable = SerializableIsotonicRegression {
            fitted_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            x_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            constraint: MonotonicityConstraint::Global { increasing: true },
            loss_function: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            is_fitted: true,
        };

        // Test interpolation
        let test_x = array![1.5, 2.5, 3.5];
        let predictions = serializable.predict(&test_x).unwrap();

        assert_abs_diff_eq!(predictions[0], 1.5, epsilon = 1e-6);
        assert_abs_diff_eq!(predictions[1], 2.5, epsilon = 1e-6);
        assert_abs_diff_eq!(predictions[2], 3.5, epsilon = 1e-6);
    }

    #[test]
    fn test_serializable_extrapolation() {
        let serializable = SerializableIsotonicRegression {
            fitted_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            x_values: vec![1.0, 2.0, 3.0, 4.0, 5.0],
            constraint: MonotonicityConstraint::Global { increasing: true },
            loss_function: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            is_fitted: true,
        };

        // Test extrapolation
        let test_x = array![0.5, 6.0];
        let predictions = serializable.predict(&test_x).unwrap();

        assert_abs_diff_eq!(predictions[0], 1.0, epsilon = 1e-6); // Should use first value
        assert_abs_diff_eq!(predictions[1], 5.0, epsilon = 1e-6); // Should use last value
    }

    #[cfg(feature = "serde")]
    #[test]
    fn test_json_serialization() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();

        // Test JSON serialization
        let json = serialize_to_json(&fitted, &x).unwrap();
        let deserialized = deserialize_from_json(&json).unwrap();

        assert_eq!(deserialized.x_values, x.to_vec());
        assert_eq!(deserialized.constraint, MonotonicityConstraint::Increasing);
        assert_eq!(deserialized.is_fitted, true);
    }

    #[test]
    fn test_unfitted_model_error() {
        let serializable = SerializableIsotonicRegression::new();
        let test_x = array![1.0, 2.0, 3.0];

        let result = serializable.predict(&test_x);
        assert!(result.is_err());
    }

    #[test]
    fn test_empty_model_interpolation() {
        let serializable = SerializableIsotonicRegression {
            fitted_values: vec![],
            x_values: vec![],
            constraint: MonotonicityConstraint::Global { increasing: true },
            loss_function: LossFunction::SquaredLoss,
            y_min: None,
            y_max: None,
            is_fitted: true,
        };

        let prediction = serializable.interpolate(1.0);
        assert_eq!(prediction, 0.0);
    }

    #[test]
    fn test_serialize_convenience_functions() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![1.0, 3.0, 2.0, 4.0, 5.0];

        let iso = IsotonicRegression::new();
        let fitted = iso.fit(&x, &y).unwrap();

        // Test convenience functions
        let serializable = serialize_isotonic_model(&fitted, &x).unwrap();
        let reconstructed = deserialize_isotonic_model(&serializable).unwrap();

        // Test that reconstructed model can make predictions
        let predictions = reconstructed.predict(&x).unwrap();
        assert_eq!(predictions.len(), 5);
    }
}
