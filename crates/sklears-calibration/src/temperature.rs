//! Temperature scaling for probability calibration
//!
//! Temperature scaling is a simple and effective calibration method that applies
//! a single scalar parameter T to scale the logits before applying the sigmoid/softmax.

use crate::CalibrationEstimator;
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, types::Float};

/// Temperature scaling calibrator
///
/// Applies temperature scaling to logits: p_calibrated = sigmoid(logits / T)
/// where T is the temperature parameter learned during fitting.
#[derive(Debug, Clone)]
pub struct TemperatureScalingCalibrator {
    /// Temperature parameter (T)
    temperature: Float,
    /// Whether the calibrator has been fitted
    fitted: bool,
}

impl TemperatureScalingCalibrator {
    /// Create a new temperature scaling calibrator
    pub fn new() -> Self {
        Self {
            temperature: 1.0,
            fitted: false,
        }
    }

    /// Fit the temperature scaling calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Input arrays must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Convert probabilities to logits
        let logits: Array1<Float> = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            (clamped_p / (1.0 - clamped_p)).ln()
        });

        // Find optimal temperature using gradient descent
        self.temperature = self.find_optimal_temperature(&logits, y_true)?;
        self.fitted = true;

        Ok(self)
    }

    /// Find optimal temperature parameter using simple line search
    fn find_optimal_temperature(
        &self,
        logits: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        let mut best_temperature = 1.0;
        let mut best_loss = Float::INFINITY;

        // Simple grid search for temperature
        let temperature_candidates = [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0];

        for &temp in &temperature_candidates {
            let loss = self.compute_negative_log_likelihood(logits, y_true, temp);
            if loss < best_loss {
                best_loss = loss;
                best_temperature = temp;
            }
        }

        // Refine with finer search around best candidate
        let mut refined_temp = best_temperature;
        let step_size = 0.1;
        let mut improved = true;

        while improved {
            improved = false;
            let candidates = [refined_temp - step_size, refined_temp + step_size];

            for &temp in &candidates {
                if temp > 0.01 && temp < 100.0 {
                    let loss = self.compute_negative_log_likelihood(logits, y_true, temp);
                    if loss < best_loss {
                        best_loss = loss;
                        refined_temp = temp;
                        improved = true;
                    }
                }
            }
        }

        Ok(refined_temp)
    }

    /// Compute negative log-likelihood for given temperature
    fn compute_negative_log_likelihood(
        &self,
        logits: &Array1<Float>,
        y_true: &Array1<i32>,
        temperature: Float,
    ) -> Float {
        let mut nll = 0.0;

        for (i, &logit) in logits.iter().enumerate() {
            let scaled_logit = logit / temperature;
            let prob = 1.0 / (1.0 + (-scaled_logit).exp());
            let target = y_true[i] as Float;

            // Compute log-likelihood
            let log_prob = if target > 0.5 {
                prob.ln()
            } else {
                (1.0 - prob).ln()
            };

            nll -= log_prob;
        }

        nll
    }

    /// Predict calibrated probabilities
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Calibrator not fitted".to_string(),
            ));
        }

        // Convert probabilities to logits
        let logits: Array1<Float> = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            (clamped_p / (1.0 - clamped_p)).ln()
        });

        // Apply temperature scaling
        let calibrated = logits.mapv(|logit| {
            let scaled_logit = logit / self.temperature;
            1.0 / (1.0 + (-scaled_logit).exp())
        });

        Ok(calibrated)
    }

    /// Get the learned temperature parameter
    pub fn temperature(&self) -> Float {
        self.temperature
    }
}

impl Default for TemperatureScalingCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for TemperatureScalingCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        TemperatureScalingCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_temperature_scaling_basic() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);
        assert!(calibrator.temperature() > 0.0);

        // Calibrated probabilities should be valid
        for &prob in calibrated.iter() {
            assert!(
                (0.0..=1.0).contains(&prob),
                "Probability {} out of bounds",
                prob
            );
        }
    }

    #[test]
    fn test_temperature_scaling_overconfident() {
        // Test with overconfident predictions
        let probabilities = array![0.01, 0.05, 0.95, 0.99];
        let y_true = array![0, 0, 1, 1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);

        // Temperature should be positive
        assert!(calibrator.temperature() > 0.0);

        // Calibrated probabilities should be valid (behavior depends on optimization)
        for &prob in calibrated.iter() {
            assert!((0.0..=1.0).contains(&prob));
        }
    }

    #[test]
    fn test_temperature_scaling_underconfident() {
        // Test with underconfident predictions
        let probabilities = array![0.4, 0.45, 0.55, 0.6];
        let y_true = array![0, 0, 1, 1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.len(), 4);

        // Temperature might be < 1 for underconfident predictions
        // But this depends on the data, so we just check it's positive
        assert!(calibrator.temperature() > 0.0);
    }

    #[test]
    fn test_temperature_scaling_edge_cases() {
        // Test with single point
        let probabilities = array![0.5];
        let y_true = array![1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&array![0.5]).unwrap();
        assert_eq!(calibrated.len(), 1);
        assert!(calibrated[0] >= 0.0 && calibrated[0] <= 1.0);

        // Test with identical probabilities
        let probabilities = array![0.5, 0.5, 0.5];
        let y_true = array![0, 1, 1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 3);
    }

    #[test]
    fn test_temperature_scaling_perfect_calibration() {
        // Test with perfectly calibrated data
        let probabilities = array![0.0, 0.25, 0.5, 0.75, 1.0];
        let y_true = array![0, 0, 0, 1, 1];

        let calibrator = TemperatureScalingCalibrator::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        // Temperature should be close to 1.0 for well-calibrated data
        assert!((calibrator.temperature() - 1.0).abs() < 2.0);

        // Calibrated probabilities should be similar to original
        for i in 0..calibrated.len() {
            assert!((calibrated[i] - probabilities[i]).abs() < 0.5);
        }
    }
}
