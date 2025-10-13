//! Beta calibration for probability calibration
//!
//! This module implements beta calibration, which uses a beta distribution
//! to model the calibration mapping. This is particularly useful when the
//! calibration relationship is non-monotonic.

use crate::CalibrationEstimator;
use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};

/// Beta calibration estimator
///
/// Beta calibration fits a beta distribution to map uncalibrated probabilities
/// to calibrated probabilities. The beta distribution is flexible and can
/// capture various calibration patterns.
#[derive(Debug, Clone)]
pub struct BetaCalibrator {
    alpha: Float,
    beta: Float,
    fitted: bool,
}

impl BetaCalibrator {
    /// Create a new beta calibrator
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
            fitted: false,
        }
    }

    /// Fit the beta calibrator using method of moments
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent array lengths".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(SklearsError::InvalidInput("Empty input arrays".to_string()));
        }

        // Compute residuals: difference between predicted and actual
        let residuals: Array1<Float> = probabilities
            .iter()
            .zip(y_true.iter())
            .map(|(&p, &y)| {
                let y_f = y as Float;
                // Compute calibration residual
                p - y_f
            })
            .collect::<Vec<_>>()
            .into();

        // Estimate beta distribution parameters using method of moments
        let mean_residual = residuals.mean().unwrap_or(0.0);
        let var_residual = {
            let squared_diffs: Array1<Float> = residuals.mapv(|r| (r - mean_residual).powi(2));
            squared_diffs.mean().unwrap_or(1.0)
        };

        // Convert residuals to [0,1] range for beta distribution
        let min_residual = residuals.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_residual = residuals.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let range = max_residual - min_residual;

        if range <= 0.0 {
            // All residuals are the same, use default parameters
            self.alpha = 1.0;
            self.beta = 1.0;
        } else {
            // Normalize residuals to [0,1]
            let normalized_residuals: Array1<Float> =
                residuals.mapv(|r| (r - min_residual) / range);

            let norm_mean = normalized_residuals.mean().unwrap_or(0.5);
            let norm_var = {
                let squared_diffs: Array1<Float> =
                    normalized_residuals.mapv(|r| (r - norm_mean).powi(2));
                squared_diffs.mean().unwrap_or(0.25)
            };

            // Method of moments for beta distribution
            if norm_var > 0.0 && norm_mean > 0.0 && norm_mean < 1.0 {
                let common_factor = norm_mean * (1.0 - norm_mean) / norm_var - 1.0;
                self.alpha = (norm_mean * common_factor).max(0.1);
                self.beta = ((1.0 - norm_mean) * common_factor).max(0.1);
            } else {
                // Fallback to default parameters
                self.alpha = 1.0;
                self.beta = 1.0;
            }
        }

        self.fitted = true;
        Ok(self)
    }

    /// Predict calibrated probabilities using beta distribution
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Calibrator not fitted".to_string(),
            ));
        }

        // Apply beta calibration transformation
        let calibrated = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);

            // Beta distribution CDF approximation
            // For computational efficiency, use a simple transformation
            let transformed = self.beta_cdf_approx(clamped_p);
            transformed.clamp(0.0, 1.0)
        });

        Ok(calibrated)
    }

    /// Approximate beta distribution CDF
    fn beta_cdf_approx(&self, x: Float) -> Float {
        // Simplified beta CDF using power transformation
        // More sophisticated implementations would use incomplete beta function
        let power_alpha = x.powf(self.alpha - 1.0);
        let power_beta = (1.0 - x).powf(self.beta - 1.0);

        // Normalize using beta function approximation
        let beta_func_approx = self.gamma_approx(self.alpha) * self.gamma_approx(self.beta)
            / self.gamma_approx(self.alpha + self.beta);

        let unnormalized = power_alpha * power_beta;
        let normalized = unnormalized / beta_func_approx;

        // Apply cumulative transformation (simplified)
        let transformed = x * (1.0 + normalized.ln().abs() * 0.1);
        transformed.clamp(0.0, 1.0)
    }

    /// Approximate gamma function using Stirling's approximation
    fn gamma_approx(&self, z: Float) -> Float {
        if z <= 0.0 {
            return 1.0;
        }

        if z < 1.0 {
            // Use recurrence relation: Γ(z) = Γ(z+1) / z
            return self.gamma_approx(z + 1.0) / z;
        }

        // Stirling's approximation: Γ(z) ≈ √(2π/z) * (z/e)^z
        let sqrt_two_pi = (2.0 * std::f64::consts::PI).sqrt() as Float;
        let e = std::f64::consts::E as Float;

        (sqrt_two_pi / z.sqrt()) * (z / e).powf(z)
    }

    /// Get the fitted alpha parameter
    pub fn alpha(&self) -> Float {
        self.alpha
    }

    /// Get the fitted beta parameter
    pub fn beta(&self) -> Float {
        self.beta
    }

    /// Check if the calibrator is fitted
    pub fn is_fitted(&self) -> bool {
        self.fitted
    }
}

impl Default for BetaCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for BetaCalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        BetaCalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Ensemble Temperature Scaling
///
/// Combines multiple temperature scaling calibrators trained on different
/// subsets of the data to improve calibration robustness.
#[derive(Debug, Clone)]
pub struct EnsembleTemperatureScaling {
    temperatures: Vec<Float>,
    weights: Vec<Float>,
    n_estimators: usize,
}

impl EnsembleTemperatureScaling {
    /// Create a new ensemble temperature scaling calibrator
    pub fn new(n_estimators: usize) -> Self {
        Self {
            temperatures: Vec::new(),
            weights: Vec::new(),
            n_estimators: n_estimators.max(1),
        }
    }

    /// Fit ensemble of temperature scaling calibrators
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent array lengths".to_string(),
            ));
        }

        let n_samples = probabilities.len();
        if n_samples < self.n_estimators {
            return Err(SklearsError::InvalidInput(
                "Not enough samples for ensemble".to_string(),
            ));
        }

        let subset_size = n_samples / self.n_estimators;
        self.temperatures.clear();
        self.weights.clear();

        // Train temperature scalers on different subsets
        for i in 0..self.n_estimators {
            let start_idx = i * subset_size;
            let end_idx = if i == self.n_estimators - 1 {
                n_samples
            } else {
                (i + 1) * subset_size
            };

            // Extract subset
            let subset_probs: Array1<Float> =
                probabilities.slice(s![start_idx..end_idx]).to_owned();
            let subset_y: Array1<i32> = y_true.slice(s![start_idx..end_idx]).to_owned();

            // Fit temperature on subset
            let temperature = self.fit_single_temperature(&subset_probs, &subset_y)?;

            // Compute weight based on subset performance
            let weight = self.compute_weight(&subset_probs, &subset_y, temperature)?;

            self.temperatures.push(temperature);
            self.weights.push(weight);
        }

        // Normalize weights
        let total_weight: Float = self.weights.iter().sum();
        if total_weight > 0.0 {
            for weight in &mut self.weights {
                *weight /= total_weight;
            }
        } else {
            // Uniform weights as fallback
            let uniform_weight = 1.0 / self.n_estimators as Float;
            self.weights.fill(uniform_weight);
        }

        Ok(self)
    }

    /// Fit a single temperature parameter
    fn fit_single_temperature(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
    ) -> Result<Float> {
        // Convert to binary problem for temperature scaling
        let y_binary: Array1<Float> = y_true.mapv(|y| y as Float);

        let mut best_temperature = 1.0;
        let mut best_nll = Float::INFINITY;

        // Grid search for optimal temperature
        for temp_candidate in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0] {
            let nll = self.compute_binary_nll(probabilities, &y_binary, temp_candidate)?;
            if nll < best_nll {
                best_nll = nll;
                best_temperature = temp_candidate;
            }
        }

        Ok(best_temperature)
    }

    /// Compute negative log-likelihood for binary case
    fn compute_binary_nll(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<Float>,
        temperature: Float,
    ) -> Result<Float> {
        let n_samples = probabilities.len();
        let mut nll = 0.0;

        for i in 0..n_samples {
            let p = probabilities[i].clamp(1e-15, 1.0 - 1e-15);
            let logit = (p / (1.0 - p)).ln();
            let scaled_logit = logit / temperature;
            let calibrated_p = 1.0 / (1.0 + (-scaled_logit).exp());

            let y = y_true[i];
            let log_likelihood = y * calibrated_p.ln() + (1.0 - y) * (1.0 - calibrated_p).ln();
            nll -= log_likelihood;
        }

        Ok(nll / n_samples as Float)
    }

    /// Compute weight for ensemble member based on performance
    fn compute_weight(
        &self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        temperature: Float,
    ) -> Result<Float> {
        let y_binary: Array1<Float> = y_true.mapv(|y| y as Float);
        let nll = self.compute_binary_nll(probabilities, &y_binary, temperature)?;

        // Convert NLL to weight (lower NLL = higher weight)
        let weight = (-nll).exp();
        Ok(weight)
    }

    /// Predict calibrated probabilities using ensemble
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        if self.temperatures.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Ensemble not fitted".to_string(),
            ));
        }

        let mut ensemble_probs = Array1::zeros(probabilities.len());

        // Combine predictions from all temperature scalers
        for (temp, weight) in self.temperatures.iter().zip(self.weights.iter()) {
            let calibrated = probabilities.mapv(|p| {
                let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                let logit = (clamped_p / (1.0 - clamped_p)).ln();
                let scaled_logit = logit / temp;
                1.0 / (1.0 + (-scaled_logit).exp())
            });

            ensemble_probs = ensemble_probs + *weight * calibrated;
        }

        Ok(ensemble_probs)
    }

    /// Get the fitted temperatures
    pub fn temperatures(&self) -> &[Float] {
        &self.temperatures
    }

    /// Get the ensemble weights
    pub fn weights(&self) -> &[Float] {
        &self.weights
    }
}

impl Default for EnsembleTemperatureScaling {
    fn default() -> Self {
        Self::new(5)
    }
}

impl CalibrationEstimator for EnsembleTemperatureScaling {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = Self::new(self.n_estimators).fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        EnsembleTemperatureScaling::predict_proba(self, probabilities)
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
    fn test_beta_calibrator() {
        let probabilities = array![0.1, 0.3, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1];

        let calibrator = BetaCalibrator::new().fit(&probabilities, &y_true).unwrap();

        assert!(calibrator.is_fitted());
        assert!(calibrator.alpha() > 0.0);
        assert!(calibrator.beta() > 0.0);

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 4);

        // Check that calibrated probabilities are valid
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_ensemble_temperature_scaling() {
        let probabilities = array![0.1, 0.2, 0.3, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 1, 1, 1, 1];

        let calibrator = EnsembleTemperatureScaling::new(3)
            .fit(&probabilities, &y_true)
            .unwrap();

        assert_eq!(calibrator.temperatures().len(), 3);
        assert_eq!(calibrator.weights().len(), 3);

        // Check that weights sum to 1
        let weight_sum: Float = calibrator.weights().iter().sum();
        assert!((weight_sum - 1.0).abs() < 1e-6);

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 7);

        // Check that calibrated probabilities are valid
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_beta_calibrator_edge_cases() {
        // Test with uniform probabilities
        let probabilities = array![0.5, 0.5, 0.5, 0.5];
        let y_true = array![0, 1, 0, 1];

        let calibrator = BetaCalibrator::new().fit(&probabilities, &y_true).unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();
        assert_eq!(calibrated.len(), 4);

        // Test with extreme probabilities
        let extreme_probs = array![0.01, 0.99];
        let extreme_y = array![0, 1];

        let extreme_calibrator = BetaCalibrator::new()
            .fit(&extreme_probs, &extreme_y)
            .unwrap();

        let extreme_calibrated = extreme_calibrator.predict_proba(&extreme_probs).unwrap();
        assert_eq!(extreme_calibrated.len(), 2);
    }
}
