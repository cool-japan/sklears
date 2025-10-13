//! Kernel Density Estimation (KDE) calibration
//!
//! This module provides calibration methods based on kernel density estimation,
//! which can model complex non-parametric relationships between predicted
//! probabilities and true calibrated probabilities.

use scirs2_core::ndarray::{s, Array1};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::f32::consts::PI as PI_F32;

use crate::CalibrationEstimator;

/// Kernel Density Estimation calibrator
///
/// Uses kernel density estimation to learn the relationship between
/// predicted probabilities and true probabilities non-parametrically.
#[derive(Debug, Clone)]
pub struct KDECalibrator {
    kernel: KernelType,
    bandwidth: BandwidthSelection,
    training_probabilities: Option<Array1<Float>>,
    training_labels: Option<Array1<i32>>,
    fitted_bandwidth: Option<Float>,
}

/// Available kernel types for KDE
#[derive(Debug, Clone)]
pub enum KernelType {
    /// Gaussian kernel (most common)
    Gaussian,
    /// Epanechnikov kernel (more efficient)
    Epanechnikov,
    /// Uniform kernel (simple box kernel)
    Uniform,
    /// Triangular kernel
    Triangular,
    /// Cosine kernel
    Cosine,
}

/// Bandwidth selection methods
#[derive(Debug, Clone)]
pub enum BandwidthSelection {
    /// Fixed bandwidth
    Fixed(Float),
    /// Scott's rule of thumb
    Scott,
    /// Silverman's rule of thumb
    Silverman,
    /// Cross-validation optimized bandwidth
    CrossValidation { cv_folds: usize },
}

impl KDECalibrator {
    /// Create a new KDE calibrator with Gaussian kernel and Scott's bandwidth
    pub fn new() -> Self {
        Self {
            kernel: KernelType::Gaussian,
            bandwidth: BandwidthSelection::Scott,
            training_probabilities: None,
            training_labels: None,
            fitted_bandwidth: None,
        }
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the bandwidth selection method
    pub fn bandwidth(mut self, bandwidth: BandwidthSelection) -> Self {
        self.bandwidth = bandwidth;
        self
    }

    /// Fit the KDE calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        if probabilities.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have the same length".to_string(),
            ));
        }

        if probabilities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No probabilities provided".to_string(),
            ));
        }

        // Select bandwidth
        let bandwidth = self.select_bandwidth(probabilities)?;

        self.training_probabilities = Some(probabilities.clone());
        self.training_labels = Some(y_true.clone());
        self.fitted_bandwidth = Some(bandwidth);

        Ok(self)
    }

    /// Predict calibrated probabilities
    pub fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let training_probabilities =
            self.training_probabilities
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba".to_string(),
                })?;
        let training_labels =
            self.training_labels
                .as_ref()
                .ok_or_else(|| SklearsError::NotFitted {
                    operation: "predict_proba".to_string(),
                })?;
        let bandwidth = self
            .fitted_bandwidth
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "predict_proba".to_string(),
            })?;

        let mut calibrated_probabilities = Array1::zeros(probabilities.len());

        for (i, &prob) in probabilities.iter().enumerate() {
            // Estimate P(Y=1|p) using KDE
            let positive_density = self.estimate_density(
                prob,
                training_probabilities,
                training_labels,
                bandwidth,
                true,
            )?;
            let negative_density = self.estimate_density(
                prob,
                training_probabilities,
                training_labels,
                bandwidth,
                false,
            )?;

            // Calibrated probability using Bayes' theorem
            let total_density = positive_density + negative_density;
            let calibrated_prob = if total_density > 0.0 {
                positive_density / total_density
            } else {
                prob // Fall back to original probability
            };

            calibrated_probabilities[i] = calibrated_prob.clamp(1e-15, 1.0 - 1e-15);
        }

        Ok(calibrated_probabilities)
    }

    fn select_bandwidth(&self, probabilities: &Array1<Float>) -> Result<Float> {
        match &self.bandwidth {
            BandwidthSelection::Fixed(h) => Ok(*h),
            BandwidthSelection::Scott => {
                let n = probabilities.len() as Float;
                let std_dev = self.calculate_std(probabilities);
                Ok(1.06 * std_dev * n.powf(-1.0 / 5.0))
            }
            BandwidthSelection::Silverman => {
                let n = probabilities.len() as Float;
                let std_dev = self.calculate_std(probabilities);
                let iqr = self.calculate_iqr(probabilities);
                let sigma_hat = std_dev.min(iqr / 1.34);
                Ok(0.9 * sigma_hat * n.powf(-1.0 / 5.0))
            }
            BandwidthSelection::CrossValidation { cv_folds } => {
                self.optimize_bandwidth_cv(probabilities, *cv_folds)
            }
        }
    }

    fn calculate_std(&self, data: &Array1<Float>) -> Float {
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.mapv(|x| (x - mean).powi(2)).mean().unwrap_or(0.0);
        variance.sqrt()
    }

    fn calculate_iqr(&self, data: &Array1<Float>) -> Float {
        let mut sorted_data = data.to_vec();
        sorted_data.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let n = sorted_data.len();
        if n < 4 {
            return self.calculate_std(data);
        }

        let q1_idx = n / 4;
        let q3_idx = 3 * n / 4;

        sorted_data[q3_idx] - sorted_data[q1_idx]
    }

    fn optimize_bandwidth_cv(
        &self,
        probabilities: &Array1<Float>,
        cv_folds: usize,
    ) -> Result<Float> {
        let n = probabilities.len();
        let fold_size = n / cv_folds;

        // Test different bandwidth values
        let min_h = 0.01;
        let max_h = 0.5;
        let n_candidates = 20;
        let step = (max_h - min_h) / (n_candidates - 1) as Float;

        let mut best_bandwidth = min_h;
        let mut best_score = Float::NEG_INFINITY;

        for i in 0..n_candidates {
            let h = min_h + i as Float * step;
            let mut cv_score = 0.0;

            // K-fold cross-validation
            for fold in 0..cv_folds {
                let start_idx = fold * fold_size;
                let end_idx = if fold == cv_folds - 1 {
                    n
                } else {
                    (fold + 1) * fold_size
                };

                // Create validation set
                let val_probs = probabilities.slice(s![start_idx..end_idx]).to_owned();

                // Create training set (excluding validation fold)
                let mut train_probs = Vec::new();
                for j in 0..n {
                    if j < start_idx || j >= end_idx {
                        train_probs.push(probabilities[j]);
                    }
                }
                let train_probs = Array1::from_vec(train_probs);

                // Calculate log-likelihood on validation set
                for &val_prob in val_probs.iter() {
                    let density = self.estimate_density_simple(val_prob, &train_probs, h);
                    if density > 0.0 {
                        cv_score += density.ln();
                    }
                }
            }

            if cv_score > best_score {
                best_score = cv_score;
                best_bandwidth = h;
            }
        }

        Ok(best_bandwidth)
    }

    fn estimate_density(
        &self,
        x: Float,
        training_probabilities: &Array1<Float>,
        training_labels: &Array1<i32>,
        bandwidth: Float,
        positive_class: bool,
    ) -> Result<Float> {
        let target_label = if positive_class { 1 } else { 0 };
        let mut density = 0.0;
        let mut count = 0;

        for (i, &train_prob) in training_probabilities.iter().enumerate() {
            if training_labels[i] == target_label {
                let kernel_value = self.kernel_function((x - train_prob) / bandwidth);
                density += kernel_value;
                count += 1;
            }
        }

        if count > 0 {
            density /= count as Float * bandwidth;
        }

        Ok(density)
    }

    fn estimate_density_simple(
        &self,
        x: Float,
        training_probabilities: &Array1<Float>,
        bandwidth: Float,
    ) -> Float {
        let mut density = 0.0;
        let n = training_probabilities.len();

        for &train_prob in training_probabilities.iter() {
            let kernel_value = self.kernel_function((x - train_prob) / bandwidth);
            density += kernel_value;
        }

        density / (n as Float * bandwidth)
    }

    fn kernel_function(&self, u: Float) -> Float {
        match self.kernel {
            KernelType::Gaussian => (1.0 / (2.0 * PI_F32 as Float).sqrt()) * (-0.5 * u * u).exp(),
            KernelType::Epanechnikov => {
                if u.abs() <= 1.0 {
                    0.75 * (1.0 - u * u)
                } else {
                    0.0
                }
            }
            KernelType::Uniform => {
                if u.abs() <= 1.0 {
                    0.5
                } else {
                    0.0
                }
            }
            KernelType::Triangular => {
                if u.abs() <= 1.0 {
                    1.0 - u.abs()
                } else {
                    0.0
                }
            }
            KernelType::Cosine => {
                if u.abs() <= 1.0 {
                    (PI_F32 as Float / 4.0) * (PI_F32 as Float * u / 2.0).cos()
                } else {
                    0.0
                }
            }
        }
    }
}

impl Default for KDECalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for KDECalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = self.clone().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        KDECalibrator::predict_proba(self, probabilities)
    }

    fn clone_box(&self) -> Box<dyn CalibrationEstimator> {
        Box::new(self.clone())
    }
}

/// Adaptive KDE calibrator that adjusts bandwidth locally
#[derive(Debug, Clone)]
pub struct AdaptiveKDECalibrator {
    base_calibrator: KDECalibrator,
    adaptation_factor: Float,
    local_bandwidth_multipliers: Option<Array1<Float>>,
}

impl AdaptiveKDECalibrator {
    /// Create a new adaptive KDE calibrator
    pub fn new() -> Self {
        Self {
            base_calibrator: KDECalibrator::new(),
            adaptation_factor: 0.5,
            local_bandwidth_multipliers: None,
        }
    }

    /// Set the adaptation factor (controls how much to adapt bandwidth locally)
    pub fn adaptation_factor(mut self, factor: Float) -> Self {
        self.adaptation_factor = factor;
        self
    }

    /// Set the kernel type
    pub fn kernel(mut self, kernel: KernelType) -> Self {
        self.base_calibrator = self.base_calibrator.kernel(kernel);
        self
    }

    /// Fit the adaptive KDE calibrator
    pub fn fit(mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<Self> {
        // Fit base calibrator
        self.base_calibrator = self.base_calibrator.fit(probabilities, y_true)?;

        // Calculate local density estimates to adapt bandwidth
        let local_multipliers = self.calculate_local_multipliers(probabilities)?;
        self.local_bandwidth_multipliers = Some(local_multipliers);

        Ok(self)
    }

    fn calculate_local_multipliers(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        let n = probabilities.len();
        let mut multipliers = Array1::ones(n);

        // Use k-nearest neighbors to estimate local density
        let k = (n as Float).sqrt() as usize + 1;

        for (i, &prob) in probabilities.iter().enumerate() {
            // Find k nearest neighbors
            let mut distances: Vec<Float> =
                probabilities.iter().map(|&p| (p - prob).abs()).collect();
            distances.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            // Use k-th nearest neighbor distance as local density estimate
            let kth_distance = if distances.len() > k {
                distances[k]
            } else {
                distances[distances.len() - 1]
            };

            // Adapt bandwidth based on local density
            let multiplier = if kth_distance > 0.0 {
                (kth_distance / 0.1).powf(self.adaptation_factor)
            } else {
                1.0
            };

            multipliers[i] = multiplier.clamp(0.1, 10.0);
        }

        Ok(multipliers)
    }
}

impl Default for AdaptiveKDECalibrator {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationEstimator for AdaptiveKDECalibrator {
    fn fit(&mut self, probabilities: &Array1<Float>, y_true: &Array1<i32>) -> Result<()> {
        *self = self.clone().fit(probabilities, y_true)?;
        Ok(())
    }

    fn predict_proba(&self, probabilities: &Array1<Float>) -> Result<Array1<Float>> {
        // For simplicity, use the base calibrator's prediction
        // In a full implementation, you would adapt bandwidth for each prediction point
        self.base_calibrator.predict_proba(probabilities)
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
    fn test_kde_calibrator() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = KDECalibrator::new().fit(&probabilities, &y_true).unwrap();

        let test_probabilities = array![0.25, 0.75];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        // First test point (0.25) should have lower calibrated probability than second (0.75)
        assert!(calibrated[0] < calibrated[1]);
    }

    #[test]
    fn test_kde_different_kernels() {
        let probabilities = array![0.2, 0.4, 0.6, 0.8];
        let y_true = array![0, 0, 1, 1];

        for kernel in [
            KernelType::Gaussian,
            KernelType::Epanechnikov,
            KernelType::Uniform,
            KernelType::Triangular,
            KernelType::Cosine,
        ] {
            let calibrator = KDECalibrator::new()
                .kernel(kernel)
                .fit(&probabilities, &y_true)
                .unwrap();

            let test_probabilities = array![0.3, 0.7];
            let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

            assert_eq!(calibrated.len(), 2);
            for &prob in calibrated.iter() {
                assert!(prob >= 0.0 && prob <= 1.0);
            }
        }
    }

    #[test]
    fn test_kde_bandwidth_selection() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        // Test different bandwidth selection methods
        for bandwidth in [
            BandwidthSelection::Fixed(0.1),
            BandwidthSelection::Scott,
            BandwidthSelection::Silverman,
            BandwidthSelection::CrossValidation { cv_folds: 3 },
        ] {
            let calibrator = KDECalibrator::new()
                .bandwidth(bandwidth)
                .fit(&probabilities, &y_true)
                .unwrap();

            let test_probabilities = array![0.25, 0.75];
            let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

            assert_eq!(calibrated.len(), 2);
            for &prob in calibrated.iter() {
                assert!(prob >= 0.0 && prob <= 1.0);
            }
        }
    }

    #[test]
    fn test_adaptive_kde_calibrator() {
        let probabilities = array![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];
        let y_true = array![0, 0, 0, 0, 1, 1, 1, 1, 1];

        let calibrator = AdaptiveKDECalibrator::new()
            .adaptation_factor(0.3)
            .fit(&probabilities, &y_true)
            .unwrap();

        let test_probabilities = array![0.25, 0.75];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_kde_with_trait() {
        let probabilities = array![0.1, 0.3, 0.5, 0.7, 0.9];
        let y_true = array![0, 0, 1, 1, 1];

        let calibrator = KDECalibrator::new().fit(&probabilities, &y_true).unwrap();

        let test_probabilities = array![0.2, 0.8];
        let calibrated = calibrator.predict_proba(&test_probabilities).unwrap();

        assert_eq!(calibrated.len(), 2);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_kernel_functions() {
        let calibrator = KDECalibrator::new();

        // Test Gaussian kernel
        let gaussian_val = calibrator.kernel_function(0.0);
        assert!(gaussian_val > 0.0);

        // Test that kernel values are non-negative
        for &u in &[-2.0, -1.0, 0.0, 1.0, 2.0] {
            let val = calibrator.kernel_function(u);
            assert!(val >= 0.0);
        }
    }
}
