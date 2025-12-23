//! Multi-class calibration methods
//!
//! This module provides specialized calibration methods for multi-class problems,
//! including one-vs-one calibration, multiclass temperature scaling, matrix scaling,
//! and Dirichlet calibration.

use crate::CalibrationEstimator;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// One-vs-One calibration for multi-class problems
///
/// For k classes, trains k(k-1)/2 binary calibrators between each pair of classes.
/// This can provide better calibration when classes are imbalanced.
#[derive(Debug, Clone)]
pub struct OneVsOneCalibrator {
    calibrators: Option<HashMap<(i32, i32), Box<dyn CalibrationEstimator>>>,
    classes: Option<Array1<i32>>,
}

impl OneVsOneCalibrator {
    /// Create a new one-vs-one calibrator
    pub fn new() -> Self {
        Self {
            calibrators: None,
            classes: None,
        }
    }

    /// Fit the one-vs-one calibrator
    pub fn fit(
        mut self,
        probabilities: &Array2<Float>,
        y_true: &Array1<i32>,
        base_calibrator_fn: Box<dyn Fn() -> Box<dyn CalibrationEstimator>>,
    ) -> Result<Self> {
        let mut classes: Vec<i32> = y_true
            .iter()
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        classes.sort();
        let n_classes = classes.len();

        if n_classes < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 classes for one-vs-one calibration".to_string(),
            ));
        }

        let mut calibrators = HashMap::new();

        // Train calibrator for each pair of classes
        for i in 0..n_classes {
            for j in (i + 1)..n_classes {
                let class_i = classes[i];
                let class_j = classes[j];

                // Create mask for samples belonging to either class_i or class_j
                let mask: Vec<usize> = y_true
                    .iter()
                    .enumerate()
                    .filter_map(|(idx, &y)| {
                        if y == class_i || y == class_j {
                            Some(idx)
                        } else {
                            None
                        }
                    })
                    .collect();

                if mask.len() < 2 {
                    continue; // Skip if not enough samples
                }

                // Extract relevant samples
                let y_binary: Array1<i32> = mask
                    .iter()
                    .map(|&idx| if y_true[idx] == class_i { 0 } else { 1 })
                    .collect::<Vec<_>>()
                    .into();

                // Get probabilities ratio for binary problem
                let prob_ratios: Array1<Float> = mask
                    .iter()
                    .map(|&idx| {
                        let p_i = probabilities[[idx, i]];
                        let p_j = probabilities[[idx, j]];
                        let sum = p_i + p_j;
                        if sum > 0.0 {
                            p_i / sum
                        } else {
                            0.5
                        }
                    })
                    .collect::<Vec<_>>()
                    .into();

                // Train binary calibrator
                let mut calibrator = base_calibrator_fn();
                calibrator.fit(&prob_ratios, &y_binary)?;

                calibrators.insert((class_i, class_j), calibrator);
            }
        }

        self.calibrators = Some(calibrators);
        self.classes = Some(Array1::from(classes));
        Ok(self)
    }

    /// Predict calibrated probabilities using one-vs-one approach
    pub fn predict_proba(&self, probabilities: &Array2<Float>) -> Result<Array2<Float>> {
        let calibrators = self
            .calibrators
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))?;
        let classes = self
            .classes
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))?;

        let (n_samples, n_features) = probabilities.dim();
        let n_classes = classes.len();

        if n_features != n_classes {
            return Err(SklearsError::InvalidInput(
                "Probability matrix dimension mismatch".to_string(),
            ));
        }

        let mut calibrated_probas = Array2::zeros((n_samples, n_classes));

        // For each sample, aggregate predictions from all binary calibrators
        for sample_idx in 0..n_samples {
            let mut votes: Array1<Float> = Array1::zeros(n_classes);

            for ((class_i, class_j), calibrator) in calibrators.iter() {
                let i = classes.iter().position(|&x| x == *class_i).unwrap();
                let j = classes.iter().position(|&x| x == *class_j).unwrap();

                let p_i = probabilities[[sample_idx, i]];
                let p_j = probabilities[[sample_idx, j]];
                let sum = p_i + p_j;

                if sum > 0.0 {
                    let prob_ratio = Array1::from(vec![p_i / sum]);
                    let calibrated_ratio = calibrator.predict_proba(&prob_ratio)?;

                    // Vote based on calibrated probability
                    if calibrated_ratio[0] > 0.5 {
                        votes[i] += 1.0;
                    } else {
                        votes[j] += 1.0;
                    }
                }
            }

            // Normalize votes to probabilities
            let total_votes: Float = votes.sum();
            if total_votes > 0.0 {
                calibrated_probas
                    .row_mut(sample_idx)
                    .assign(&(votes / total_votes));
            } else {
                // Fallback to uniform distribution
                calibrated_probas
                    .row_mut(sample_idx)
                    .fill(1.0 / n_classes as Float);
            }
        }

        Ok(calibrated_probas)
    }
}

impl Default for OneVsOneCalibrator {
    fn default() -> Self {
        Self::new()
    }
}

/// Multiclass Temperature Scaling
///
/// Applies a single temperature parameter to scale the entire logit vector
/// before applying softmax. This preserves the ranking while adjusting confidence.
#[derive(Debug, Clone)]
pub struct MulticlassTemperatureScaling {
    temperature: Float,
}

impl MulticlassTemperatureScaling {
    /// Create a new multiclass temperature scaling calibrator
    pub fn new() -> Self {
        Self { temperature: 1.0 }
    }

    /// Fit the temperature parameter using negative log-likelihood optimization
    pub fn fit(mut self, probabilities: &Array2<Float>, y_true: &Array1<i32>) -> Result<Self> {
        let (n_samples, _n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent number of samples".to_string(),
            ));
        }

        // Convert probabilities to logits
        let logits = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            clamped_p.ln()
        });

        // Find optimal temperature using simple grid search
        let mut best_temperature = 1.0;
        let mut best_nll = Float::INFINITY;

        for temp_candidate in [0.1, 0.2, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0] {
            let nll = self.compute_nll(&logits, y_true, temp_candidate)?;
            if nll < best_nll {
                best_nll = nll;
                best_temperature = temp_candidate;
            }
        }

        // Fine-tune with smaller steps around best candidate
        for i in 1..=10 {
            let offset = (i as Float - 5.0) * 0.1;
            let temp_candidate = (best_temperature + offset).max(0.01);
            let nll = self.compute_nll(&logits, y_true, temp_candidate)?;
            if nll < best_nll {
                best_nll = nll;
                best_temperature = temp_candidate;
            }
        }

        self.temperature = best_temperature;
        Ok(self)
    }

    /// Compute negative log-likelihood for given temperature
    fn compute_nll(
        &self,
        logits: &Array2<Float>,
        y_true: &Array1<i32>,
        temperature: Float,
    ) -> Result<Float> {
        let (n_samples, _) = logits.dim();
        let mut nll = 0.0;

        for i in 0..n_samples {
            let scaled_logits = logits.row(i).mapv(|x| x / temperature);

            // Compute log-softmax for numerical stability
            let max_logit = scaled_logits
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let log_sum_exp =
                (scaled_logits.mapv(|x| (x - max_logit).exp()).sum()).ln() + max_logit;

            let true_class_idx = y_true[i] as usize;
            if true_class_idx < scaled_logits.len() {
                let log_prob = scaled_logits[true_class_idx] - log_sum_exp;
                nll -= log_prob;
            }
        }

        Ok(nll / n_samples as Float)
    }

    /// Apply temperature scaling to probabilities
    pub fn predict_proba(&self, probabilities: &Array2<Float>) -> Result<Array2<Float>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut calibrated = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = probabilities.row(i);

            // Convert to logits
            let logits: Array1<Float> = row.mapv(|p| {
                let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                clamped_p.ln()
            });

            // Apply temperature scaling
            let scaled_logits = logits.mapv(|x| x / self.temperature);

            // Convert back to probabilities using softmax
            let max_logit = scaled_logits
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();

            calibrated.row_mut(i).assign(&(exp_logits / sum_exp));
        }

        Ok(calibrated)
    }

    /// Get the fitted temperature parameter
    pub fn temperature(&self) -> Float {
        self.temperature
    }
}

impl Default for MulticlassTemperatureScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Matrix Scaling for multiclass calibration
///
/// Learns a linear transformation W and bias b such that:
/// calibrated_logits = W * logits + b
/// This is more flexible than temperature scaling as it can learn different
/// scaling factors for different classes.
#[derive(Debug, Clone)]
pub struct MatrixScaling {
    weight_matrix: Option<Array2<Float>>,
    bias: Option<Array1<Float>>,
}

impl MatrixScaling {
    /// Create a new matrix scaling calibrator
    pub fn new() -> Self {
        Self {
            weight_matrix: None,
            bias: None,
        }
    }

    /// Fit the matrix scaling parameters
    pub fn fit(mut self, probabilities: &Array2<Float>, y_true: &Array1<i32>) -> Result<Self> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent number of samples".to_string(),
            ));
        }

        // Convert probabilities to logits
        let logits = probabilities.mapv(|p| {
            let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
            clamped_p.ln()
        });

        // Initialize weight matrix as identity and bias as zeros
        let mut weight_matrix = Array2::eye(n_classes);
        let mut bias = Array1::zeros(n_classes);

        // Simple iterative optimization (simplified implementation)
        // In practice, would use L-BFGS or similar optimizer
        let learning_rate = 0.01;
        let n_iterations = 100;

        for _iter in 0..n_iterations {
            let mut weight_grad = Array2::zeros((n_classes, n_classes));
            let mut bias_grad = Array1::zeros(n_classes);
            let mut _total_loss = 0.0;

            for i in 0..n_samples {
                let logit_row = logits.row(i);
                let true_class = y_true[i] as usize;

                // Forward pass
                let scaled_logits = weight_matrix.dot(&logit_row) + &bias;

                // Softmax
                let max_logit = scaled_logits
                    .iter()
                    .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
                let sum_exp = exp_logits.sum();
                let probs = exp_logits.mapv(|x| x / sum_exp);

                // Cross-entropy loss
                if true_class < probs.len() {
                    _total_loss -= probs[true_class].ln();

                    // Compute gradients
                    let mut target = Array1::zeros(n_classes);
                    target[true_class] = 1.0;
                    let error = probs - target;

                    // Gradient w.r.t. bias
                    bias_grad += &error;

                    // Gradient w.r.t. weight matrix
                    for j in 0..n_classes {
                        for k in 0..n_classes {
                            weight_grad[[j, k]] += error[j] * logit_row[k];
                        }
                    }
                }
            }

            // Update parameters
            weight_matrix = weight_matrix - learning_rate * weight_grad / n_samples as Float;
            bias = bias - learning_rate * bias_grad / n_samples as Float;
        }

        self.weight_matrix = Some(weight_matrix);
        self.bias = Some(bias);
        Ok(self)
    }

    /// Apply matrix scaling to probabilities
    pub fn predict_proba(&self, probabilities: &Array2<Float>) -> Result<Array2<Float>> {
        let weight_matrix = self
            .weight_matrix
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))?;
        let bias = self
            .bias
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))?;

        let (n_samples, n_classes) = probabilities.dim();
        let mut calibrated = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = probabilities.row(i);

            // Convert to logits
            let logits: Array1<Float> = row.mapv(|p| {
                let clamped_p = p.clamp(1e-15, 1.0 - 1e-15);
                clamped_p.ln()
            });

            // Apply matrix scaling
            let scaled_logits = weight_matrix.dot(&logits) + bias;

            // Convert back to probabilities using softmax
            let max_logit = scaled_logits
                .iter()
                .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits = scaled_logits.mapv(|x| (x - max_logit).exp());
            let sum_exp = exp_logits.sum();

            calibrated.row_mut(i).assign(&(exp_logits / sum_exp));
        }

        Ok(calibrated)
    }

    /// Get the fitted weight matrix
    pub fn weight_matrix(&self) -> Option<&Array2<Float>> {
        self.weight_matrix.as_ref()
    }

    /// Get the fitted bias vector
    pub fn bias(&self) -> Option<&Array1<Float>> {
        self.bias.as_ref()
    }
}

impl Default for MatrixScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Dirichlet Calibration for multiclass problems
///
/// Models the calibrated probabilities using a Dirichlet distribution.
/// This provides a principled Bayesian approach to multiclass calibration.
#[derive(Debug, Clone)]
pub struct DirichletCalibration {
    alpha_params: Option<Array1<Float>>,
    concentration: Float,
}

impl DirichletCalibration {
    /// Create a new Dirichlet calibration estimator
    pub fn new() -> Self {
        Self {
            alpha_params: None,
            concentration: 1.0,
        }
    }

    /// Set the concentration parameter
    pub fn concentration(mut self, concentration: Float) -> Self {
        self.concentration = concentration;
        self
    }

    /// Fit the Dirichlet calibration parameters
    pub fn fit(mut self, probabilities: &Array2<Float>, y_true: &Array1<i32>) -> Result<Self> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Inconsistent number of samples".to_string(),
            ));
        }

        // Estimate Dirichlet parameters using method of moments
        let mut alpha_params = Array1::ones(n_classes) * self.concentration;

        // Count class frequencies to estimate base rates
        let mut class_counts: Array1<Float> = Array1::zeros(n_classes);
        for &y in y_true.iter() {
            if (y as usize) < n_classes {
                class_counts[y as usize] += 1.0;
            }
        }

        // Normalize to get empirical class probabilities
        let total_count: Float = class_counts.sum();
        if total_count > 0.0 {
            class_counts /= total_count;
        }

        // Update alpha parameters based on empirical frequencies
        for i in 0..n_classes {
            if class_counts[i] > 0.0 {
                alpha_params[i] = self.concentration * class_counts[i] * n_classes as Float;
            }
        }

        // Ensure minimum concentration
        alpha_params.mapv_inplace(|x: Float| x.max(0.1));

        self.alpha_params = Some(alpha_params);
        Ok(self)
    }

    /// Apply Dirichlet calibration to probabilities
    pub fn predict_proba(&self, probabilities: &Array2<Float>) -> Result<Array2<Float>> {
        let alpha_params = self
            .alpha_params
            .as_ref()
            .ok_or_else(|| SklearsError::InvalidInput("Model not fitted".to_string()))?;

        let (n_samples, n_classes) = probabilities.dim();
        let mut calibrated = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = probabilities.row(i);

            // Apply Dirichlet transformation: p_calibrated = (p * alpha + 1) / (sum(p * alpha) + K)
            let weighted_probs = row.mapv(|p| p.max(1e-15)) * alpha_params;
            let normalizer = weighted_probs.sum() + n_classes as Float;

            let calibrated_row = (weighted_probs + 1.0) / normalizer;
            calibrated.row_mut(i).assign(&calibrated_row);
        }

        Ok(calibrated)
    }

    /// Get the fitted alpha parameters
    pub fn alpha_params(&self) -> Option<&Array1<Float>> {
        self.alpha_params.as_ref()
    }
}

impl Default for DirichletCalibration {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;
    use scirs2_core::ndarray::array;
    use scirs2_core::Axis;

    #[test]
    fn test_multiclass_temperature_scaling() {
        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        let calibrator = MulticlassTemperatureScaling::new()
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for row in calibrated.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Temperature should be positive
        assert!(calibrator.temperature() > 0.0);
    }

    #[test]
    fn test_matrix_scaling() {
        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        let calibrator = MatrixScaling::new().fit(&probabilities, &y_true).unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for row in calibrated.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that weight matrix and bias are fitted
        assert!(calibrator.weight_matrix().is_some());
        assert!(calibrator.bias().is_some());
    }

    #[test]
    fn test_dirichlet_calibration() {
        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        let calibrator = DirichletCalibration::new()
            .concentration(2.0)
            .fit(&probabilities, &y_true)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for row in calibrated.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }

        // Check that alpha parameters are fitted
        assert!(calibrator.alpha_params().is_some());
        let alphas = calibrator.alpha_params().unwrap();
        assert!(alphas.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn test_one_vs_one_calibrator() {
        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.1, 0.7],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        let calibrator_fn =
            Box::new(|| -> Box<dyn CalibrationEstimator> { Box::new(SigmoidCalibrator::new()) });

        let calibrator = OneVsOneCalibrator::new()
            .fit(&probabilities, &y_true, calibrator_fn)
            .unwrap();

        let calibrated = calibrator.predict_proba(&probabilities).unwrap();

        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for row in calibrated.axis_iter(Axis(0)) {
            let sum: Float = row.sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
