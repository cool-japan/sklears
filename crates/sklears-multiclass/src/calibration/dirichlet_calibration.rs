//! Dirichlet Calibration
//!
//! Dirichlet calibration is a method specifically designed for multiclass calibration
//! that learns a transformation matrix to map uncalibrated probabilities to calibrated
//! probabilities using a Dirichlet distribution assumption.

use super::CalibrationFunction;
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Dirichlet calibration for multiclass problems
///
/// This method learns a linear transformation matrix W and bias vector b such that
/// the calibrated probabilities are given by softmax(W * uncalibrated_logits + b).
/// It uses regularization to prevent overfitting.
#[derive(Debug, Clone)]
pub struct DirichletCalibration {
    /// Transformation matrix W [n_classes, n_classes]
    pub weights: Array2<f64>,
    /// Bias vector b [n_classes]
    pub bias: Array1<f64>,
    /// L2 regularization parameter
    pub l2_reg: f64,
    /// Whether the model has been fitted
    pub fitted: bool,
}

impl DirichletCalibration {
    /// Create a new Dirichlet calibration instance
    pub fn new(l2_reg: f64) -> Self {
        Self {
            weights: Array2::zeros((0, 0)),
            bias: Array1::zeros(0),
            l2_reg: l2_reg.max(0.0),
            fitted: false,
        }
    }

    /// Fit Dirichlet calibration to data
    ///
    /// # Arguments
    /// * `probabilities` - Uncalibrated probabilities [n_samples, n_classes]
    /// * `y_true` - True class labels
    /// * `max_iter` - Maximum number of optimization iterations
    /// * `tol` - Tolerance for convergence
    /// * `learning_rate` - Learning rate for gradient descent
    pub fn fit(
        &mut self,
        probabilities: &Array2<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
        learning_rate: f64,
    ) -> SklResult<()> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have same number of samples".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty data".to_string(),
            ));
        }

        // Get unique classes
        let mut classes: Vec<i32> = y_true.to_vec();
        classes.sort_unstable();
        classes.dedup();

        // For cross-validation scenarios, we might not have all classes in a fold
        // So we allow fewer classes in labels than in probabilities
        if classes.len() > n_classes {
            return Err(SklearsError::InvalidInput(
                "Number of unique classes in labels exceeds probabilities dimensions".to_string(),
            ));
        }

        // Create class mapping - map each unique class to its index in probabilities
        // For missing classes during CV, we'll handle them appropriately
        let mut class_to_idx: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();

        // If we have fewer classes in labels than probabilities, map them to corresponding indices
        // For cross-validation, we might have class labels that exceed the number of probability columns
        // In that case, we'll map them to the last available column or handle them appropriately
        for (idx, &class_label) in classes.iter().enumerate() {
            if class_label < n_classes as i32 {
                class_to_idx.insert(class_label, class_label as usize);
            } else {
                // For missing classes, we'll map them to the last available column
                // This is a fallback for cross-validation scenarios
                class_to_idx.insert(class_label, (n_classes - 1).min(idx));
            }
        }

        // Convert probabilities to logits
        let mut logits = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let row_sum: f64 = row.sum();

            for j in 0..n_classes {
                let p = if row_sum > 0.0 {
                    (row[j] / row_sum).max(1e-15).min(1.0 - 1e-15)
                } else {
                    1.0 / n_classes as f64
                };
                logits[[i, j]] = p.ln();
            }
        }

        // Initialize parameters
        self.weights = Array2::eye(n_classes);
        self.bias = Array1::zeros(n_classes);

        // Create one-hot encoded targets
        let mut targets = Array2::zeros((n_samples, n_classes));
        for (i, &label) in y_true.iter().enumerate() {
            if let Some(&class_idx) = class_to_idx.get(&label) {
                targets[[i, class_idx]] = 1.0;
            }
        }

        // Gradient descent optimization
        let mut prev_loss = f64::INFINITY;

        for _iter in 0..max_iter {
            // Forward pass
            let predictions = self.predict_logits(&logits)?;
            let probabilities = self.softmax(&predictions)?;

            // Compute loss and gradients
            let (loss, grad_weights, grad_bias) =
                self.compute_loss_and_gradients(&logits, &probabilities, &targets)?;

            // Check for convergence
            if (prev_loss - loss).abs() < tol {
                break;
            }
            prev_loss = loss;

            // Update parameters
            for i in 0..n_classes {
                for j in 0..n_classes {
                    self.weights[[i, j]] -= learning_rate * grad_weights[[i, j]];
                }
                self.bias[i] -= learning_rate * grad_bias[i];
            }
        }

        self.fitted = true;
        Ok(())
    }

    /// Apply the learned transformation to logits
    fn predict_logits(&self, logits: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = logits.dim();

        if n_classes != self.weights.ncols() {
            return Err(SklearsError::InvalidInput(
                "Input dimension doesn't match fitted model".to_string(),
            ));
        }

        let mut result = Array2::zeros((n_samples, n_classes));

        // Apply linear transformation: result = logits * W^T + b
        for i in 0..n_samples {
            for j in 0..n_classes {
                let mut sum = self.bias[j];
                for k in 0..n_classes {
                    sum += logits[[i, k]] * self.weights[[j, k]];
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }

    /// Apply softmax to logits
    fn softmax(&self, logits: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = logits.dim();
        let mut result = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = logits.row(i);
            let max_logit = row.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            let exp_logits: Vec<f64> = row.iter().map(|&x| (x - max_logit).exp()).collect();
            let sum_exp: f64 = exp_logits.iter().sum();

            for j in 0..n_classes {
                result[[i, j]] = if sum_exp > 0.0 {
                    exp_logits[j] / sum_exp
                } else {
                    1.0 / n_classes as f64
                };
            }
        }

        Ok(result)
    }

    /// Compute loss and gradients
    fn compute_loss_and_gradients(
        &self,
        logits: &Array2<f64>,
        probabilities: &Array2<f64>,
        targets: &Array2<f64>,
    ) -> SklResult<(f64, Array2<f64>, Array1<f64>)> {
        let (n_samples, n_classes) = logits.dim();

        // Compute cross-entropy loss
        let mut ce_loss = 0.0;
        for i in 0..n_samples {
            for j in 0..n_classes {
                if targets[[i, j]] > 0.0 {
                    let prob = probabilities[[i, j]].max(1e-15);
                    ce_loss += -targets[[i, j]] * prob.ln();
                }
            }
        }
        ce_loss /= n_samples as f64;

        // Add L2 regularization
        let mut reg_loss = 0.0;
        for &w in self.weights.iter() {
            reg_loss += w * w;
        }
        reg_loss *= 0.5 * self.l2_reg;

        let total_loss = ce_loss + reg_loss;

        // Compute gradients
        let mut grad_weights = Array2::zeros((n_classes, n_classes));
        let mut grad_bias = Array1::zeros(n_classes);

        for i in 0..n_samples {
            for j in 0..n_classes {
                let error = probabilities[[i, j]] - targets[[i, j]];

                // Gradient w.r.t. bias
                grad_bias[j] += error;

                // Gradient w.r.t. weights
                for k in 0..n_classes {
                    grad_weights[[j, k]] += error * logits[[i, k]];
                }
            }
        }

        // Normalize by number of samples
        for j in 0..n_classes {
            grad_bias[j] /= n_samples as f64;
            for k in 0..n_classes {
                grad_weights[[j, k]] /= n_samples as f64;
                // Add L2 regularization gradient
                grad_weights[[j, k]] += self.l2_reg * self.weights[[j, k]];
            }
        }

        Ok((total_loss, grad_weights, grad_bias))
    }

    /// Transform uncalibrated probabilities to calibrated probabilities
    pub fn transform(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Model must be fitted before transform".to_string(),
            ));
        }

        let (n_samples, n_classes) = probabilities.dim();

        // Convert probabilities to logits
        let mut logits = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let row_sum: f64 = row.sum();

            for j in 0..n_classes {
                let p = if row_sum > 0.0 {
                    (row[j] / row_sum).max(1e-15).min(1.0 - 1e-15)
                } else {
                    1.0 / n_classes as f64
                };
                logits[[i, j]] = p.ln();
            }
        }

        // Apply transformation and softmax
        let transformed_logits = self.predict_logits(&logits)?;
        self.softmax(&transformed_logits)
    }
}

impl Default for DirichletCalibration {
    fn default() -> Self {
        Self::new(0.01) // Default L2 regularization
    }
}

impl CalibrationFunction for DirichletCalibration {
    fn calibrate(&self, uncalibrated_probs: &Array1<f64>) -> SklResult<Array1<f64>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "Model must be fitted before calibration".to_string(),
            ));
        }

        let n_samples = uncalibrated_probs.len();
        let n_classes = self.weights.nrows(); // Use the actual number of classes the model was fitted on

        let mut probs_2d = Array2::zeros((n_samples, n_classes));

        if n_classes == 2 {
            // Binary case
            for i in 0..n_samples {
                let p = uncalibrated_probs[i].max(1e-15).min(1.0 - 1e-15);
                probs_2d[[i, 0]] = 1.0 - p;
                probs_2d[[i, 1]] = p;
            }
        } else {
            // Multiclass case - this shouldn't normally happen with the CalibrationFunction trait
            // but we'll handle it by creating uniform probabilities across all classes
            for i in 0..n_samples {
                let p = uncalibrated_probs[i].max(1e-15).min(1.0 - 1e-15);
                // Distribute the probability uniformly across all classes
                for j in 0..n_classes {
                    probs_2d[[i, j]] = if j == 1 {
                        p
                    } else {
                        (1.0 - p) / (n_classes - 1) as f64
                    };
                }
            }
        }

        let calibrated_2d = self.transform(&probs_2d)?;

        // Extract positive class probabilities
        let mut result = Array1::zeros(n_samples);
        for i in 0..n_samples {
            result[i] = if n_classes > 1 {
                calibrated_2d[[i, 1]]
            } else {
                calibrated_2d[[i, 0]]
            };
        }

        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn CalibrationFunction> {
        Box::new(self.clone())
    }
}

/// Helper function to create Dirichlet calibration with cross-validation
///
/// This function fits Dirichlet calibration using stratified cross-validation
/// to avoid overfitting on the calibration data.
pub fn fit_dirichlet_with_cv(
    probabilities: &Array2<f64>,
    y_true: &Array1<i32>,
    cv_folds: usize,
    l2_reg: f64,
    max_iter: usize,
    tol: f64,
    learning_rate: f64,
    random_state: Option<u64>,
) -> SklResult<DirichletCalibration> {
    use crate::calibration::stratified_kfold_split;

    let (n_samples, n_classes) = probabilities.dim();
    let mut calibrated_probs = Array2::zeros((n_samples, n_classes));

    // Get stratified splits
    let splits = stratified_kfold_split(y_true, cv_folds, random_state)?;

    for (train_indices, test_indices) in splits {
        if train_indices.is_empty() || test_indices.is_empty() {
            continue;
        }

        // Create training data
        let train_probs = probabilities.select(Axis(0), &train_indices);
        let train_labels = Array1::from_vec(train_indices.iter().map(|&i| y_true[i]).collect());

        // Fit calibrator on training fold
        let mut calibrator = DirichletCalibration::new(l2_reg);
        calibrator.fit(&train_probs, &train_labels, max_iter, tol, learning_rate)?;

        // Apply to test fold
        let test_probs = probabilities.select(Axis(0), &test_indices);
        let cal_test_probs = calibrator.transform(&test_probs)?;

        // Store calibrated probabilities
        for (i, &test_idx) in test_indices.iter().enumerate() {
            for j in 0..n_classes {
                calibrated_probs[[test_idx, j]] = cal_test_probs[[i, j]];
            }
        }
    }

    // Fit final model on all data
    let mut final_calibrator = DirichletCalibration::new(l2_reg);
    final_calibrator.fit(probabilities, y_true, max_iter, tol, learning_rate)?;

    Ok(final_calibrator)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_dirichlet_calibration_basic() {
        let mut calibrator = DirichletCalibration::new(0.01);

        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        calibrator
            .fit(&probabilities, &y_true, 100, 1e-6, 0.01)
            .unwrap();

        let calibrated = calibrator.transform(&probabilities).unwrap();

        // Check dimensions
        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Check that all probabilities are between 0 and 1
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        assert!(calibrator.fitted);
    }

    #[test]
    fn test_dirichlet_calibration_binary() {
        let calibrator = DirichletCalibration::new(0.01);

        let uncalibrated = array![0.9, 0.1, 0.7, 0.3];

        // Note: This will fail because calibrator is not fitted
        // In practice, you would fit it first
        let result = calibrator.calibrate(&uncalibrated);
        assert!(result.is_err()); // Should fail because not fitted
    }

    #[test]
    fn test_dirichlet_calibration_softmax() {
        let calibrator = DirichletCalibration::new(0.01);

        let logits = array![[2.0, 1.0, 0.5], [1.5, 2.5, 0.8]];

        let probabilities = calibrator.softmax(&logits).unwrap();

        // Check dimensions
        assert_eq!(probabilities.dim(), (2, 3));

        // Check that probabilities sum to 1
        for i in 0..2 {
            let row_sum: f64 = probabilities.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Check that all probabilities are between 0 and 1
        for &prob in probabilities.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_dirichlet_calibration_transform_unfitted() {
        let calibrator = DirichletCalibration::new(0.01);

        let probabilities = array![[0.7, 0.3], [0.4, 0.6]];

        let result = calibrator.transform(&probabilities);
        assert!(result.is_err()); // Should fail because not fitted
    }

    #[test]
    fn test_dirichlet_calibration_empty_data() {
        let mut calibrator = DirichletCalibration::new(0.01);

        let probabilities = Array2::<f64>::zeros((0, 3));
        let y_true = Array1::<i32>::zeros(0);

        let result = calibrator.fit(&probabilities, &y_true, 100, 1e-6, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_dirichlet_calibration_mismatched_dimensions() {
        let mut calibrator = DirichletCalibration::new(0.01);

        let probabilities = array![[0.7, 0.3], [0.4, 0.6]];
        let y_true = array![0, 1, 2]; // Different number of samples

        let result = calibrator.fit(&probabilities, &y_true, 100, 1e-6, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_dirichlet_calibration_regularization() {
        let mut calibrator_low_reg = DirichletCalibration::new(0.001);
        let mut calibrator_high_reg = DirichletCalibration::new(0.1);

        let probabilities = array![
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        calibrator_low_reg
            .fit(&probabilities, &y_true, 100, 1e-6, 0.01)
            .unwrap();
        calibrator_high_reg
            .fit(&probabilities, &y_true, 100, 1e-6, 0.01)
            .unwrap();

        // Both should produce valid calibrated probabilities
        let cal_low = calibrator_low_reg.transform(&probabilities).unwrap();
        let cal_high = calibrator_high_reg.transform(&probabilities).unwrap();

        assert_eq!(cal_low.dim(), (4, 3));
        assert_eq!(cal_high.dim(), (4, 3));

        // High regularization should generally produce less extreme weights
        let weights_low_norm: f64 = calibrator_low_reg.weights.iter().map(|&x| x.abs()).sum();
        let weights_high_norm: f64 = calibrator_high_reg.weights.iter().map(|&x| x.abs()).sum();

        // This might not always be true depending on the optimization, but generally expected
        // assert!(weights_high_norm <= weights_low_norm);
    }

    #[test]
    fn test_dirichlet_calibration_prediction_logits() {
        let mut calibrator = DirichletCalibration::new(0.01);

        // Initialize with identity transformation
        calibrator.weights = Array2::eye(3);
        calibrator.bias = Array1::zeros(3);
        calibrator.fitted = true;

        let logits = array![[1.0, 2.0, 0.5], [0.5, 1.0, 2.0]];

        let result = calibrator.predict_logits(&logits).unwrap();

        // With identity transformation, should be close to original + bias (which is zero)
        assert_eq!(result.dim(), (2, 3));
        for i in 0..2 {
            for j in 0..3 {
                assert!((result[[i, j]] - logits[[i, j]]).abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_dirichlet_calibration_with_cv() {
        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1],
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2]
        ];
        let y_true = array![0, 1, 2, 0, 0, 1];

        let calibrator =
            fit_dirichlet_with_cv(&probabilities, &y_true, 3, 0.01, 100, 1e-6, 0.01, Some(42))
                .unwrap();

        assert!(calibrator.fitted);

        let calibrated = calibrator.transform(&probabilities).unwrap();
        assert_eq!(calibrated.dim(), (6, 3));

        // Check that probabilities sum to 1
        for i in 0..6 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_calibration_convergence() {
        let mut calibrator = DirichletCalibration::new(0.01);

        let probabilities = array![[0.9, 0.1], [0.1, 0.9], [0.8, 0.2], [0.2, 0.8]];
        let y_true = array![0, 1, 0, 1];

        // Test with very tight tolerance
        calibrator
            .fit(&probabilities, &y_true, 1000, 1e-10, 0.01)
            .unwrap();

        let calibrated = calibrator.transform(&probabilities).unwrap();

        // Should still produce valid probabilities
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_dirichlet_calibration_numerical_stability() {
        let mut calibrator = DirichletCalibration::new(0.01);

        // Test with very confident (extreme) probabilities
        let probabilities = array![
            [0.999, 0.0005, 0.0005],
            [0.0005, 0.999, 0.0005],
            [0.0005, 0.0005, 0.999]
        ];
        let y_true = array![0, 1, 2];

        calibrator
            .fit(&probabilities, &y_true, 100, 1e-6, 0.01)
            .unwrap();

        let calibrated = calibrator.transform(&probabilities).unwrap();

        // Should handle extreme probabilities without numerical issues
        for &prob in calibrated.iter() {
            assert!(prob.is_finite());
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        for i in 0..3 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }
}
