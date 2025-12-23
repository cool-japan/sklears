//! Platt Scaling Calibration
//!
//! Platt scaling is a method for calibrating the output of a classification model
//! using logistic regression. It fits a sigmoid function to the classifier's outputs
//! to transform them into calibrated probabilities.

use super::CalibrationFunction;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Platt scaling calibrator using sigmoid transformation
///
/// Fits parameters A and B such that the calibrated probability is:
/// P(y=1|f) = 1 / (1 + exp(A*f + B))
///
/// For multiclass problems, separate Platt scalers are trained for each class
/// using a One-vs-Rest approach.
#[derive(Debug, Clone)]
pub struct PlattScaling {
    /// Sigmoid parameter A (slope)
    pub a: f64,
    /// Sigmoid parameter B (intercept)
    pub b: f64,
}

impl PlattScaling {
    /// Create a new Platt scaling calibrator
    pub fn new() -> Self {
        Self { a: 0.0, b: 0.0 }
    }

    /// Fit the Platt scaling parameters to binary classification data
    ///
    /// # Arguments
    /// * `decision_values` - Raw decision function outputs from classifier
    /// * `y_true` - True binary labels (0 or 1)
    /// * `max_iter` - Maximum number of optimization iterations
    /// * `tol` - Tolerance for convergence
    pub fn fit(
        &mut self,
        decision_values: &Array1<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
    ) -> SklResult<()> {
        if decision_values.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Decision values and true labels must have same length".to_string(),
            ));
        }

        let n_samples = decision_values.len();
        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty data".to_string(),
            ));
        }

        // Convert labels to target probabilities with smoothing
        let mut target_probs = Array1::zeros(n_samples);
        let mut n_pos = 0;
        let mut n_neg = 0;

        for &label in y_true.iter() {
            if label == 1 {
                n_pos += 1;
            } else {
                n_neg += 1;
            }
        }

        if n_pos == 0 || n_neg == 0 {
            // Only one class present, use default calibration
            self.a = 0.0;
            self.b = if n_pos > 0 { 3.0 } else { -3.0 }; // For > 95% prob
            return Ok(());
        }

        // Apply prior smoothing to avoid overfitting
        let pos_prior = (n_pos as f64 + 1.0) / (n_samples as f64 + 2.0);
        let neg_prior = (n_neg as f64 + 1.0) / (n_samples as f64 + 2.0);

        for (i, &label) in y_true.iter().enumerate() {
            target_probs[i] = if label == 1 { pos_prior } else { neg_prior };
        }

        // Initialize parameters with better starting point
        let mut a = 1.0; // Start with slope of 1
        let mut b = (neg_prior / pos_prior).ln();

        // Newton-Raphson optimization with line search
        let mut prev_log_likelihood = f64::NEG_INFINITY;

        for iter in 0..max_iter {
            let mut hessian = Array2::<f64>::zeros((2, 2));
            let mut gradient = Array1::<f64>::zeros(2);
            let mut log_likelihood = 0.0;

            for i in 0..n_samples {
                let fval = decision_values[i];
                let t = target_probs[i];
                let d1 = a * fval + b;

                // Compute sigmoid and its derivatives
                let p = self.sigmoid(d1);
                let p_clipped = p.clamp(1e-15, 1.0 - 1e-15);

                // Log-likelihood
                log_likelihood += t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln();

                // Gradient
                let diff = p - t;
                gradient[0] += fval * diff;
                gradient[1] += diff;

                // Hessian (with regularization to ensure positive definiteness)
                let pp = p * (1.0 - p);
                hessian[[0, 0]] += fval * fval * pp + 1e-12;
                hessian[[0, 1]] += fval * pp;
                hessian[[1, 0]] += fval * pp;
                hessian[[1, 1]] += pp + 1e-12;
            }

            // Check for convergence
            let grad_norm: f64 = (gradient[0] * gradient[0] + gradient[1] * gradient[1]).sqrt();
            if grad_norm < tol {
                break;
            }

            // Check if log likelihood is improving (with tolerance for numerical issues)
            if iter > 0 && log_likelihood < prev_log_likelihood - 1e-8 {
                break;
            }
            prev_log_likelihood = log_likelihood;

            // Solve Hessian * step = -gradient
            let det = hessian[[0, 0]] * hessian[[1, 1]] - hessian[[0, 1]] * hessian[[1, 0]];
            if det.abs() < 1e-12 {
                break; // Singular matrix
            }

            let step_a = (-gradient[0] * hessian[[1, 1]] + gradient[1] * hessian[[0, 1]]) / det;
            let step_b = (gradient[0] * hessian[[1, 0]] - gradient[1] * hessian[[0, 0]]) / det;

            // Line search to find appropriate step size
            let mut step_size = 1.0;
            for _ in 0..10 {
                let new_a = a + step_size * step_a;
                let new_b = b + step_size * step_b;

                // Check if new parameters give better likelihood
                let mut new_likelihood = 0.0;
                let mut valid = true;

                for i in 0..n_samples {
                    let fval = decision_values[i];
                    let t = target_probs[i];
                    let d1 = new_a * fval + new_b;
                    let p = self.sigmoid(d1);
                    let p_clipped = p.clamp(1e-15, 1.0 - 1e-15);

                    if p_clipped.ln().is_finite() && (1.0 - p_clipped).ln().is_finite() {
                        new_likelihood += t * p_clipped.ln() + (1.0 - t) * (1.0 - p_clipped).ln();
                    } else {
                        valid = false;
                        break;
                    }
                }

                if valid && new_likelihood >= log_likelihood - 1e-4 {
                    break;
                }
                step_size *= 0.5;

                if step_size < 1e-8 {
                    step_size = 0.0;
                    break;
                }
            }

            // Update parameters with found step size
            a += step_size * step_a;
            b += step_size * step_b;

            // Prevent extreme parameter values
            a = a.clamp(-100.0, 100.0);
            b = b.clamp(-100.0, 100.0);
        }

        self.a = a;
        self.b = b;
        Ok(())
    }

    /// Apply sigmoid transformation to decision values
    pub fn transform(&self, decision_values: &Array1<f64>) -> SklResult<Array1<f64>> {
        let mut calibrated = Array1::zeros(decision_values.len());

        for (i, &val) in decision_values.iter().enumerate() {
            calibrated[i] = self.sigmoid(self.a * val + self.b);
        }

        Ok(calibrated)
    }

    /// Sigmoid function with numerical stability
    fn sigmoid(&self, x: f64) -> f64 {
        if x > 700.0 {
            1.0
        } else if x < -700.0 {
            0.0
        } else {
            1.0 / (1.0 + (-x).exp())
        }
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

impl CalibrationFunction for PlattScaling {
    fn calibrate(&self, uncalibrated_probs: &Array1<f64>) -> SklResult<Array1<f64>> {
        // Convert probabilities to decision values (logits)
        let mut decision_values = Array1::zeros(uncalibrated_probs.len());

        for (i, &prob) in uncalibrated_probs.iter().enumerate() {
            let p_clipped = prob.clamp(1e-15, 1.0 - 1e-15);
            decision_values[i] = (p_clipped / (1.0 - p_clipped)).ln();
        }

        self.transform(&decision_values)
    }

    fn clone_box(&self) -> Box<dyn CalibrationFunction> {
        Box::new(self.clone())
    }
}

/// Multiclass Platt scaling wrapper
///
/// Handles calibration for multiclass problems by training separate
/// Platt scalers for each class using One-vs-Rest decomposition.
#[derive(Debug, Clone)]
pub struct MulticlassPlattScaling {
    /// Individual Platt scalers for each class
    pub scalers: Vec<PlattScaling>,
    /// Class labels
    pub classes: Array1<i32>,
}

impl MulticlassPlattScaling {
    /// Create a new multiclass Platt scaling calibrator
    pub fn new() -> Self {
        Self {
            scalers: Vec::new(),
            classes: Array1::zeros(0),
        }
    }

    /// Fit multiclass Platt scaling
    ///
    /// # Arguments
    /// * `probabilities` - Uncalibrated probabilities from classifier [n_samples, n_classes]
    /// * `y_true` - True class labels
    /// * `max_iter` - Maximum iterations for optimization
    /// * `tol` - Convergence tolerance
    pub fn fit(
        &mut self,
        probabilities: &Array2<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
    ) -> SklResult<()> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have same number of samples".to_string(),
            ));
        }

        // Get unique classes
        let mut classes = y_true.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.classes = Array1::from_vec(classes.clone());

        if classes.len() != n_classes {
            return Err(SklearsError::InvalidInput(
                "Number of classes in probabilities doesn't match labels".to_string(),
            ));
        }

        // Train one Platt scaler per class (One-vs-Rest)
        self.scalers = Vec::with_capacity(n_classes);

        for (class_idx, &target_class) in classes.iter().enumerate() {
            let mut scaler = PlattScaling::new();

            // Create binary labels for current class
            let binary_labels = Array1::from_vec(
                y_true
                    .iter()
                    .map(|&label| if label == target_class { 1 } else { 0 })
                    .collect(),
            );

            // Extract probabilities for current class
            let class_probs = probabilities.column(class_idx).to_owned();

            // Convert probabilities to decision values
            let mut decision_values = Array1::zeros(n_samples);
            for (i, &prob) in class_probs.iter().enumerate() {
                let p_clipped = prob.clamp(1e-15, 1.0 - 1e-15);
                decision_values[i] = (p_clipped / (1.0 - p_clipped)).ln();
            }

            scaler.fit(&decision_values, &binary_labels, max_iter, tol)?;
            self.scalers.push(scaler);
        }

        Ok(())
    }

    /// Transform uncalibrated probabilities to calibrated probabilities
    pub fn transform(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();

        if n_classes != self.scalers.len() {
            return Err(SklearsError::InvalidInput(
                "Number of classes doesn't match fitted model".to_string(),
            ));
        }

        let mut calibrated = Array2::zeros((n_samples, n_classes));

        // Apply each scaler to its corresponding class probabilities
        for class_idx in 0..n_classes {
            let class_probs = probabilities.column(class_idx).to_owned();
            let calibrated_probs = self.scalers[class_idx].calibrate(&class_probs)?;

            for i in 0..n_samples {
                calibrated[[i, class_idx]] = calibrated_probs[i];
            }
        }

        // Normalize probabilities to sum to 1
        for i in 0..n_samples {
            let row_sum: f64 = calibrated.row(i).sum();
            if row_sum > 0.0 {
                for j in 0..n_classes {
                    calibrated[[i, j]] /= row_sum;
                }
            } else {
                // Uniform distribution fallback
                for j in 0..n_classes {
                    calibrated[[i, j]] = 1.0 / n_classes as f64;
                }
            }
        }

        Ok(calibrated)
    }
}

impl Default for MulticlassPlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_platt_scaling_basic() {
        let mut scaler = PlattScaling::new();

        let decision_values = array![1.0, 2.0, -1.0, -2.0];
        let y_true = array![1, 1, 0, 0];

        scaler.fit(&decision_values, &y_true, 100, 1e-6).unwrap();

        let calibrated = scaler.transform(&decision_values).unwrap();

        // Check that calibrated probabilities are between 0 and 1
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }

        // Positive decision values should give higher probabilities
        assert!(calibrated[0] > calibrated[2]);
        assert!(calibrated[1] > calibrated[3]);
    }

    #[test]
    fn test_platt_scaling_sigmoid() {
        let scaler = PlattScaling { a: 1.0, b: 0.0 };

        // Test sigmoid function properties
        assert!((scaler.sigmoid(0.0) - 0.5).abs() < 1e-10);
        assert!(scaler.sigmoid(10.0) > 0.99);
        assert!(scaler.sigmoid(-10.0) < 0.01);

        // Test extreme values for numerical stability
        assert_eq!(scaler.sigmoid(1000.0), 1.0);
        assert_eq!(scaler.sigmoid(-1000.0), 0.0);
    }

    #[test]
    fn test_platt_scaling_single_class() {
        let mut scaler = PlattScaling::new();

        let decision_values = array![1.0, 2.0, 3.0];
        let y_true = array![1, 1, 1]; // Only positive class

        scaler.fit(&decision_values, &y_true, 100, 1e-6).unwrap();

        let calibrated = scaler.transform(&decision_values).unwrap();

        // All probabilities should be high for positive-only data
        for &prob in calibrated.iter() {
            assert!(prob > 0.9);
        }
    }

    #[test]
    fn test_multiclass_platt_scaling() {
        let mut scaler = MulticlassPlattScaling::new();

        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        scaler.fit(&probabilities, &y_true, 100, 1e-6).unwrap();

        let calibrated = scaler.transform(&probabilities).unwrap();

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
    }

    #[test]
    fn test_platt_scaling_calibration_function() {
        let scaler = PlattScaling { a: 1.0, b: 0.0 };

        let uncalibrated = array![0.8, 0.2, 0.6, 0.4];
        let calibrated = scaler.calibrate(&uncalibrated).unwrap();

        assert_eq!(calibrated.len(), 4);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_platt_scaling_empty_data() {
        let mut scaler = PlattScaling::new();

        let decision_values = Array1::<f64>::zeros(0);
        let y_true = Array1::<i32>::zeros(0);

        let result = scaler.fit(&decision_values, &y_true, 100, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_platt_scaling_mismatched_lengths() {
        let mut scaler = PlattScaling::new();

        let decision_values = array![1.0, 2.0];
        let y_true = array![1, 1, 0]; // Different length

        let result = scaler.fit(&decision_values, &y_true, 100, 1e-6);
        assert!(result.is_err());
    }

    #[test]
    fn test_multiclass_platt_scaling_dimension_mismatch() {
        let mut scaler = MulticlassPlattScaling::new();

        let probabilities = array![[0.7, 0.3], [0.4, 0.6]];
        let y_true = array![0, 1, 2]; // Different number of samples

        let result = scaler.fit(&probabilities, &y_true, 100, 1e-6);
        assert!(result.is_err());
    }
}
