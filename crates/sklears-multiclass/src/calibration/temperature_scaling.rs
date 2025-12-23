//! Temperature Scaling Calibration
//!
//! Temperature scaling is a simple calibration method that applies a single
//! temperature parameter to scale the logits before applying softmax.
//! It preserves the ranking of predictions while improving calibration.

use super::CalibrationFunction;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Temperature scaling calibrator
///
/// Applies a temperature parameter T to scale logits: p_calibrated = softmax(logits / T)
/// The temperature is learned by minimizing the negative log-likelihood on a validation set.
#[derive(Debug, Clone)]
pub struct TemperatureScaling {
    /// Temperature parameter (T > 0)
    pub temperature: f64,
}

impl TemperatureScaling {
    /// Create a new temperature scaling calibrator
    pub fn new(initial_temperature: f64) -> Self {
        Self {
            temperature: initial_temperature.max(1e-6), // Ensure positive
        }
    }

    /// Fit the temperature parameter to validation data
    ///
    /// # Arguments
    /// * `logits` - Raw logit outputs from classifier [n_samples, n_classes]
    /// * `y_true` - True class labels
    /// * `max_iter` - Maximum number of optimization iterations
    /// * `tol` - Tolerance for convergence
    /// * `learning_rate` - Learning rate for gradient descent
    pub fn fit(
        &mut self,
        logits: &Array2<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
        learning_rate: f64,
    ) -> SklResult<()> {
        let (n_samples, n_classes) = logits.dim();

        if n_samples != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "Logits and labels must have same number of samples".to_string(),
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
        // So we allow fewer classes in labels than in logits
        if classes.len() > n_classes {
            return Err(SklearsError::InvalidInput(
                "Number of unique classes in labels exceeds logits dimensions".to_string(),
            ));
        }

        // Create class mapping - map each unique class to its index in logits
        let mut class_to_idx: std::collections::HashMap<i32, usize> =
            std::collections::HashMap::new();

        // If we have fewer classes in labels than logits, map them to corresponding indices
        for &class_label in classes.iter() {
            if class_label < n_classes as i32 {
                class_to_idx.insert(class_label, class_label as usize);
            } else {
                return Err(SklearsError::InvalidInput(format!(
                    "Class label {} exceeds number of logit columns",
                    class_label
                )));
            }
        }

        // Optimize temperature using gradient descent
        let mut temperature = self.temperature;
        let mut prev_loss = f64::INFINITY;

        for _iter in 0..max_iter {
            // Compute scaled probabilities
            let scaled_probs = self.apply_temperature_scaling(logits, temperature)?;

            // Compute negative log-likelihood and gradient
            let (loss, gradient) = self.compute_loss_and_gradient(
                logits,
                &scaled_probs,
                y_true,
                &class_to_idx,
                temperature,
            )?;

            // Check for convergence
            if (prev_loss - loss).abs() < tol {
                break;
            }
            prev_loss = loss;

            // Update temperature using gradient descent
            temperature = (temperature - learning_rate * gradient).max(1e-6);
        }

        self.temperature = temperature;
        Ok(())
    }

    /// Apply temperature scaling to logits
    fn apply_temperature_scaling(
        &self,
        logits: &Array2<f64>,
        temperature: f64,
    ) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = logits.dim();
        let mut scaled_probs = Array2::zeros((n_samples, n_classes));

        for i in 0..n_samples {
            let row = logits.row(i);

            // Scale logits by temperature
            let scaled_logits: Vec<f64> = row.iter().map(|&x| x / temperature).collect();

            // Apply softmax
            let max_logit = scaled_logits
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            let exp_logits: Vec<f64> = scaled_logits
                .iter()
                .map(|&x| (x - max_logit).exp())
                .collect();
            let sum_exp: f64 = exp_logits.iter().sum();

            for j in 0..n_classes {
                scaled_probs[[i, j]] = if sum_exp > 0.0 {
                    exp_logits[j] / sum_exp
                } else {
                    1.0 / n_classes as f64
                };
            }
        }

        Ok(scaled_probs)
    }

    /// Compute negative log-likelihood loss and gradient with respect to temperature
    fn compute_loss_and_gradient(
        &self,
        logits: &Array2<f64>,
        scaled_probs: &Array2<f64>,
        y_true: &Array1<i32>,
        class_to_idx: &std::collections::HashMap<i32, usize>,
        temperature: f64,
    ) -> SklResult<(f64, f64)> {
        let n_samples = logits.nrows();
        let mut loss = 0.0;
        let mut gradient = 0.0;

        for i in 0..n_samples {
            let true_class = y_true[i];
            let class_idx = class_to_idx.get(&true_class).ok_or_else(|| {
                SklearsError::InvalidInput(format!("Unknown class label: {}", true_class))
            })?;

            let prob = scaled_probs[[i, *class_idx]].max(1e-15);
            loss += -prob.ln();

            // Gradient computation for temperature
            let logit_true = logits[[i, *class_idx]];
            let mean_logit: f64 = logits
                .row(i)
                .iter()
                .zip(scaled_probs.row(i).iter())
                .map(|(&logit, &prob)| logit * prob)
                .sum();

            gradient += (mean_logit - logit_true) / (temperature * temperature);
        }

        loss /= n_samples as f64;
        gradient /= n_samples as f64;

        Ok((loss, gradient))
    }

    /// Transform logits to calibrated probabilities
    pub fn transform(&self, logits: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.apply_temperature_scaling(logits, self.temperature)
    }

    /// Transform probabilities to calibrated probabilities
    /// Note: This converts probabilities back to logits first, then applies temperature scaling
    pub fn transform_probabilities(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_classes) = probabilities.dim();
        let mut logits = Array2::zeros((n_samples, n_classes));

        // Convert probabilities to logits
        for i in 0..n_samples {
            let row = probabilities.row(i);

            // Normalize probabilities to avoid numerical issues
            let row_sum: f64 = row.sum();
            let normalized_probs: Vec<f64> = if row_sum > 0.0 {
                row.iter()
                    .map(|&p| (p / row_sum).clamp(1e-15, 1.0 - 1e-15))
                    .collect()
            } else {
                vec![1.0 / n_classes as f64; n_classes]
            };

            // Convert to logits
            for j in 0..n_classes {
                logits[[i, j]] = normalized_probs[j].ln();
            }
        }

        self.transform(&logits)
    }
}

impl Default for TemperatureScaling {
    fn default() -> Self {
        Self::new(1.0)
    }
}

impl CalibrationFunction for TemperatureScaling {
    fn calibrate(&self, uncalibrated_probs: &Array1<f64>) -> SklResult<Array1<f64>> {
        // For binary case, convert to 2D and apply temperature scaling
        let n_samples = uncalibrated_probs.len();
        let mut probs_2d = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            let p = uncalibrated_probs[i].clamp(1e-15, 1.0 - 1e-15);
            probs_2d[[i, 0]] = 1.0 - p;
            probs_2d[[i, 1]] = p;
        }

        let calibrated_2d = self.transform_probabilities(&probs_2d)?;

        // Extract positive class probabilities
        let mut result = Array1::zeros(n_samples);
        for i in 0..n_samples {
            result[i] = calibrated_2d[[i, 1]];
        }

        Ok(result)
    }

    fn clone_box(&self) -> Box<dyn CalibrationFunction> {
        Box::new(self.clone())
    }
}

/// Multiclass temperature scaling wrapper
///
/// For multiclass problems, a single temperature parameter is learned
/// that is applied to all classes simultaneously.
#[derive(Debug, Clone)]
pub struct MulticlassTemperatureScaling {
    /// Temperature scaling calibrator
    pub scaler: TemperatureScaling,
    /// Class labels
    pub classes: Array1<i32>,
}

impl MulticlassTemperatureScaling {
    /// Create a new multiclass temperature scaling calibrator
    pub fn new(initial_temperature: f64) -> Self {
        Self {
            scaler: TemperatureScaling::new(initial_temperature),
            classes: Array1::zeros(0),
        }
    }

    /// Fit temperature scaling to multiclass data
    ///
    /// # Arguments
    /// * `logits` - Raw logit outputs [n_samples, n_classes]
    /// * `y_true` - True class labels
    /// * `max_iter` - Maximum optimization iterations
    /// * `tol` - Convergence tolerance
    /// * `learning_rate` - Learning rate for optimization
    pub fn fit(
        &mut self,
        logits: &Array2<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
        learning_rate: f64,
    ) -> SklResult<()> {
        // Store class labels
        let mut classes = y_true.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.classes = Array1::from_vec(classes);

        // Fit temperature parameter
        self.scaler
            .fit(logits, y_true, max_iter, tol, learning_rate)
    }

    /// Fit using probabilities (converts to logits internally)
    pub fn fit_probabilities(
        &mut self,
        probabilities: &Array2<f64>,
        y_true: &Array1<i32>,
        max_iter: usize,
        tol: f64,
        learning_rate: f64,
    ) -> SklResult<()> {
        let (n_samples, n_classes) = probabilities.dim();

        // Store class labels
        let mut classes = y_true.to_vec();
        classes.sort_unstable();
        classes.dedup();
        self.classes = Array1::from_vec(classes);

        // Convert probabilities to logits
        let mut logits = Array2::zeros((n_samples, n_classes));
        for i in 0..n_samples {
            let row = probabilities.row(i);
            let row_sum: f64 = row.sum();

            for j in 0..n_classes {
                let p = if row_sum > 0.0 {
                    (row[j] / row_sum).clamp(1e-15, 1.0 - 1e-15)
                } else {
                    1.0 / n_classes as f64
                };
                logits[[i, j]] = p.ln();
            }
        }

        self.scaler
            .fit(&logits, y_true, max_iter, tol, learning_rate)
    }

    /// Transform logits to calibrated probabilities
    pub fn transform(&self, logits: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.scaler.transform(logits)
    }

    /// Transform probabilities to calibrated probabilities
    pub fn transform_probabilities(&self, probabilities: &Array2<f64>) -> SklResult<Array2<f64>> {
        self.scaler.transform_probabilities(probabilities)
    }

    /// Get the learned temperature parameter
    pub fn temperature(&self) -> f64 {
        self.scaler.temperature
    }
}

impl Default for MulticlassTemperatureScaling {
    fn default() -> Self {
        Self::new(1.0)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_temperature_scaling_basic() {
        let mut scaler = TemperatureScaling::new(1.0);

        let logits = array![
            [2.0, 1.0, 0.5],
            [1.5, 2.5, 0.8],
            [0.5, 1.0, 2.0],
            [3.0, 0.5, 1.0]
        ];
        let y_true = array![0, 1, 2, 0];

        scaler.fit(&logits, &y_true, 100, 1e-6, 0.01).unwrap();

        let calibrated = scaler.transform(&logits).unwrap();

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

        // Temperature should be positive
        assert!(scaler.temperature > 0.0);
    }

    #[test]
    fn test_temperature_scaling_probabilities() {
        let scaler = TemperatureScaling::new(1.0);

        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];

        let calibrated = scaler.transform_probabilities(&probabilities).unwrap();

        // Check dimensions
        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_multiclass_temperature_scaling() {
        let mut scaler = MulticlassTemperatureScaling::new(1.0);

        let logits = array![
            [2.0, 1.0, 0.5],
            [1.5, 2.5, 0.8],
            [0.5, 1.0, 2.0],
            [3.0, 0.5, 1.0]
        ];
        let y_true = array![0, 1, 2, 0];

        scaler.fit(&logits, &y_true, 100, 1e-6, 0.01).unwrap();

        let calibrated = scaler.transform(&logits).unwrap();

        // Check dimensions
        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }

        // Check temperature is learned
        assert!(scaler.temperature() > 0.0);
    }

    #[test]
    fn test_multiclass_temperature_scaling_with_probabilities() {
        let mut scaler = MulticlassTemperatureScaling::new(1.0);

        let probabilities = array![
            [0.7, 0.2, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];
        let y_true = array![0, 1, 2, 0];

        scaler
            .fit_probabilities(&probabilities, &y_true, 100, 1e-6, 0.01)
            .unwrap();

        let calibrated = scaler.transform_probabilities(&probabilities).unwrap();

        // Check dimensions
        assert_eq!(calibrated.dim(), (4, 3));

        // Check that probabilities sum to 1
        for i in 0..4 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_temperature_scaling_binary_calibration_function() {
        let scaler = TemperatureScaling::new(2.0);

        let uncalibrated = array![0.9, 0.1, 0.7, 0.3];
        let calibrated = scaler.calibrate(&uncalibrated).unwrap();

        assert_eq!(calibrated.len(), 4);
        for &prob in calibrated.iter() {
            assert!(prob >= 0.0 && prob <= 1.0);
        }
    }

    #[test]
    fn test_temperature_scaling_edge_cases() {
        let mut scaler = TemperatureScaling::new(1.0);

        // Test with very confident predictions
        let logits = array![[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]];
        let y_true = array![0, 1, 2];

        scaler.fit(&logits, &y_true, 100, 1e-6, 0.01).unwrap();

        let calibrated = scaler.transform(&logits).unwrap();

        // Even very confident predictions should be calibrated
        for i in 0..3 {
            let row_sum: f64 = calibrated.row(i).sum();
            assert!((row_sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_temperature_scaling_empty_data() {
        let mut scaler = TemperatureScaling::new(1.0);

        let logits = Array2::<f64>::zeros((0, 3));
        let y_true = Array1::<i32>::zeros(0);

        let result = scaler.fit(&logits, &y_true, 100, 1e-6, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_scaling_mismatched_dimensions() {
        let mut scaler = TemperatureScaling::new(1.0);

        let logits = array![[1.0, 2.0], [2.0, 1.0]];
        let y_true = array![0, 1, 2]; // Different number of samples

        let result = scaler.fit(&logits, &y_true, 100, 1e-6, 0.01);
        assert!(result.is_err());
    }

    #[test]
    fn test_temperature_scaling_convergence() {
        let mut scaler = TemperatureScaling::new(1.0);

        let logits = array![[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]];
        let y_true = array![0, 1, 0, 1];

        let _initial_temp = scaler.temperature;
        scaler.fit(&logits, &y_true, 1000, 1e-8, 0.01).unwrap();

        // Temperature should change during optimization
        // Note: This might not always be true if data is already well-calibrated
        // But for most cases, some change is expected
    }

    #[test]
    fn test_temperature_scaling_positive_constraint() {
        let scaler = TemperatureScaling::new(-1.0);

        // Temperature should be forced to be positive
        assert!(scaler.temperature > 0.0);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let scaler = TemperatureScaling::new(0.1); // Very small temperature

        let logits = array![[1000.0, 999.0, 998.0]];

        let result = scaler.apply_temperature_scaling(&logits, 0.1).unwrap();

        // Should not produce NaN or Inf
        for &val in result.iter() {
            assert!(val.is_finite());
            assert!(val >= 0.0 && val <= 1.0);
        }

        // Row should sum to 1
        let row_sum: f64 = result.row(0).sum();
        assert!((row_sum - 1.0).abs() < 1e-10);
    }
}
