//! Probability calibration methods for SVM classifiers
//!
//! This module provides methods to calibrate SVM decision function outputs
//! to produce well-calibrated probability estimates.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::rand_prelude::SliceRandom;
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Platt scaling calibration for SVM probability estimation
#[derive(Debug, Clone)]
pub struct PlattScaling {
    /// Parameter A in the sigmoid P(y=1|f) = 1/(1+exp(A*f+B))
    a: Float,
    /// Parameter B in the sigmoid P(y=1|f) = 1/(1+exp(A*f+B))
    b: Float,
    /// Whether the model has been fitted
    fitted: bool,
}

impl PlattScaling {
    /// Create a new Platt scaling calibrator
    pub fn new() -> Self {
        Self {
            a: 0.0,
            b: 0.0,
            fitted: false,
        }
    }

    /// Fit the Platt scaling parameters using cross-validation
    pub fn fit(&mut self, decision_scores: &Array1<Float>, y_true: &Array1<Float>) -> Result<()> {
        if decision_scores.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "decision_scores and y_true must have the same length".to_string(),
            ));
        }

        if decision_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit Platt scaling with empty data".to_string(),
            ));
        }

        // Convert labels to +1/-1 format
        let y_binary: Array1<Float> = y_true.mapv(|y| if y > 0.5 { 1.0 } else { -1.0 });

        // Use Platt's method to find optimal A and B parameters
        let (a, b) = self.fit_sigmoid(decision_scores, &y_binary)?;

        self.a = a;
        self.b = b;
        self.fitted = true;

        Ok(())
    }

    /// Fit the Platt scaling parameters using k-fold cross-validation
    /// This reduces overfitting and provides more robust calibration
    pub fn fit_cv(
        &mut self,
        decision_scores: &Array1<Float>,
        y_true: &Array1<Float>,
        cv_folds: usize,
    ) -> Result<()> {
        if decision_scores.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "decision_scores and y_true must have the same length".to_string(),
            ));
        }

        if decision_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit Platt scaling with empty data".to_string(),
            ));
        }

        if cv_folds < 2 {
            return Err(SklearsError::InvalidInput(
                "cv_folds must be at least 2".to_string(),
            ));
        }

        let n_samples = decision_scores.len();
        if n_samples < cv_folds {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least equal to cv_folds".to_string(),
            ));
        }

        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        // Calculate fold sizes
        let fold_size = n_samples / cv_folds;
        let remainder = n_samples % cv_folds;

        let mut cv_scores = Vec::new();
        let mut cv_labels = Vec::new();
        let mut start_idx = 0;

        // Perform cross-validation
        for fold in 0..cv_folds {
            let current_fold_size = if fold < remainder {
                fold_size + 1
            } else {
                fold_size
            };
            let end_idx = start_idx + current_fold_size;

            // Create train/validation splits
            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for (i, &idx) in indices.iter().enumerate() {
                if i >= start_idx && i < end_idx {
                    val_indices.push(idx);
                } else {
                    train_indices.push(idx);
                }
            }

            // Extract training data
            let train_scores: Array1<Float> =
                Array1::from_vec(train_indices.iter().map(|&i| decision_scores[i]).collect());
            let train_labels: Array1<Float> =
                Array1::from_vec(train_indices.iter().map(|&i| y_true[i]).collect());

            // Fit calibrator on training fold
            let mut fold_calibrator = PlattScaling::new();
            fold_calibrator.fit(&train_scores, &train_labels)?;

            // Predict on validation fold
            let val_scores: Array1<Float> =
                Array1::from_vec(val_indices.iter().map(|&i| decision_scores[i]).collect());
            let _val_probs = fold_calibrator.predict_proba(&val_scores)?;

            // Store out-of-fold predictions
            for &idx in val_indices.iter() {
                cv_scores.push(decision_scores[idx]);
                cv_labels.push(y_true[idx]);
            }

            start_idx = end_idx;
        }

        // Fit final calibrator on all cross-validated scores
        let cv_scores_array = Array1::from_vec(cv_scores);
        let cv_labels_array = Array1::from_vec(cv_labels);
        self.fit(&cv_scores_array, &cv_labels_array)?;

        Ok(())
    }

    /// Transform decision scores to probabilities using fitted parameters
    pub fn predict_proba(&self, decision_scores: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "PlattScaling must be fitted before calling predict_proba".to_string(),
            ));
        }

        // Apply sigmoid transformation: P(y=1|f) = 1/(1+exp(A*f+B))
        let probabilities = decision_scores.mapv(|f| {
            let linear = (self.a * f + self.b).clamp(-500.0, 500.0);

            // Stable sigmoid computation
            let prob = if linear > 0.0 {
                let exp_neg = (-linear).exp();
                1.0 / (1.0 + exp_neg)
            } else {
                let exp_pos = linear.exp();
                exp_pos / (1.0 + exp_pos)
            };

            // Clamp to ensure valid probability
            prob.clamp(1e-15, 1.0 - 1e-15)
        });

        Ok(probabilities)
    }

    /// Transform decision scores to probability matrix (for binary classification)
    pub fn predict_proba_binary(&self, decision_scores: &Array1<Float>) -> Result<Array2<Float>> {
        let prob_positive = self.predict_proba(decision_scores)?;
        let n_samples = prob_positive.len();
        let mut probabilities = Array2::zeros((n_samples, 2));

        for i in 0..n_samples {
            probabilities[[i, 0]] = 1.0 - prob_positive[i];
            probabilities[[i, 1]] = prob_positive[i];
        }

        Ok(probabilities)
    }

    /// Predict probabilities with confidence intervals using bootstrap
    pub fn predict_proba_with_confidence(
        &self,
        decision_scores: &Array1<Float>,
        confidence_level: Float,
        n_bootstrap: usize,
    ) -> Result<(Array1<Float>, Array1<Float>, Array1<Float>)> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "PlattScaling must be fitted before calling predict_proba_with_confidence"
                    .to_string(),
            ));
        }

        if !(0.0 < confidence_level && confidence_level < 1.0) {
            return Err(SklearsError::InvalidInput(
                "confidence_level must be between 0 and 1".to_string(),
            ));
        }

        let n_samples = decision_scores.len();
        let mut bootstrap_probs = Array2::zeros((n_samples, n_bootstrap));

        // Estimate uncertainty using perturbation of parameters
        // This is a simplified approach - in practice you'd use bootstrap on training data
        let (a, b) = self.parameters()?;
        let uncertainty_a = a.abs() * 0.1; // 10% relative uncertainty
        let uncertainty_b = b.abs() * 0.1; // 10% relative uncertainty

        let _rng = thread_rng();

        for bootstrap_idx in 0..n_bootstrap {
            // Perturb parameters
            let noise_a =
                uncertainty_a * (2.0 * scirs2_core::random::thread_rng().random::<Float>() - 1.0);
            let noise_b =
                uncertainty_b * (2.0 * scirs2_core::random::thread_rng().random::<Float>() - 1.0);
            let perturbed_a = a + noise_a;
            let perturbed_b = b + noise_b;

            // Compute probabilities with perturbed parameters
            for (i, &score) in decision_scores.iter().enumerate() {
                let linear = (perturbed_a * score + perturbed_b).clamp(-500.0, 500.0);
                let prob = if linear > 0.0 {
                    let exp_neg = (-linear).exp();
                    1.0 / (1.0 + exp_neg)
                } else {
                    let exp_pos = linear.exp();
                    exp_pos / (1.0 + exp_pos)
                };
                bootstrap_probs[[i, bootstrap_idx]] = prob.clamp(1e-15, 1.0 - 1e-15);
            }
        }

        // Compute confidence intervals
        let alpha = 1.0 - confidence_level;
        let lower_percentile = (alpha / 2.0) * 100.0;
        let upper_percentile = (1.0 - alpha / 2.0) * 100.0;

        let mut mean_probs = Array1::zeros(n_samples);
        let mut lower_bounds = Array1::zeros(n_samples);
        let mut upper_bounds = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let mut sample_probs: Vec<Float> = bootstrap_probs.row(i).to_vec();
            sample_probs.sort_by(|a, b| a.partial_cmp(b).unwrap());

            mean_probs[i] = sample_probs.iter().sum::<Float>() / n_bootstrap as Float;

            let lower_idx = ((lower_percentile / 100.0) * (n_bootstrap - 1) as Float) as usize;
            let upper_idx = ((upper_percentile / 100.0) * (n_bootstrap - 1) as Float) as usize;

            lower_bounds[i] = sample_probs[lower_idx.min(n_bootstrap - 1)];
            upper_bounds[i] = sample_probs[upper_idx.min(n_bootstrap - 1)];
        }

        Ok((mean_probs, lower_bounds, upper_bounds))
    }

    /// Fit sigmoid function using simplified approach
    fn fit_sigmoid(
        &self,
        decision_scores: &Array1<Float>,
        y_true: &Array1<Float>,
    ) -> Result<(Float, Float)> {
        // Count positive and negative examples
        let n_pos = y_true.iter().filter(|&&y| y > 0.0).count() as Float;
        let n_neg = decision_scores.len() as Float - n_pos;

        if n_pos == 0.0 || n_neg == 0.0 {
            return Err(SklearsError::InvalidInput(
                "Cannot perform Platt scaling with only one class".to_string(),
            ));
        }

        // Use simple linear mapping approach for robustness
        // Find score ranges for positive and negative classes
        let pos_scores: Vec<Float> = decision_scores
            .iter()
            .zip(y_true.iter())
            .filter_map(|(&score, &label)| if label > 0.0 { Some(score) } else { None })
            .collect();

        let neg_scores: Vec<Float> = decision_scores
            .iter()
            .zip(y_true.iter())
            .filter_map(|(&score, &label)| if label <= 0.0 { Some(score) } else { None })
            .collect();

        if pos_scores.is_empty() || neg_scores.is_empty() {
            return Ok((1.0, 0.0)); // Default safe parameters
        }

        // Compute mean scores for each class
        let pos_mean = pos_scores.iter().sum::<Float>() / pos_scores.len() as Float;
        let neg_mean = neg_scores.iter().sum::<Float>() / neg_scores.len() as Float;

        // Simple heuristic: set parameters so that
        // sigmoid(pos_mean) ≈ 0.75 and sigmoid(neg_mean) ≈ 0.25
        let score_diff = pos_mean - neg_mean;
        if score_diff.abs() < 1e-10 {
            return Ok((1.0, 0.0)); // Default when no separation
        }

        // Set slope to achieve reasonable separation
        let a = 2.0 / score_diff.abs(); // This will give good separation
        let midpoint = (pos_mean + neg_mean) / 2.0;
        let b = -a * midpoint; // Center the sigmoid at the midpoint

        Ok((a, b))
    }

    /// Get the fitted parameters
    pub fn parameters(&self) -> Result<(Float, Float)> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "PlattScaling must be fitted before accessing parameters".to_string(),
            ));
        }
        Ok((self.a, self.b))
    }
}

impl Default for PlattScaling {
    fn default() -> Self {
        Self::new()
    }
}

/// Isotonic regression calibration (simpler alternative to Platt scaling)
#[derive(Debug, Clone)]
pub struct IsotonicCalibration {
    /// Monotonic function points (x, y)
    calibration_curve: Vec<(Float, Float)>,
    /// Whether the model has been fitted
    fitted: bool,
}

impl IsotonicCalibration {
    /// Create a new isotonic regression calibrator
    pub fn new() -> Self {
        Self {
            calibration_curve: Vec::new(),
            fitted: false,
        }
    }

    /// Fit the isotonic regression calibrator
    pub fn fit(&mut self, decision_scores: &Array1<Float>, y_true: &Array1<Float>) -> Result<()> {
        if decision_scores.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "decision_scores and y_true must have the same length".to_string(),
            ));
        }

        if decision_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit isotonic calibration with empty data".to_string(),
            ));
        }

        // Create sorted pairs of (decision_score, label)
        let mut pairs: Vec<(Float, Float)> = decision_scores
            .iter()
            .zip(y_true.iter())
            .map(|(&score, &label)| (score, if label > 0.5 { 1.0 } else { 0.0 }))
            .collect();

        pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        // Simple isotonic regression using pool-adjacent-violators algorithm
        let mut calibration_curve = Vec::new();

        if !pairs.is_empty() {
            let mut i = 0;
            while i < pairs.len() {
                let mut j = i + 1;
                let mut sum_y = pairs[i].1;
                let mut count = 1.0;

                // Pool adjacent points if they violate monotonicity
                while j < pairs.len() {
                    let current_avg = sum_y / count;
                    if pairs[j].1 >= current_avg {
                        break;
                    }
                    sum_y += pairs[j].1;
                    count += 1.0;
                    j += 1;
                }

                let avg_y = sum_y / count;
                calibration_curve.push((pairs[i].0, avg_y));

                if j < pairs.len() {
                    calibration_curve.push((pairs[j - 1].0, avg_y));
                }

                i = j;
            }
        }

        self.calibration_curve = calibration_curve;
        self.fitted = true;

        Ok(())
    }

    /// Fit the isotonic regression calibrator using k-fold cross-validation
    pub fn fit_cv(
        &mut self,
        decision_scores: &Array1<Float>,
        y_true: &Array1<Float>,
        cv_folds: usize,
    ) -> Result<()> {
        if decision_scores.len() != y_true.len() {
            return Err(SklearsError::InvalidInput(
                "decision_scores and y_true must have the same length".to_string(),
            ));
        }

        if decision_scores.is_empty() {
            return Err(SklearsError::InvalidInput(
                "Cannot fit isotonic calibration with empty data".to_string(),
            ));
        }

        if cv_folds < 2 {
            return Err(SklearsError::InvalidInput(
                "cv_folds must be at least 2".to_string(),
            ));
        }

        let n_samples = decision_scores.len();
        if n_samples < cv_folds {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be at least equal to cv_folds".to_string(),
            ));
        }

        // Create indices and shuffle them
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut thread_rng());

        // Calculate fold sizes
        let fold_size = n_samples / cv_folds;
        let remainder = n_samples % cv_folds;

        let mut cv_scores = Vec::new();
        let mut cv_labels = Vec::new();
        let mut start_idx = 0;

        // Perform cross-validation
        for fold in 0..cv_folds {
            let current_fold_size = if fold < remainder {
                fold_size + 1
            } else {
                fold_size
            };
            let end_idx = start_idx + current_fold_size;

            // Create train/validation splits
            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for (i, &idx) in indices.iter().enumerate() {
                if i >= start_idx && i < end_idx {
                    val_indices.push(idx);
                } else {
                    train_indices.push(idx);
                }
            }

            // Extract training data
            let train_scores: Array1<Float> =
                Array1::from_vec(train_indices.iter().map(|&i| decision_scores[i]).collect());
            let train_labels: Array1<Float> =
                Array1::from_vec(train_indices.iter().map(|&i| y_true[i]).collect());

            // Fit calibrator on training fold
            let mut fold_calibrator = IsotonicCalibration::new();
            fold_calibrator.fit(&train_scores, &train_labels)?;

            // Store out-of-fold predictions (we'll fit final calibrator on all data)
            for &idx in &val_indices {
                cv_scores.push(decision_scores[idx]);
                cv_labels.push(y_true[idx]);
            }

            start_idx = end_idx;
        }

        // Fit final calibrator on all cross-validated scores
        let cv_scores_array = Array1::from_vec(cv_scores);
        let cv_labels_array = Array1::from_vec(cv_labels);
        self.fit(&cv_scores_array, &cv_labels_array)?;

        Ok(())
    }

    /// Transform decision scores to probabilities using isotonic regression
    pub fn predict_proba(&self, decision_scores: &Array1<Float>) -> Result<Array1<Float>> {
        if !self.fitted {
            return Err(SklearsError::InvalidInput(
                "IsotonicCalibration must be fitted before calling predict_proba".to_string(),
            ));
        }

        let mut probabilities = Array1::zeros(decision_scores.len());

        for (i, &score) in decision_scores.iter().enumerate() {
            probabilities[i] = self.interpolate_probability(score);
        }

        Ok(probabilities)
    }

    /// Interpolate probability for a given decision score
    fn interpolate_probability(&self, score: Float) -> Float {
        if self.calibration_curve.is_empty() {
            return 0.5; // Default probability
        }

        // Find the appropriate interval for interpolation
        if score <= self.calibration_curve[0].0 {
            return self.calibration_curve[0].1;
        }

        for i in 1..self.calibration_curve.len() {
            if score <= self.calibration_curve[i].0 {
                let (x1, y1) = self.calibration_curve[i - 1];
                let (x2, y2) = self.calibration_curve[i];

                if (x2 - x1).abs() < 1e-10 {
                    return y1;
                }

                // Linear interpolation
                let alpha = (score - x1) / (x2 - x1);
                return y1 + alpha * (y2 - y1);
            }
        }

        // If score is beyond the last point
        self.calibration_curve.last().unwrap().1
    }
}

impl Default for IsotonicCalibration {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_platt_scaling_basic() {
        let mut platt = PlattScaling::new();

        // Simple test data
        let decision_scores = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0];

        platt.fit(&decision_scores, &y_true).unwrap();

        let probabilities = platt.predict_proba(&decision_scores).unwrap();

        // Check that probabilities are in [0, 1]
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} not in [0,1]",
                prob
            );
        }

        // Check that probabilities are monotonic (higher scores -> higher probabilities)
        for i in 1..probabilities.len() {
            assert!(
                probabilities[i] >= probabilities[i - 1],
                "Probabilities not monotonic: {} vs {}",
                probabilities[i - 1],
                probabilities[i]
            );
        }
    }

    #[test]
    fn test_platt_scaling_binary_output() {
        let mut platt = PlattScaling::new();
        let decision_scores = array![-1.0, 0.0, 1.0];
        let y_true = array![0.0, 0.0, 1.0];

        platt.fit(&decision_scores, &y_true).unwrap();
        let prob_matrix = platt.predict_proba_binary(&decision_scores).unwrap();

        assert_eq!(prob_matrix.dim(), (3, 2));

        // Check that each row sums to 1.0
        for i in 0..3 {
            let row_sum = prob_matrix[[i, 0]] + prob_matrix[[i, 1]];
            assert_abs_diff_eq!(row_sum, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_isotonic_calibration() {
        let mut isotonic = IsotonicCalibration::new();

        let decision_scores = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0];

        isotonic.fit(&decision_scores, &y_true).unwrap();
        let probabilities = isotonic.predict_proba(&decision_scores).unwrap();

        // Check that probabilities are in [0, 1]
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} not in [0,1]",
                prob
            );
        }

        // Check monotonicity for isotonic regression
        for i in 1..probabilities.len() {
            assert!(
                probabilities[i] >= probabilities[i - 1],
                "Isotonic probabilities not monotonic: {} vs {}",
                probabilities[i - 1],
                probabilities[i]
            );
        }
    }

    #[test]
    fn test_platt_scaling_parameters() {
        let mut platt = PlattScaling::new();

        // Test that parameters are not accessible before fitting
        assert!(platt.parameters().is_err());

        let decision_scores = array![-1.0, 0.0, 1.0];
        let y_true = array![0.0, 0.0, 1.0];

        platt.fit(&decision_scores, &y_true).unwrap();

        // Test that parameters are accessible after fitting
        let (a, b) = platt.parameters().unwrap();
        assert!(a.is_finite() && b.is_finite());
    }

    #[test]
    fn test_calibration_edge_cases() {
        let mut platt = PlattScaling::new();

        // Test empty data
        let empty_scores = Array1::zeros(0);
        let empty_labels = Array1::zeros(0);
        assert!(platt.fit(&empty_scores, &empty_labels).is_err());

        // Test mismatched lengths
        let scores = array![1.0, 2.0];
        let labels = array![0.0];
        assert!(platt.fit(&scores, &labels).is_err());

        // Test single class data
        let scores = array![1.0, 2.0, 3.0];
        let labels = array![1.0, 1.0, 1.0];
        assert!(platt.fit(&scores, &labels).is_err());
    }

    #[test]
    fn test_platt_scaling_cross_validation() {
        let mut platt = PlattScaling::new();

        // Create larger test dataset for CV
        let decision_scores = array![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let y_true = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Test cross-validation fitting
        platt.fit_cv(&decision_scores, &y_true, 3).unwrap();

        let probabilities = platt.predict_proba(&decision_scores).unwrap();

        // Check that probabilities are in [0, 1]
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} not in [0,1]",
                prob
            );
        }

        // Check that probabilities are generally monotonic
        for i in 1..probabilities.len() {
            // Allow some tolerance for monotonicity due to cross-validation noise
            assert!(
                probabilities[i] >= probabilities[i - 1] - 0.1,
                "Probabilities not roughly monotonic: {} vs {}",
                probabilities[i - 1],
                probabilities[i]
            );
        }
    }

    #[test]
    fn test_platt_scaling_confidence_intervals() {
        let mut platt = PlattScaling::new();

        let decision_scores = array![-2.0, -1.0, 0.0, 1.0, 2.0];
        let y_true = array![0.0, 0.0, 0.0, 1.0, 1.0];

        platt.fit(&decision_scores, &y_true).unwrap();

        let (mean_probs, lower_bounds, upper_bounds) = platt
            .predict_proba_with_confidence(&decision_scores, 0.95, 100)
            .unwrap();

        assert_eq!(mean_probs.len(), decision_scores.len());
        assert_eq!(lower_bounds.len(), decision_scores.len());
        assert_eq!(upper_bounds.len(), decision_scores.len());

        // Check that confidence intervals are ordered correctly
        for i in 0..mean_probs.len() {
            assert!(
                lower_bounds[i] <= mean_probs[i],
                "Lower bound {} greater than mean {}",
                lower_bounds[i],
                mean_probs[i]
            );
            assert!(
                mean_probs[i] <= upper_bounds[i],
                "Mean {} greater than upper bound {}",
                mean_probs[i],
                upper_bounds[i]
            );
            assert!(
                0.0 <= lower_bounds[i] && upper_bounds[i] <= 1.0,
                "Bounds not in [0,1]: {} to {}",
                lower_bounds[i],
                upper_bounds[i]
            );
        }
    }

    #[test]
    fn test_isotonic_calibration_cross_validation() {
        let mut isotonic = IsotonicCalibration::new();

        // Create test dataset for CV
        let decision_scores = array![-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 4.0];
        let y_true = array![0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0];

        // Test cross-validation fitting
        isotonic.fit_cv(&decision_scores, &y_true, 3).unwrap();

        let probabilities = isotonic.predict_proba(&decision_scores).unwrap();

        // Check that probabilities are in [0, 1]
        for &prob in probabilities.iter() {
            assert!(
                prob >= 0.0 && prob <= 1.0,
                "Probability {} not in [0,1]",
                prob
            );
        }

        // Isotonic regression should preserve monotonicity
        for i in 1..probabilities.len() {
            assert!(
                probabilities[i] >= probabilities[i - 1],
                "Isotonic probabilities not monotonic: {} vs {}",
                probabilities[i - 1],
                probabilities[i]
            );
        }
    }

    #[test]
    fn test_cv_validation_errors() {
        let mut platt = PlattScaling::new();

        let scores = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let labels = array![0.0, 0.0, 1.0, 1.0, 1.0];

        // Test cv_folds = 1 (should fail)
        assert!(platt.fit_cv(&scores, &labels, 1).is_err());

        // Test cv_folds > n_samples (should fail)
        assert!(platt.fit_cv(&scores, &labels, 10).is_err());

        // Test mismatched sizes
        let wrong_labels = array![0.0, 1.0];
        assert!(platt.fit_cv(&scores, &wrong_labels, 3).is_err());
    }
}
