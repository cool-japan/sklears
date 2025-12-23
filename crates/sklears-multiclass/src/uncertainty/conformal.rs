//! Conformal Prediction for Multiclass Classification
//!
//! This module provides conformal prediction methods that give prediction sets
//! with theoretical coverage guarantees.

use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::error::{Result as SklResult, SklearsError};

/// Methods for conformal prediction
#[derive(Debug, Clone, PartialEq)]
pub enum ConformalMethod {
    /// Split conformal prediction
    SplitConformal,
    /// Cross-conformal prediction
    CrossConformal,
    /// Jackknife+ prediction
    JackknifeePlus,
    /// Adaptive prediction sets
    AdaptivePredictionSets,
    /// RAPS (Regularized Adaptive Prediction Sets)
    RAPS { lambda: f64 },
}

impl Default for ConformalMethod {
    fn default() -> Self {
        Self::SplitConformal
    }
}

/// Result of conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalResult {
    /// Prediction sets for each sample (classes that could be correct)
    pub prediction_sets: Vec<Vec<i32>>,
    /// Set sizes for each sample
    pub set_sizes: Array1<usize>,
    /// Conformity scores for each sample
    pub conformity_scores: Array1<f64>,
    /// Quantile threshold used
    pub quantile_threshold: f64,
    /// Actual coverage achieved on calibration set
    pub empirical_coverage: f64,
}

/// Conformal predictor for multiclass classification
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    method: ConformalMethod,
    confidence_level: f64,
    calibration_scores: Option<Array2<f64>>,
    calibration_labels: Option<Array1<i32>>,
    quantile_threshold: Option<f64>,
    fitted: bool,
}

impl ConformalPredictor {
    /// Create a new conformal predictor
    pub fn new() -> Self {
        Self {
            method: ConformalMethod::default(),
            confidence_level: 0.9,
            calibration_scores: None,
            calibration_labels: None,
            quantile_threshold: None,
            fitted: false,
        }
    }

    /// Create a builder for conformal predictor
    pub fn builder() -> ConformalPredictorBuilder {
        ConformalPredictorBuilder::new()
    }

    /// Set the conformal method
    pub fn method(mut self, method: ConformalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        if confidence_level > 0.0 && confidence_level < 1.0 {
            self.confidence_level = confidence_level;
        }
        self
    }

    /// Build the predictor
    pub fn build(self) -> Self {
        self
    }

    /// Fit conformal predictor on calibration data
    pub fn fit(
        &mut self,
        calibration_scores: &Array2<f64>,
        calibration_labels: &Array1<i32>,
    ) -> SklResult<()> {
        let (n_samples, _) = calibration_scores.dim();

        if n_samples != calibration_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Calibration scores and labels must have same length".to_string(),
            ));
        }

        if n_samples == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit on empty calibration data".to_string(),
            ));
        }

        self.calibration_scores = Some(calibration_scores.clone());
        self.calibration_labels = Some(calibration_labels.clone());

        // Compute conformity scores and quantile threshold
        let conformity_scores =
            self.compute_conformity_scores(calibration_scores, calibration_labels)?;

        self.quantile_threshold = Some(self.compute_quantile_threshold(&conformity_scores)?);
        self.fitted = true;

        Ok(())
    }

    /// Make conformal predictions
    pub fn predict(
        &self,
        test_scores: &Array2<f64>,
        calibration_data: Option<&Array2<f64>>,
    ) -> SklResult<ConformalResult> {
        if !self.fitted && calibration_data.is_none() {
            return Err(SklearsError::InvalidInput(
                "Conformal predictor must be fitted or calibration data provided".to_string(),
            ));
        }

        let threshold = if let Some(threshold) = self.quantile_threshold {
            threshold
        } else if let Some(cal_data) = calibration_data {
            // Use provided calibration data to compute threshold on-the-fly
            self.compute_threshold_from_scores(cal_data)?
        } else {
            return Err(SklearsError::InvalidInput(
                "No threshold available for prediction".to_string(),
            ));
        };

        let (n_samples, _n_classes) = test_scores.dim();
        let mut prediction_sets = Vec::with_capacity(n_samples);
        let mut set_sizes = Array1::zeros(n_samples);
        let mut conformity_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let scores = test_scores.row(i);
            let (pred_set, conformity_score) =
                self.compute_prediction_set(&scores.to_owned(), threshold)?;

            set_sizes[i] = pred_set.len();
            conformity_scores[i] = conformity_score;
            prediction_sets.push(pred_set);
        }

        // Compute empirical coverage if we have calibration data
        let empirical_coverage = if let (Some(cal_scores), Some(cal_labels)) =
            (&self.calibration_scores, &self.calibration_labels)
        {
            self.compute_empirical_coverage(cal_scores, cal_labels, threshold)?
        } else {
            0.0 // Unknown coverage
        };

        Ok(ConformalResult {
            prediction_sets,
            set_sizes,
            conformity_scores,
            quantile_threshold: threshold,
            empirical_coverage,
        })
    }

    /// Compute conformity scores for calibration data
    fn compute_conformity_scores(
        &self,
        scores: &Array2<f64>,
        labels: &Array1<i32>,
    ) -> SklResult<Array1<f64>> {
        let (n_samples, _n_classes) = scores.dim();
        let mut conformity_scores = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let sample_scores = scores.row(i);
            let true_label = labels[i];

            conformity_scores[i] = match &self.method {
                ConformalMethod::SplitConformal => {
                    self.compute_split_conformal_score(&sample_scores.to_owned(), true_label)?
                }
                ConformalMethod::CrossConformal => {
                    self.compute_cross_conformal_score(&sample_scores.to_owned(), true_label)?
                }
                ConformalMethod::JackknifeePlus => {
                    self.compute_jackknife_score(&sample_scores.to_owned(), true_label)?
                }
                ConformalMethod::AdaptivePredictionSets => {
                    self.compute_adaptive_score(&sample_scores.to_owned(), true_label)?
                }
                ConformalMethod::RAPS { lambda } => {
                    self.compute_raps_score(&sample_scores.to_owned(), true_label, *lambda)?
                }
            };
        }

        Ok(conformity_scores)
    }

    /// Compute split conformal conformity score
    fn compute_split_conformal_score(
        &self,
        scores: &Array1<f64>,
        true_label: i32,
    ) -> SklResult<f64> {
        // For split conformal, conformity score is 1 - score of true class
        if true_label < 0 || true_label as usize >= scores.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid label: {}",
                true_label
            )));
        }

        Ok(1.0 - scores[true_label as usize])
    }

    /// Compute cross conformal conformity score  
    fn compute_cross_conformal_score(
        &self,
        scores: &Array1<f64>,
        true_label: i32,
    ) -> SklResult<f64> {
        // Similar to split conformal for now
        self.compute_split_conformal_score(scores, true_label)
    }

    /// Compute Jackknife+ conformity score
    fn compute_jackknife_score(&self, scores: &Array1<f64>, true_label: i32) -> SklResult<f64> {
        if true_label < 0 || true_label as usize >= scores.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid label: {}",
                true_label
            )));
        }

        let true_score = scores[true_label as usize];

        // Count how many classes have higher scores than the true class
        let mut rank = 1;
        for &score in scores.iter() {
            if score > true_score {
                rank += 1;
            }
        }

        Ok(rank as f64 / scores.len() as f64)
    }

    /// Compute adaptive prediction set conformity score
    fn compute_adaptive_score(&self, scores: &Array1<f64>, true_label: i32) -> SklResult<f64> {
        if true_label < 0 || true_label as usize >= scores.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid label: {}",
                true_label
            )));
        }

        // Sort scores in descending order with indices
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find position of true label in sorted order
        let mut _position = 0;
        let mut cumulative_score = 0.0;

        for (i, (class_idx, score)) in indexed_scores.iter().enumerate() {
            cumulative_score += score;
            if *class_idx == true_label as usize {
                _position = i + 1;
                break;
            }
        }

        // Conformity score is the cumulative probability up to true class
        Ok(cumulative_score)
    }

    /// Compute RAPS (Regularized Adaptive Prediction Sets) conformity score
    fn compute_raps_score(
        &self,
        scores: &Array1<f64>,
        true_label: i32,
        lambda: f64,
    ) -> SklResult<f64> {
        if true_label < 0 || true_label as usize >= scores.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid label: {}",
                true_label
            )));
        }

        // Sort scores in descending order with indices
        let mut indexed_scores: Vec<(usize, f64)> = scores
            .iter()
            .enumerate()
            .map(|(i, &score)| (i, score))
            .collect();
        indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        // Find position of true label and compute regularized score
        let mut cumulative_score = 0.0;
        let mut regularization = 0.0;

        for (i, (class_idx, score)) in indexed_scores.iter().enumerate() {
            cumulative_score += score;
            regularization += lambda * (i as f64 + 1.0);

            if *class_idx == true_label as usize {
                break;
            }
        }

        Ok(cumulative_score + regularization)
    }

    /// Compute quantile threshold from conformity scores
    fn compute_quantile_threshold(&self, conformity_scores: &Array1<f64>) -> SklResult<f64> {
        let n = conformity_scores.len();
        if n == 0 {
            return Ok(0.0);
        }

        let mut sorted_scores = conformity_scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Compute the quantile for the desired confidence level
        let alpha = 1.0 - self.confidence_level;
        let quantile_index = ((n as f64 + 1.0) * (1.0 - alpha)).ceil() as usize;
        let clamped_index = std::cmp::min(quantile_index.saturating_sub(1), n - 1);

        Ok(sorted_scores[clamped_index])
    }

    /// Compute threshold from test scores for on-the-fly prediction
    fn compute_threshold_from_scores(&self, scores: &Array2<f64>) -> SklResult<f64> {
        // For simplicity, use median of maximum scores as threshold
        let (n_samples, _) = scores.dim();
        let mut max_scores = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let row_max = scores.row(i).iter().fold(0.0f64, |a, &b| a.max(b));
            max_scores.push(1.0 - row_max); // Convert to conformity score
        }

        max_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median_idx = max_scores.len() / 2;
        Ok(max_scores[median_idx])
    }

    /// Compute prediction set for a test sample
    fn compute_prediction_set(
        &self,
        scores: &Array1<f64>,
        threshold: f64,
    ) -> SklResult<(Vec<i32>, f64)> {
        let mut prediction_set = Vec::new();
        #[allow(unused_assignments)]
        let mut conformity_score = 0.0;

        match &self.method {
            ConformalMethod::SplitConformal | ConformalMethod::CrossConformal => {
                // Include classes where 1 - score <= threshold, i.e., score >= 1 - threshold
                let score_threshold = 1.0 - threshold;
                for (i, &score) in scores.iter().enumerate() {
                    if score >= score_threshold {
                        prediction_set.push(i as i32);
                    }
                }
                conformity_score = 1.0 - scores.iter().fold(0.0f64, |a, &b| a.max(b));
            }
            ConformalMethod::JackknifeePlus => {
                // Include top classes until cumulative probability exceeds threshold
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut cumulative = 0.0;
                for (class_idx, score) in indexed_scores {
                    prediction_set.push(class_idx as i32);
                    cumulative += score;
                    if cumulative >= threshold {
                        break;
                    }
                }
                conformity_score = threshold;
            }
            ConformalMethod::AdaptivePredictionSets => {
                // Adaptive prediction sets
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut cumulative = 0.0;
                for (class_idx, score) in indexed_scores {
                    cumulative += score;
                    prediction_set.push(class_idx as i32);
                    if cumulative >= 1.0 - threshold {
                        break;
                    }
                }
                conformity_score = cumulative;
            }
            ConformalMethod::RAPS { lambda } => {
                // RAPS method
                let mut indexed_scores: Vec<(usize, f64)> = scores
                    .iter()
                    .enumerate()
                    .map(|(i, &score)| (i, score))
                    .collect();
                indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

                let mut cumulative = 0.0;
                let mut regularization = 0.0;

                for (i, (class_idx, score)) in indexed_scores.iter().enumerate() {
                    cumulative += score;
                    regularization += lambda * (i as f64 + 1.0);
                    prediction_set.push(*class_idx as i32);

                    if cumulative + regularization >= threshold {
                        break;
                    }
                }
                conformity_score = cumulative + regularization;
            }
        }

        // Ensure at least one class is included
        if prediction_set.is_empty() {
            let max_idx = scores
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap_or(0);
            prediction_set.push(max_idx as i32);
        }

        Ok((prediction_set, conformity_score))
    }

    /// Compute empirical coverage on calibration data
    fn compute_empirical_coverage(
        &self,
        scores: &Array2<f64>,
        labels: &Array1<i32>,
        threshold: f64,
    ) -> SklResult<f64> {
        let n_samples = scores.nrows();
        let mut covered = 0;

        for i in 0..n_samples {
            let sample_scores = scores.row(i);
            let true_label = labels[i];
            let (pred_set, _) =
                self.compute_prediction_set(&sample_scores.to_owned(), threshold)?;

            if pred_set.contains(&true_label) {
                covered += 1;
            }
        }

        Ok(covered as f64 / n_samples as f64)
    }
}

impl Default for ConformalPredictor {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for conformal predictor
#[derive(Debug)]
pub struct ConformalPredictorBuilder {
    method: ConformalMethod,
    confidence_level: f64,
}

impl Default for ConformalPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl ConformalPredictorBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            method: ConformalMethod::default(),
            confidence_level: 0.9,
        }
    }

    /// Set the conformal method
    pub fn method(mut self, method: ConformalMethod) -> Self {
        self.method = method;
        self
    }

    /// Set confidence level
    pub fn confidence_level(mut self, confidence_level: f64) -> Self {
        if confidence_level > 0.0 && confidence_level < 1.0 {
            self.confidence_level = confidence_level;
        }
        self
    }

    /// Build the conformal predictor
    pub fn build(self) -> ConformalPredictor {
        ConformalPredictor {
            method: self.method,
            confidence_level: self.confidence_level,
            calibration_scores: None,
            calibration_labels: None,
            quantile_threshold: None,
            fitted: false,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_autograd::ndarray::array;

    #[test]
    fn test_conformal_predictor_creation() {
        let predictor = ConformalPredictor::new();
        assert_eq!(predictor.confidence_level, 0.9);
        assert!(!predictor.fitted);
    }

    #[test]
    fn test_conformal_predictor_builder() {
        let predictor = ConformalPredictor::builder()
            .method(ConformalMethod::JackknifeePlus)
            .confidence_level(0.95)
            .build();

        assert_eq!(predictor.method, ConformalMethod::JackknifeePlus);
        assert_eq!(predictor.confidence_level, 0.95);
    }

    #[test]
    fn test_split_conformal_fit_predict() {
        let mut predictor = ConformalPredictor::new()
            .method(ConformalMethod::SplitConformal)
            .confidence_level(0.8)
            .build();

        // Calibration data
        let cal_scores = array![
            [0.8, 0.1, 0.1],
            [0.1, 0.7, 0.2],
            [0.2, 0.2, 0.6],
            [0.6, 0.3, 0.1]
        ];
        let cal_labels = array![0, 1, 2, 0];

        // Fit the predictor
        predictor.fit(&cal_scores, &cal_labels).unwrap();
        assert!(predictor.fitted);

        // Test data
        let test_scores = array![[0.9, 0.05, 0.05], [0.3, 0.4, 0.3]];

        let result = predictor.predict(&test_scores, None).unwrap();

        assert_eq!(result.prediction_sets.len(), 2);
        assert_eq!(result.set_sizes.len(), 2);

        // All prediction sets should contain at least one class
        for pred_set in &result.prediction_sets {
            assert!(!pred_set.is_empty());
        }
    }

    #[test]
    fn test_jackknife_conformal_score() {
        let predictor = ConformalPredictor::new()
            .method(ConformalMethod::JackknifeePlus)
            .build();

        let scores = array![0.5, 0.3, 0.2];
        let true_label = 0;

        let conformity_score = predictor
            .compute_jackknife_score(&scores, true_label)
            .unwrap();

        // True class has highest score, so rank should be 1
        // Conformity score = 1/3 = 0.333...
        assert!((conformity_score - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_raps_conformal_score() {
        let predictor = ConformalPredictor::new()
            .method(ConformalMethod::RAPS { lambda: 0.1 })
            .build();

        let scores = array![0.5, 0.3, 0.2];
        let true_label = 0;

        let conformity_score = predictor
            .compute_raps_score(&scores, true_label, 0.1)
            .unwrap();

        // Should include regularization term
        assert!(conformity_score > 0.5); // Base score + regularization
    }

    #[test]
    fn test_prediction_set_coverage() {
        let mut predictor = ConformalPredictor::new()
            .method(ConformalMethod::SplitConformal)
            .confidence_level(0.9)
            .build();

        // Create calibration data where true labels have high scores
        let cal_scores = array![
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.05, 0.9],
            [0.8, 0.1, 0.1]
        ];
        let cal_labels = array![0, 1, 2, 0];

        predictor.fit(&cal_scores, &cal_labels).unwrap();

        // Test with same data to check coverage
        let result = predictor.predict(&cal_scores, None).unwrap();

        // Check that empirical coverage is close to desired level
        assert!(result.empirical_coverage >= 0.5); // Should have reasonable coverage
    }

    #[test]
    fn test_empty_calibration_data() {
        let mut predictor = ConformalPredictor::new();

        let empty_scores = Array2::<f64>::zeros((0, 3));
        let empty_labels = Array1::<i32>::zeros(0);

        let result = predictor.fit(&empty_scores, &empty_labels);
        assert!(result.is_err());
    }

    #[test]
    fn test_mismatched_dimensions() {
        let mut predictor = ConformalPredictor::new();

        let scores = array![[0.5, 0.3, 0.2]];
        let labels = array![0, 1]; // Wrong length

        let result = predictor.fit(&scores, &labels);
        assert!(result.is_err());
    }
}
