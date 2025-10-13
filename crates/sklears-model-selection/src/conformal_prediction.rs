//! Conformal prediction methods for uncertainty quantification
//!
//! This module provides conformal prediction algorithms that generate
//! prediction intervals with finite-sample validity guarantees.

use numrs2::prelude::*;
use sklears_core::error::{Result, SklearsError};
use std::cmp::Ordering;
use std::collections::HashMap;

/// Configuration for conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalPredictionConfig {
    /// Significance level (1 - coverage probability)
    pub alpha: f64,
    /// Method for computing nonconformity scores
    pub nonconformity_method: NonconformityMethod,
    /// Whether to use normalized nonconformity scores
    pub normalize: bool,
    /// Method for handling class imbalance in classification
    pub class_conditional: bool,
    /// Random state for reproducible results
    pub random_state: Option<u64>,
    /// Whether to use inductive (split) conformal prediction
    pub inductive: bool,
    /// Fraction of data to use for calibration in inductive setting
    pub calibration_fraction: f64,
}

impl Default for ConformalPredictionConfig {
    fn default() -> Self {
        Self {
            alpha: 0.1, // 90% coverage
            nonconformity_method: NonconformityMethod::AbsoluteError,
            normalize: false,
            class_conditional: false,
            random_state: None,
            inductive: true,
            calibration_fraction: 0.2,
        }
    }
}

/// Methods for computing nonconformity scores
#[derive(Debug, Clone)]
pub enum NonconformityMethod {
    /// Absolute error: |y - ŷ|
    AbsoluteError,
    /// Squared error: (y - ŷ)²
    SquaredError,
    /// Signed error: y - ŷ (for quantile regression)
    SignedError,
    /// Margin-based (for classification): margin to true class
    Margin,
    /// Inverse probability score (for classification)
    InverseProbability,
    /// Custom nonconformity function
    Custom(fn(&[f64], &[f64]) -> Vec<f64>),
}

/// Results from conformal prediction
#[derive(Debug, Clone)]
pub struct ConformalPredictionResult {
    /// Prediction intervals (lower, upper) for regression
    pub prediction_intervals: Option<Vec<(f64, f64)>>,
    /// Prediction sets for classification
    pub prediction_sets: Option<Vec<Vec<usize>>>,
    /// Nonconformity scores used for calibration
    pub calibration_scores: Vec<f64>,
    /// Quantile threshold used for prediction intervals
    pub quantile_threshold: f64,
    /// Coverage statistics
    pub coverage_stats: CoverageStatistics,
    /// Efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
}

/// Coverage statistics for conformal prediction
#[derive(Debug, Clone)]
pub struct CoverageStatistics {
    /// Empirical coverage rate
    pub empirical_coverage: f64,
    /// Target coverage rate (1 - alpha)
    pub target_coverage: f64,
    /// Coverage difference from target
    pub coverage_gap: f64,
    /// Coverage per class (for classification)
    pub class_coverage: Option<HashMap<usize, f64>>,
}

/// Efficiency metrics for conformal prediction
#[derive(Debug, Clone)]
pub struct EfficiencyMetrics {
    /// Average interval width (regression)
    pub average_interval_width: Option<f64>,
    /// Average set size (classification)
    pub average_set_size: Option<f64>,
    /// Interval width variability
    pub interval_width_std: Option<f64>,
    /// Set size variability
    pub set_size_std: Option<f64>,
    /// Singleton rate (classification: fraction with single prediction)
    pub singleton_rate: Option<f64>,
    /// Empty set rate (classification: fraction with no predictions)
    pub empty_set_rate: Option<f64>,
}

/// Conformal predictor for regression and classification
#[derive(Debug, Clone)]
pub struct ConformalPredictor {
    config: ConformalPredictionConfig,
    calibration_scores: Option<Vec<f64>>,
    quantile_threshold: Option<f64>,
    class_thresholds: Option<HashMap<usize, f64>>,
}

impl ConformalPredictor {
    pub fn new(config: ConformalPredictionConfig) -> Self {
        Self {
            config,
            calibration_scores: None,
            quantile_threshold: None,
            class_thresholds: None,
        }
    }

    /// Fit conformal predictor on calibration data
    pub fn fit(
        &mut self,
        calibration_predictions: &[f64],
        calibration_targets: &[f64],
    ) -> Result<()> {
        if calibration_predictions.len() != calibration_targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        // Compute nonconformity scores
        let scores =
            self.compute_nonconformity_scores(calibration_predictions, calibration_targets)?;

        // Calculate quantile threshold
        let quantile_level = 1.0 - self.config.alpha;
        let threshold = self.compute_quantile(&scores, quantile_level);

        self.calibration_scores = Some(scores);
        self.quantile_threshold = Some(threshold);

        Ok(())
    }

    /// Fit conformal predictor for classification
    pub fn fit_classification(
        &mut self,
        calibration_probabilities: &[Vec<f64>],
        calibration_labels: &[usize],
    ) -> Result<()> {
        if calibration_probabilities.len() != calibration_labels.len() {
            return Err(SklearsError::InvalidInput(
                "Probabilities and labels must have the same length".to_string(),
            ));
        }

        let scores =
            self.compute_classification_scores(calibration_probabilities, calibration_labels)?;

        if self.config.class_conditional {
            // Compute separate thresholds for each class
            let mut class_thresholds = HashMap::new();
            let unique_classes = self.get_unique_classes(calibration_labels);

            for &class in &unique_classes {
                let class_scores: Vec<f64> = scores
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| calibration_labels[*i] == class)
                    .map(|(_, &score)| score)
                    .collect();

                if !class_scores.is_empty() {
                    let quantile_level = 1.0 - self.config.alpha;
                    let threshold = self.compute_quantile(&class_scores, quantile_level);
                    class_thresholds.insert(class, threshold);
                }
            }

            self.class_thresholds = Some(class_thresholds);
        } else {
            // Single threshold for all classes
            let quantile_level = 1.0 - self.config.alpha;
            let threshold = self.compute_quantile(&scores, quantile_level);
            self.quantile_threshold = Some(threshold);
        }

        self.calibration_scores = Some(scores);

        Ok(())
    }

    /// Generate prediction intervals for regression
    pub fn predict_intervals(
        &self,
        predictions: &[f64],
        prediction_errors: Option<&[f64]>,
    ) -> Result<ConformalPredictionResult> {
        if self.quantile_threshold.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let threshold = self.quantile_threshold.unwrap();
        let mut intervals = Vec::new();

        for (i, &pred) in predictions.iter().enumerate() {
            let error_scale = if self.config.normalize && prediction_errors.is_some() {
                prediction_errors.unwrap()[i].max(1e-8) // Avoid division by zero
            } else {
                1.0
            };

            let margin = threshold * error_scale;
            intervals.push((pred - margin, pred + margin));
        }

        // Calculate efficiency metrics
        let average_width =
            intervals.iter().map(|(l, u)| u - l).sum::<f64>() / intervals.len() as f64;
        let width_std =
            self.calculate_std(&intervals.iter().map(|(l, u)| u - l).collect::<Vec<_>>());

        let efficiency_metrics = EfficiencyMetrics {
            average_interval_width: Some(average_width),
            average_set_size: None,
            interval_width_std: Some(width_std),
            set_size_std: None,
            singleton_rate: None,
            empty_set_rate: None,
        };

        let coverage_stats = CoverageStatistics {
            empirical_coverage: 0.0, // Would need true labels to compute
            target_coverage: 1.0 - self.config.alpha,
            coverage_gap: 0.0,
            class_coverage: None,
        };

        Ok(ConformalPredictionResult {
            prediction_intervals: Some(intervals),
            prediction_sets: None,
            calibration_scores: self.calibration_scores.clone().unwrap_or_default(),
            quantile_threshold: threshold,
            coverage_stats,
            efficiency_metrics,
        })
    }

    /// Generate prediction sets for classification
    pub fn predict_sets(
        &self,
        prediction_probabilities: &[Vec<f64>],
    ) -> Result<ConformalPredictionResult> {
        if self.quantile_threshold.is_none() && self.class_thresholds.is_none() {
            return Err(SklearsError::NotFitted {
                operation: "making predictions".to_string(),
            });
        }

        let mut prediction_sets = Vec::new();

        for probs in prediction_probabilities {
            let mut prediction_set = Vec::new();

            for (class_idx, &prob) in probs.iter().enumerate() {
                let threshold = if let Some(ref class_thresholds) = self.class_thresholds {
                    class_thresholds.get(&class_idx).copied().unwrap_or(0.0)
                } else {
                    self.quantile_threshold.unwrap()
                };

                // Include class if its score is below threshold
                let score = match self.config.nonconformity_method {
                    NonconformityMethod::InverseProbability => 1.0 - prob,
                    _ => 1.0 - prob, // Default for classification
                };

                if score <= threshold {
                    prediction_set.push(class_idx);
                }
            }

            prediction_sets.push(prediction_set);
        }

        // Calculate efficiency metrics
        let set_sizes: Vec<f64> = prediction_sets.iter().map(|s| s.len() as f64).collect();
        let average_set_size = set_sizes.iter().sum::<f64>() / set_sizes.len() as f64;
        let set_size_std = self.calculate_std(&set_sizes);
        let singleton_rate =
            set_sizes.iter().filter(|&&size| size == 1.0).count() as f64 / set_sizes.len() as f64;
        let empty_set_rate =
            set_sizes.iter().filter(|&&size| size == 0.0).count() as f64 / set_sizes.len() as f64;

        let efficiency_metrics = EfficiencyMetrics {
            average_interval_width: None,
            average_set_size: Some(average_set_size),
            interval_width_std: None,
            set_size_std: Some(set_size_std),
            singleton_rate: Some(singleton_rate),
            empty_set_rate: Some(empty_set_rate),
        };

        let coverage_stats = CoverageStatistics {
            empirical_coverage: 0.0, // Would need true labels to compute
            target_coverage: 1.0 - self.config.alpha,
            coverage_gap: 0.0,
            class_coverage: None,
        };

        let threshold = self.quantile_threshold.unwrap_or(0.0);

        Ok(ConformalPredictionResult {
            prediction_intervals: None,
            prediction_sets: Some(prediction_sets),
            calibration_scores: self.calibration_scores.clone().unwrap_or_default(),
            quantile_threshold: threshold,
            coverage_stats,
            efficiency_metrics,
        })
    }

    /// Evaluate coverage and efficiency on test data
    pub fn evaluate_coverage(
        &self,
        predictions: &[f64],
        true_values: &[f64],
        prediction_errors: Option<&[f64]>,
    ) -> Result<CoverageStatistics> {
        let result = self.predict_intervals(predictions, prediction_errors)?;
        let intervals = result.prediction_intervals.unwrap();

        let mut covered = 0;
        for (i, &true_val) in true_values.iter().enumerate() {
            let (lower, upper) = intervals[i];
            if true_val >= lower && true_val <= upper {
                covered += 1;
            }
        }

        let empirical_coverage = covered as f64 / true_values.len() as f64;
        let target_coverage = 1.0 - self.config.alpha;
        let coverage_gap = empirical_coverage - target_coverage;

        Ok(CoverageStatistics {
            empirical_coverage,
            target_coverage,
            coverage_gap,
            class_coverage: None,
        })
    }

    /// Evaluate classification coverage
    pub fn evaluate_classification_coverage(
        &self,
        prediction_probabilities: &[Vec<f64>],
        true_labels: &[usize],
    ) -> Result<CoverageStatistics> {
        let result = self.predict_sets(prediction_probabilities)?;
        let prediction_sets = result.prediction_sets.unwrap();

        let mut covered = 0;
        let mut class_coverage_counts: HashMap<usize, (usize, usize)> = HashMap::new();

        for (i, &true_label) in true_labels.iter().enumerate() {
            let prediction_set = &prediction_sets[i];
            let is_covered = prediction_set.contains(&true_label);

            if is_covered {
                covered += 1;
            }

            // Track class-specific coverage
            let (class_covered, class_total) =
                class_coverage_counts.entry(true_label).or_insert((0, 0));
            if is_covered {
                *class_covered += 1;
            }
            *class_total += 1;
        }

        let empirical_coverage = covered as f64 / true_labels.len() as f64;
        let target_coverage = 1.0 - self.config.alpha;
        let coverage_gap = empirical_coverage - target_coverage;

        // Calculate per-class coverage
        let mut class_coverage = HashMap::new();
        for (&class, &(covered_count, total_count)) in &class_coverage_counts {
            class_coverage.insert(class, covered_count as f64 / total_count as f64);
        }

        Ok(CoverageStatistics {
            empirical_coverage,
            target_coverage,
            coverage_gap,
            class_coverage: Some(class_coverage),
        })
    }

    /// Compute nonconformity scores
    fn compute_nonconformity_scores(
        &self,
        predictions: &[f64],
        targets: &[f64],
    ) -> Result<Vec<f64>> {
        match self.config.nonconformity_method {
            NonconformityMethod::AbsoluteError => Ok(predictions
                .iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (target - pred).abs())
                .collect()),
            NonconformityMethod::SquaredError => Ok(predictions
                .iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| (target - pred).powi(2))
                .collect()),
            NonconformityMethod::SignedError => Ok(predictions
                .iter()
                .zip(targets.iter())
                .map(|(&pred, &target)| target - pred)
                .collect()),
            NonconformityMethod::Custom(func) => Ok(func(predictions, targets)),
            _ => Err(SklearsError::InvalidInput(
                "Invalid nonconformity method for regression".to_string(),
            )),
        }
    }

    /// Compute classification nonconformity scores
    fn compute_classification_scores(
        &self,
        probabilities: &[Vec<f64>],
        labels: &[usize],
    ) -> Result<Vec<f64>> {
        match self.config.nonconformity_method {
            NonconformityMethod::InverseProbability => {
                let scores = probabilities
                    .iter()
                    .zip(labels.iter())
                    .map(|(probs, &label)| 1.0 - probs.get(label).copied().unwrap_or(0.0))
                    .collect();
                Ok(scores)
            }
            NonconformityMethod::Margin => {
                let scores = probabilities
                    .iter()
                    .zip(labels.iter())
                    .map(|(probs, &label)| {
                        let true_class_prob = probs.get(label).copied().unwrap_or(0.0);
                        let max_other_prob = probs
                            .iter()
                            .enumerate()
                            .filter(|(i, _)| *i != label)
                            .map(|(_, &prob)| prob)
                            .fold(0.0, f64::max);
                        max_other_prob - true_class_prob
                    })
                    .collect();
                Ok(scores)
            }
            _ => Err(SklearsError::InvalidInput(
                "Invalid nonconformity method for classification".to_string(),
            )),
        }
    }

    /// Compute quantile of a vector
    fn compute_quantile(&self, values: &[f64], quantile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }

        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));

        let n = sorted_values.len();
        let index = (quantile * (n + 1) as f64).ceil() as usize;
        let index = index.min(n).saturating_sub(1);

        sorted_values[index]
    }

    /// Get unique classes from labels
    fn get_unique_classes(&self, labels: &[usize]) -> Vec<usize> {
        let mut unique_classes: Vec<usize> = labels.to_vec();
        unique_classes.sort_unstable();
        unique_classes.dedup();
        unique_classes
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }
}

/// Jackknife+ conformal prediction for better efficiency
#[derive(Debug, Clone)]
pub struct JackknifeConformalPredictor {
    base_predictor: ConformalPredictor,
    jackknife_predictions: Option<Vec<Vec<f64>>>,
}

impl JackknifeConformalPredictor {
    pub fn new(config: ConformalPredictionConfig) -> Self {
        Self {
            base_predictor: ConformalPredictor::new(config),
            jackknife_predictions: None,
        }
    }

    /// Fit using jackknife+ method
    pub fn fit_jackknife(
        &mut self,
        all_predictions: &[Vec<f64>], // Predictions from leave-one-out models
        targets: &[f64],
    ) -> Result<()> {
        if all_predictions.len() != targets.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions and targets must have the same length".to_string(),
            ));
        }

        // Compute residuals for each jackknife prediction
        let mut residuals = Vec::new();
        for (i, preds) in all_predictions.iter().enumerate() {
            if !preds.is_empty() {
                let residual = (targets[i] - preds[0]).abs(); // Use first prediction
                residuals.push(residual);
            }
        }

        // Use residuals as calibration scores
        self.base_predictor.calibration_scores = Some(residuals.clone());
        let quantile_level = 1.0 - self.base_predictor.config.alpha;
        let threshold = self
            .base_predictor
            .compute_quantile(&residuals, quantile_level);
        self.base_predictor.quantile_threshold = Some(threshold);
        self.jackknife_predictions = Some(all_predictions.to_vec());

        Ok(())
    }

    /// Generate jackknife+ prediction intervals
    pub fn predict_jackknife_intervals(
        &self,
        predictions: &[f64],
    ) -> Result<ConformalPredictionResult> {
        self.base_predictor.predict_intervals(predictions, None)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_conformal_prediction_regression() {
        let config = ConformalPredictionConfig::default();
        let mut predictor = ConformalPredictor::new(config);

        // Create synthetic calibration data
        let cal_preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cal_targets = vec![1.1, 1.9, 3.2, 3.8, 5.1];

        predictor.fit(&cal_preds, &cal_targets).unwrap();

        // Make predictions
        let test_preds = vec![2.5, 4.5];
        let result = predictor.predict_intervals(&test_preds, None).unwrap();

        assert!(result.prediction_intervals.is_some());
        let intervals = result.prediction_intervals.unwrap();
        assert_eq!(intervals.len(), 2);

        // Check that intervals have positive width
        for (lower, upper) in intervals {
            assert!(upper > lower, "Interval should have positive width");
        }
    }

    #[test]
    fn test_conformal_prediction_classification() {
        let config = ConformalPredictionConfig {
            nonconformity_method: NonconformityMethod::InverseProbability,
            ..ConformalPredictionConfig::default()
        };
        let mut predictor = ConformalPredictor::new(config);

        // Create synthetic calibration data (3 classes)
        let cal_probs = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.2, 0.7, 0.1],
            vec![0.1, 0.2, 0.7],
            vec![0.6, 0.3, 0.1],
            vec![0.1, 0.1, 0.8],
        ];
        let cal_labels = vec![0, 1, 2, 0, 2];

        predictor
            .fit_classification(&cal_probs, &cal_labels)
            .unwrap();

        // Make predictions
        let test_probs = vec![vec![0.5, 0.3, 0.2], vec![0.2, 0.6, 0.2]];
        let result = predictor.predict_sets(&test_probs).unwrap();

        assert!(result.prediction_sets.is_some());
        let sets = result.prediction_sets.unwrap();
        assert_eq!(sets.len(), 2);
    }

    #[test]
    fn test_coverage_evaluation() {
        let config = ConformalPredictionConfig {
            alpha: 0.2, // 80% coverage
            ..Default::default()
        };
        let mut predictor = ConformalPredictor::new(config);

        // Create calibration data
        let cal_preds = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let cal_targets = vec![1.0, 2.0, 3.0, 4.0, 5.0]; // Perfect predictions

        predictor.fit(&cal_preds, &cal_targets).unwrap();

        // Evaluate on test data
        let test_preds = vec![1.5, 2.5];
        let test_targets = vec![1.5, 2.5]; // Also perfect
        let coverage = predictor
            .evaluate_coverage(&test_preds, &test_targets, None)
            .unwrap();

        assert!(coverage.empirical_coverage >= 0.0);
        assert!(coverage.empirical_coverage <= 1.0);
        assert_eq!(coverage.target_coverage, 0.8);
    }
}
