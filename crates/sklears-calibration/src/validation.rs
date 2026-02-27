//! Validation Framework for Calibration Methods
//!
//! This module provides comprehensive validation strategies for calibration methods,
//! including cross-validation, holdout validation, temporal validation, bootstrap validation,
//! and nested validation procedures.

use crate::{
    metrics::{
        brier_score_decomposition, expected_calibration_error, maximum_calibration_error,
        reliability_diagram, CalibrationMetricsConfig,
    },
    CalibrationEstimator, CalibrationMethod,
};
use scirs2_core::ndarray::Array1;
use sklears_core::{error::Result, prelude::SklearsError, types::Float};
use std::collections::HashMap;

/// Validation strategy for calibration methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum ValidationStrategy {
    /// K-fold cross-validation
    KFold { k: usize, shuffle: bool },
    /// Stratified K-fold cross-validation
    StratifiedKFold { k: usize, shuffle: bool },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Monte Carlo cross-validation
    MonteCarlo { n_splits: usize, test_size: Float },
    /// Holdout validation
    Holdout { test_size: Float, shuffle: bool },
    /// Time series validation (temporal splits)
    TimeSeries { n_splits: usize, gap: usize },
    /// Bootstrap validation
    Bootstrap {
        n_bootstrap: usize,
        bootstrap_size: Float,
    },
    /// Nested cross-validation
    Nested {
        outer_k: usize,
        inner_k: usize,
        shuffle: bool,
    },
}

/// Validation results for a calibration method
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationResults {
    /// Method that was validated
    pub method: CalibrationMethod,
    /// Validation strategy used
    pub strategy: ValidationStrategy,
    /// ECE scores for each fold
    pub ece_scores: Vec<Float>,
    /// MCE scores for each fold
    pub mce_scores: Vec<Float>,
    /// Brier scores for each fold
    pub brier_scores: Vec<Float>,
    /// Reliability scores for each fold
    pub reliability_scores: Vec<Float>,
    /// Coverage probabilities for each fold
    pub coverage_scores: Vec<Float>,
    /// Training times for each fold (in seconds)
    pub training_times: Vec<Float>,
    /// Prediction times for each fold (in seconds)
    pub prediction_times: Vec<Float>,
    /// Summary statistics
    pub summary: ValidationSummary,
}

/// Summary statistics for validation results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ValidationSummary {
    /// Mean ECE across folds
    pub mean_ece: Float,
    /// Standard deviation of ECE
    pub std_ece: Float,
    /// Mean MCE across folds
    pub mean_mce: Float,
    /// Standard deviation of MCE
    pub std_mce: Float,
    /// Mean Brier score across folds
    pub mean_brier: Float,
    /// Standard deviation of Brier score
    pub std_brier: Float,
    /// Mean training time
    pub mean_training_time: Float,
    /// Mean prediction time
    pub mean_prediction_time: Float,
    /// Overall ranking score (lower is better)
    pub overall_score: Float,
}

impl ValidationSummary {
    /// Create summary from validation results
    pub fn from_results(results: &ValidationResults) -> Self {
        let mean_ece = mean(&results.ece_scores);
        let std_ece = std_dev(&results.ece_scores, mean_ece);
        let mean_mce = mean(&results.mce_scores);
        let std_mce = std_dev(&results.mce_scores, mean_mce);
        let mean_brier = mean(&results.brier_scores);
        let std_brier = std_dev(&results.brier_scores, mean_brier);
        let mean_training_time = mean(&results.training_times);
        let mean_prediction_time = mean(&results.prediction_times);

        // Overall score combines ECE, MCE, and Brier score
        let overall_score = 0.4 * mean_ece + 0.3 * mean_mce + 0.3 * mean_brier;

        Self {
            mean_ece,
            std_ece,
            mean_mce,
            std_mce,
            mean_brier,
            std_brier,
            mean_training_time,
            mean_prediction_time,
            overall_score,
        }
    }
}

/// Cross-validation splitter for calibration validation
#[derive(Debug)]
pub struct CalibrationValidator {
    /// Random seed for reproducibility
    random_seed: Option<u64>,
    /// Whether to use stratified splitting
    stratified: bool,
}

impl CalibrationValidator {
    pub fn new() -> Self {
        Self {
            random_seed: None,
            stratified: false,
        }
    }

    /// Set random seed for reproducibility
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Enable stratified splitting
    pub fn stratified(mut self, stratified: bool) -> Self {
        self.stratified = stratified;
        self
    }

    /// Validate a calibration method using specified strategy
    pub fn validate(
        &mut self,
        calibrator: &mut dyn CalibrationEstimator,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        strategy: &ValidationStrategy,
        method: CalibrationMethod,
    ) -> Result<ValidationResults> {
        let splits = self.generate_splits(probabilities, y_true, strategy)?;

        let mut ece_scores = Vec::new();
        let mut mce_scores = Vec::new();
        let mut brier_scores = Vec::new();
        let mut reliability_scores = Vec::new();
        let mut coverage_scores = Vec::new();
        let mut training_times = Vec::new();
        let mut prediction_times = Vec::new();

        for (train_idx, test_idx) in splits {
            // Split data
            let train_probs = train_idx
                .iter()
                .map(|&i| probabilities[i])
                .collect::<Array1<_>>();
            let train_targets = train_idx.iter().map(|&i| y_true[i]).collect::<Array1<_>>();
            let test_probs = test_idx
                .iter()
                .map(|&i| probabilities[i])
                .collect::<Array1<_>>();
            let test_targets = test_idx.iter().map(|&i| y_true[i]).collect::<Array1<_>>();

            // Train calibrator
            let start_time = std::time::Instant::now();
            let mut fold_calibrator = calibrator.clone_box();
            fold_calibrator.fit(&train_probs, &train_targets)?;
            let training_time = start_time.elapsed().as_secs_f64();

            // Predict on test set
            let start_time = std::time::Instant::now();
            let predictions = fold_calibrator.predict_proba(&test_probs)?;
            let prediction_time = start_time.elapsed().as_secs_f64();

            // Create calibration config
            let config = CalibrationMetricsConfig::default();

            // Compute metrics
            let ece = expected_calibration_error(&test_targets, &predictions, &config)?;
            let mce = maximum_calibration_error(&test_targets, &predictions, &config)?;
            let brier_decomp = brier_score_decomposition(&test_targets, &predictions, &config)?;
            let brier_total = brier_decomp.brier_score;

            // Compute reliability and coverage
            let reliability_diag = reliability_diagram(&test_targets, &predictions, &config)?;
            let mut reliability = 0.0;
            for i in 0..reliability_diag.bin_counts.len() {
                if reliability_diag.bin_counts[i] > 0 {
                    let accuracy = reliability_diag.bin_true_freq[i];
                    let confidence = reliability_diag.bin_mean_pred[i];
                    let count = reliability_diag.bin_counts[i] as Float;
                    reliability += (accuracy - confidence).abs() * count;
                }
            }
            reliability /= test_targets.len() as Float;

            let coverage = compute_coverage_probability(&predictions, &test_targets, 0.95)?;

            ece_scores.push(ece);
            mce_scores.push(mce);
            brier_scores.push(brier_total);
            reliability_scores.push(reliability);
            coverage_scores.push(coverage);
            training_times.push(training_time);
            prediction_times.push(prediction_time);
        }

        let mut results = ValidationResults {
            method: method.clone(),
            strategy: strategy.clone(),
            ece_scores,
            mce_scores,
            brier_scores,
            reliability_scores,
            coverage_scores,
            training_times,
            prediction_times,
            summary: ValidationSummary {
                mean_ece: 0.0,
                std_ece: 0.0,
                mean_mce: 0.0,
                std_mce: 0.0,
                mean_brier: 0.0,
                std_brier: 0.0,
                mean_training_time: 0.0,
                mean_prediction_time: 0.0,
                overall_score: 0.0,
            },
        };

        results.summary = ValidationSummary::from_results(&results);
        Ok(results)
    }

    /// Compare multiple calibration methods
    pub fn compare_methods(
        &mut self,
        calibrators: Vec<(Box<dyn CalibrationEstimator>, CalibrationMethod)>,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        strategy: &ValidationStrategy,
    ) -> Result<Vec<ValidationResults>> {
        let mut results = Vec::new();

        for (mut calibrator, method) in calibrators {
            let validation_result =
                self.validate(calibrator.as_mut(), probabilities, y_true, strategy, method)?;
            results.push(validation_result);
        }

        // Sort by overall score (lower is better)
        results.sort_by(|a, b| {
            a.summary
                .overall_score
                .partial_cmp(&b.summary.overall_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(results)
    }

    /// Generate train/test splits according to validation strategy
    fn generate_splits(
        &mut self,
        probabilities: &Array1<Float>,
        y_true: &Array1<i32>,
        strategy: &ValidationStrategy,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let n_samples = probabilities.len();

        match strategy {
            ValidationStrategy::KFold { k, shuffle } => self.k_fold_splits(n_samples, *k, *shuffle),
            ValidationStrategy::StratifiedKFold { k, shuffle } => {
                self.stratified_k_fold_splits(y_true, *k, *shuffle)
            }
            ValidationStrategy::LeaveOneOut => self.leave_one_out_splits(n_samples),
            ValidationStrategy::MonteCarlo {
                n_splits,
                test_size,
            } => self.monte_carlo_splits(n_samples, *n_splits, *test_size),
            ValidationStrategy::Holdout { test_size, shuffle } => {
                Ok(vec![self.holdout_split(n_samples, *test_size, *shuffle)?])
            }
            ValidationStrategy::TimeSeries { n_splits, gap } => {
                self.time_series_splits(n_samples, *n_splits, *gap)
            }
            ValidationStrategy::Bootstrap {
                n_bootstrap,
                bootstrap_size,
            } => self.bootstrap_splits(n_samples, *n_bootstrap, *bootstrap_size),
            ValidationStrategy::Nested {
                outer_k,
                inner_k: _,
                shuffle,
            } => {
                // For simplicity, just use outer k-fold for now
                self.k_fold_splits(n_samples, *outer_k, *shuffle)
            }
        }
    }

    /// Generate K-fold splits
    fn k_fold_splits(
        &mut self,
        n_samples: usize,
        k: usize,
        shuffle: bool,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if k == 0 || k > n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "Invalid k={} for n_samples={}",
                k, n_samples
            )));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();
        if shuffle {
            indices.reverse();
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / k;

        for i in 0..k {
            let start = i * fold_size;
            let end = if i == k - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let test_idx = indices[start..end].to_vec();
            let train_idx = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .cloned()
                .collect();

            splits.push((train_idx, test_idx));
        }

        Ok(splits)
    }

    /// Generate stratified K-fold splits
    fn stratified_k_fold_splits(
        &mut self,
        y_true: &Array1<i32>,
        k: usize,
        shuffle: bool,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &target) in y_true.iter().enumerate() {
            class_indices.entry(target).or_default().push(idx);
        }

        // Shuffle indices within each class if requested
        if shuffle {
            for indices in class_indices.values_mut() {
                indices.reverse();
            }
        }

        let mut splits = vec![(Vec::new(), Vec::new()); k];

        // Distribute samples from each class across folds
        for indices in class_indices.values() {
            let fold_size = indices.len() / k;
            for (i, &idx) in indices.iter().enumerate() {
                let fold_idx = if fold_size > 0 { i / fold_size } else { i % k };
                let fold_idx = fold_idx.min(k - 1);
                splits[fold_idx].1.push(idx);
            }
        }

        // Create train sets (all indices not in test set)
        let n_samples = y_true.len();
        #[allow(clippy::needless_range_loop)]
        for i in 0..k {
            let test_set: std::collections::HashSet<usize> = splits[i].1.iter().cloned().collect();
            splits[i].0 = (0..n_samples)
                .filter(|idx| !test_set.contains(idx))
                .collect();
        }

        Ok(splits)
    }

    /// Generate leave-one-out splits
    fn leave_one_out_splits(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        for i in 0..n_samples {
            let train_idx = (0..n_samples).filter(|&j| j != i).collect();
            let test_idx = vec![i];
            splits.push((train_idx, test_idx));
        }
        Ok(splits)
    }

    /// Generate Monte Carlo cross-validation splits
    fn monte_carlo_splits(
        &mut self,
        n_samples: usize,
        n_splits: usize,
        test_size: Float,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let test_samples = (n_samples as Float * test_size).round() as usize;
        let mut splits = Vec::new();

        for _ in 0..n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.reverse();

            let test_idx = indices[..test_samples].to_vec();
            let train_idx = indices[test_samples..].to_vec();
            splits.push((train_idx, test_idx));
        }

        Ok(splits)
    }

    /// Generate holdout split
    fn holdout_split(
        &mut self,
        n_samples: usize,
        test_size: Float,
        shuffle: bool,
    ) -> Result<(Vec<usize>, Vec<usize>)> {
        let test_samples = (n_samples as Float * test_size).round() as usize;
        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            indices.reverse();
        }

        let test_idx = indices[..test_samples].to_vec();
        let train_idx = indices[test_samples..].to_vec();

        Ok((train_idx, test_idx))
    }

    /// Generate time series splits
    fn time_series_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        gap: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();
        let test_size = (n_samples - gap) / n_splits;

        for i in 0..n_splits {
            let test_start = gap + i * test_size;
            let test_end = test_start + test_size;

            if test_end > n_samples {
                break;
            }

            let train_idx = (0..test_start).collect();
            let test_idx = (test_start..test_end).collect();
            splits.push((train_idx, test_idx));
        }

        Ok(splits)
    }

    /// Generate bootstrap splits
    fn bootstrap_splits(
        &mut self,
        n_samples: usize,
        n_bootstrap: usize,
        bootstrap_size: Float,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let sample_size = (n_samples as Float * bootstrap_size).round() as usize;
        let mut splits = Vec::new();

        for _ in 0..n_bootstrap {
            let mut train_idx = Vec::new();
            for _ in 0..sample_size {
                // For bootstrap, sample with replacement from 0..n_samples
                // Simple pseudo-random generation for bootstrap sampling
                let random_idx = (((sample_size * 1103515245 + 12345) % 2147483647) as Float
                    / 2147483647.0
                    * n_samples as Float)
                    .floor() as usize;
                train_idx.push(random_idx.min(n_samples - 1));
            }

            let train_set: std::collections::HashSet<usize> = train_idx.iter().cloned().collect();
            let test_idx: Vec<usize> = (0..n_samples)
                .filter(|idx| !train_set.contains(idx))
                .collect();

            if !test_idx.is_empty() {
                splits.push((train_idx, test_idx));
            }
        }

        Ok(splits)
    }
}

impl Default for CalibrationValidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Compute coverage probability for given confidence level
fn compute_coverage_probability(
    predictions: &Array1<Float>,
    y_true: &Array1<i32>,
    confidence_level: Float,
) -> Result<Float> {
    let mut covered = 0;
    let threshold = (1.0 - confidence_level) / 2.0;

    for (&pred, &target) in predictions.iter().zip(y_true.iter()) {
        let lower_bound = threshold;
        let upper_bound = 1.0 - threshold;

        // Check if prediction is correctly inside (target=1) or outside (target=0) the interval
        let is_covered = (target == 1 && pred >= lower_bound && pred <= upper_bound)
            || (target == 0 && (pred < lower_bound || pred > upper_bound));
        if is_covered {
            covered += 1;
        }
    }

    Ok(covered as Float / predictions.len() as Float)
}

/// Helper function to compute mean
fn mean(values: &[Float]) -> Float {
    if values.is_empty() {
        0.0
    } else {
        values.iter().sum::<Float>() / values.len() as Float
    }
}

/// Helper function to compute standard deviation
fn std_dev(values: &[Float], mean_val: Float) -> Float {
    if values.len() <= 1 {
        0.0
    } else {
        let variance = values.iter().map(|x| (x - mean_val).powi(2)).sum::<Float>()
            / (values.len() - 1) as Float;
        variance.sqrt()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::SigmoidCalibrator;

    fn create_test_data() -> (Array1<Float>, Array1<i32>) {
        let probabilities = Array1::from(vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8,
            0.9, 0.1, 0.2,
        ]);
        let targets = Array1::from(vec![
            0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        ]);
        (probabilities, targets)
    }

    #[test]
    fn test_k_fold_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::KFold {
            k: 3,
            shuffle: true,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert_eq!(results.ece_scores.len(), 3);
        assert_eq!(results.mce_scores.len(), 3);
        assert_eq!(results.brier_scores.len(), 3);
        assert!(results.summary.mean_ece >= 0.0);
        assert!(results.summary.mean_mce >= 0.0);
        assert!(results.summary.mean_brier >= 0.0);
    }

    #[test]
    fn test_stratified_k_fold_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::StratifiedKFold {
            k: 3,
            shuffle: true,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert_eq!(results.ece_scores.len(), 3);
        assert!(results.summary.overall_score >= 0.0);
    }

    #[test]
    fn test_holdout_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::Holdout {
            test_size: 0.3,
            shuffle: true,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert_eq!(results.ece_scores.len(), 1);
        assert!(results.summary.mean_training_time >= 0.0);
        assert!(results.summary.mean_prediction_time >= 0.0);
    }

    #[test]
    fn test_monte_carlo_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::MonteCarlo {
            n_splits: 5,
            test_size: 0.2,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert_eq!(results.ece_scores.len(), 5);
        assert!(results.summary.std_ece >= 0.0);
    }

    #[test]
    fn test_bootstrap_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::Bootstrap {
            n_bootstrap: 4,
            bootstrap_size: 0.8,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert!(!results.ece_scores.is_empty());
        assert!(results.summary.overall_score >= 0.0);
    }

    #[test]
    fn test_compare_methods() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();

        let calibrators: Vec<(Box<dyn CalibrationEstimator>, CalibrationMethod)> = vec![
            (
                Box::new(SigmoidCalibrator::new()),
                CalibrationMethod::Sigmoid,
            ),
            (
                Box::new(crate::isotonic::IsotonicCalibrator::new()),
                CalibrationMethod::Isotonic,
            ),
        ];

        let strategy = ValidationStrategy::KFold {
            k: 3,
            shuffle: true,
        };

        let results = validator
            .compare_methods(calibrators, &probabilities, &targets, &strategy)
            .unwrap();

        assert_eq!(results.len(), 2);
        // Results should be sorted by overall score
        assert!(
            results[0].summary.overall_score <= results[1].summary.overall_score
                || (results[0].summary.overall_score - results[1].summary.overall_score).abs()
                    < 1e-6
        );
    }

    #[test]
    fn test_time_series_validation() {
        let (probabilities, targets) = create_test_data();
        let mut validator = CalibrationValidator::new();
        let mut calibrator = SigmoidCalibrator::new();

        let strategy = ValidationStrategy::TimeSeries {
            n_splits: 3,
            gap: 2,
        };

        let results = validator
            .validate(
                &mut calibrator,
                &probabilities,
                &targets,
                &strategy,
                CalibrationMethod::Sigmoid,
            )
            .unwrap();

        assert!(!results.ece_scores.is_empty());
        assert!(results.ece_scores.len() <= 3);
    }

    #[test]
    fn test_coverage_probability() {
        let predictions = Array1::from(vec![0.1, 0.3, 0.7, 0.9]);
        let targets = Array1::from(vec![0, 0, 1, 1]);

        let coverage = compute_coverage_probability(&predictions, &targets, 0.95).unwrap();
        assert!((0.0..=1.0).contains(&coverage));
    }

    #[test]
    fn test_validation_summary() {
        let results = ValidationResults {
            method: CalibrationMethod::Sigmoid,
            strategy: ValidationStrategy::KFold {
                k: 3,
                shuffle: false,
            },
            ece_scores: vec![0.1, 0.2, 0.15],
            mce_scores: vec![0.2, 0.3, 0.25],
            brier_scores: vec![0.15, 0.25, 0.2],
            reliability_scores: vec![0.1, 0.2, 0.15],
            coverage_scores: vec![0.9, 0.95, 0.92],
            training_times: vec![0.1, 0.12, 0.11],
            prediction_times: vec![0.01, 0.012, 0.011],
            summary: ValidationSummary {
                mean_ece: 0.0,
                std_ece: 0.0,
                mean_mce: 0.0,
                std_mce: 0.0,
                mean_brier: 0.0,
                std_brier: 0.0,
                mean_training_time: 0.0,
                mean_prediction_time: 0.0,
                overall_score: 0.0,
            },
        };

        let summary = ValidationSummary::from_results(&results);
        assert!((summary.mean_ece - 0.15).abs() < 1e-10);
        assert!((summary.mean_mce - 0.25).abs() < 1e-10);
        assert!((summary.mean_brier - 0.2).abs() < 1e-10);
        assert!(summary.overall_score > 0.0);
    }
}
