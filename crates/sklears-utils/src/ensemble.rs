//! Ensemble utilities for machine learning
//!
//! This module provides utilities for ensemble methods including bootstrap sampling,
//! bagging, and ensemble combination strategies.

use crate::{UtilsError, UtilsResult};
use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;

/// Bootstrap sample generator
///
/// Creates bootstrap samples (sampling with replacement) for ensemble methods.
#[derive(Clone, Debug)]
pub struct Bootstrap {
    n_samples: Option<usize>,
    random_state: Option<u64>,
}

impl Bootstrap {
    /// Create a new bootstrap sampler
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples to draw (None = same as input size)
    /// * `random_state` - Random seed for reproducibility
    pub fn new(n_samples: Option<usize>, random_state: Option<u64>) -> Self {
        Self {
            n_samples,
            random_state,
        }
    }

    /// Generate bootstrap sample indices
    ///
    /// # Arguments
    /// * `n_population` - Size of the population to sample from
    ///
    /// # Returns
    /// Tuple of (in-bag indices, out-of-bag indices)
    pub fn sample(&self, n_population: usize) -> UtilsResult<(Vec<usize>, Vec<usize>)> {
        if n_population == 0 {
            return Err(UtilsError::InvalidParameter(
                "Population size must be positive".to_string(),
            ));
        }

        let n_samples = self.n_samples.unwrap_or(n_population);
        let mut rng = self
            .random_state
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(|| StdRng::seed_from_u64(42));

        // Generate in-bag samples
        let mut in_bag = Vec::with_capacity(n_samples);
        let mut in_bag_set = vec![false; n_population];

        for _ in 0..n_samples {
            let idx = rng.gen_range(0..n_population);
            in_bag.push(idx);
            in_bag_set[idx] = true;
        }

        // Collect out-of-bag samples
        let out_of_bag: Vec<usize> = (0..n_population).filter(|&i| !in_bag_set[i]).collect();

        Ok((in_bag, out_of_bag))
    }

    /// Generate multiple bootstrap samples
    ///
    /// # Arguments
    /// * `n_population` - Size of the population
    /// * `n_bootstraps` - Number of bootstrap samples to generate
    ///
    /// # Returns
    /// Vector of (in-bag, out-of-bag) index pairs
    pub fn sample_multiple(
        &self,
        n_population: usize,
        n_bootstraps: usize,
    ) -> UtilsResult<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut samples = Vec::with_capacity(n_bootstraps);

        for i in 0..n_bootstraps {
            // Create new sampler with different seed for each bootstrap
            let seed = self.random_state.map(|s| s + i as u64);
            let sampler = Bootstrap::new(self.n_samples, seed);
            samples.push(sampler.sample(n_population)?);
        }

        Ok(samples)
    }
}

impl Default for Bootstrap {
    fn default() -> Self {
        Self::new(None, Some(42))
    }
}

/// Bagging utility for creating bagged ensemble predictions
#[derive(Clone, Debug)]
pub struct BaggingPredictor {
    aggregation: AggregationStrategy,
}

/// Strategy for aggregating predictions from ensemble members
#[derive(Clone, Debug, PartialEq)]
pub enum AggregationStrategy {
    /// Average predictions (for regression)
    Mean,
    /// Median of predictions (robust to outliers)
    Median,
    /// Majority voting (for classification)
    MajorityVote,
    /// Weighted average with given weights
    WeightedMean,
}

impl BaggingPredictor {
    /// Create a new bagging predictor
    pub fn new(aggregation: AggregationStrategy) -> Self {
        Self { aggregation }
    }

    /// Aggregate regression predictions
    ///
    /// # Arguments
    /// * `predictions` - Matrix of predictions (n_samples × n_estimators)
    /// * `weights` - Optional weights for weighted averaging
    ///
    /// # Returns
    /// Aggregated predictions for each sample
    pub fn aggregate_regression(
        &self,
        predictions: &Array2<f64>,
        weights: Option<&Array1<f64>>,
    ) -> UtilsResult<Array1<f64>> {
        if predictions.nrows() == 0 || predictions.ncols() == 0 {
            return Err(UtilsError::InvalidParameter(
                "Predictions array cannot be empty".to_string(),
            ));
        }

        match &self.aggregation {
            AggregationStrategy::Mean => Ok(predictions
                .mean_axis(scirs2_core::ndarray::Axis(1))
                .unwrap()),
            AggregationStrategy::Median => {
                let mut result = Array1::zeros(predictions.nrows());
                for (i, row) in predictions.outer_iter().enumerate() {
                    let mut sorted: Vec<f64> = row.to_vec();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    result[i] = if sorted.len() % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    };
                }
                Ok(result)
            }
            AggregationStrategy::WeightedMean => {
                let weights = weights.ok_or_else(|| {
                    UtilsError::InvalidParameter("Weights required for weighted mean".to_string())
                })?;

                if weights.len() != predictions.ncols() {
                    return Err(UtilsError::InvalidParameter(
                        "Number of weights must match number of estimators".to_string(),
                    ));
                }

                let weight_sum: f64 = weights.sum();
                if weight_sum <= 0.0 {
                    return Err(UtilsError::InvalidParameter(
                        "Weight sum must be positive".to_string(),
                    ));
                }

                let normalized_weights = weights / weight_sum;
                Ok(predictions.dot(&normalized_weights))
            }
            AggregationStrategy::MajorityVote => Err(UtilsError::InvalidParameter(
                "Use aggregate_classification for majority voting".to_string(),
            )),
        }
    }

    /// Aggregate classification predictions (voting)
    ///
    /// # Arguments
    /// * `predictions` - Matrix of class predictions (n_samples × n_estimators)
    ///
    /// # Returns
    /// Predicted class for each sample (majority vote)
    pub fn aggregate_classification(
        &self,
        predictions: &Array2<usize>,
    ) -> UtilsResult<Array1<usize>> {
        if predictions.nrows() == 0 || predictions.ncols() == 0 {
            return Err(UtilsError::InvalidParameter(
                "Predictions array cannot be empty".to_string(),
            ));
        }

        let mut result = Array1::zeros(predictions.nrows());

        for (i, row) in predictions.outer_iter().enumerate() {
            // Count votes for each class
            let mut vote_counts: HashMap<usize, usize> = HashMap::new();
            for &pred in row.iter() {
                *vote_counts.entry(pred).or_insert(0) += 1;
            }

            // Find class with maximum votes
            let (predicted_class, _) = vote_counts
                .iter()
                .max_by_key(|(_, &count)| count)
                .ok_or_else(|| UtilsError::InvalidParameter("No votes found".to_string()))?;

            result[i] = *predicted_class;
        }

        Ok(result)
    }

    /// Aggregate classification probabilities (soft voting)
    ///
    /// # Arguments
    /// * `probabilities` - Array of probability matrices from each estimator
    ///   `Vec<Array2>` where each Array2 is (n_samples × n_classes)
    ///
    /// # Returns
    /// Averaged probability matrix (n_samples × n_classes)
    pub fn aggregate_probabilities(
        &self,
        probabilities: &[Array2<f64>],
    ) -> UtilsResult<Array2<f64>> {
        if probabilities.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Probabilities array cannot be empty".to_string(),
            ));
        }

        let (n_samples, n_classes) = probabilities[0].dim();

        // Validate all arrays have same shape
        for probs in probabilities.iter() {
            if probs.dim() != (n_samples, n_classes) {
                return Err(UtilsError::InvalidParameter(
                    "All probability matrices must have the same shape".to_string(),
                ));
            }
        }

        // Average probabilities
        let mut result = Array2::zeros((n_samples, n_classes));
        for probs in probabilities {
            result += probs;
        }
        result /= probabilities.len() as f64;

        Ok(result)
    }
}

impl Default for BaggingPredictor {
    fn default() -> Self {
        Self::new(AggregationStrategy::Mean)
    }
}

/// Out-of-bag score estimator
///
/// Estimates model performance using out-of-bag samples from bootstrap sampling.
#[derive(Clone, Debug)]
pub struct OOBScoreEstimator;

impl OOBScoreEstimator {
    /// Compute OOB score for regression
    ///
    /// # Arguments
    /// * `y_true` - True target values
    /// * `oob_predictions` - OOB predictions for each sample
    ///
    /// # Returns
    /// R² score on OOB samples
    pub fn oob_score_regression(
        y_true: &Array1<f64>,
        oob_predictions: &Array1<f64>,
    ) -> UtilsResult<f64> {
        if y_true.len() != oob_predictions.len() {
            return Err(UtilsError::InvalidParameter(
                "y_true and predictions must have same length".to_string(),
            ));
        }

        if y_true.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Cannot compute score on empty array".to_string(),
            ));
        }

        // Compute R² score
        let y_mean = y_true.mean().unwrap();
        let ss_tot: f64 = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum();
        let ss_res: f64 = y_true
            .iter()
            .zip(oob_predictions.iter())
            .map(|(&y, &pred)| (y - pred).powi(2))
            .sum();

        if ss_tot <= 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }

    /// Compute OOB accuracy for classification
    ///
    /// # Arguments
    /// * `y_true` - True class labels
    /// * `oob_predictions` - OOB predicted class labels
    ///
    /// # Returns
    /// Accuracy score on OOB samples
    pub fn oob_accuracy(
        y_true: &Array1<usize>,
        oob_predictions: &Array1<usize>,
    ) -> UtilsResult<f64> {
        if y_true.len() != oob_predictions.len() {
            return Err(UtilsError::InvalidParameter(
                "y_true and predictions must have same length".to_string(),
            ));
        }

        if y_true.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Cannot compute score on empty array".to_string(),
            ));
        }

        let correct: usize = y_true
            .iter()
            .zip(oob_predictions.iter())
            .filter(|(&y, &pred)| y == pred)
            .count();

        Ok(correct as f64 / y_true.len() as f64)
    }
}

/// Stacking ensemble utilities
#[derive(Clone, Debug)]
pub struct StackingHelper;

impl StackingHelper {
    /// Generate cross-validated predictions for stacking
    ///
    /// # Arguments
    /// * `n_samples` - Number of samples
    /// * `n_folds` - Number of cross-validation folds
    /// * `random_state` - Random seed
    ///
    /// # Returns
    /// Vector of (train_indices, test_indices) for each fold
    pub fn generate_cv_folds(
        n_samples: usize,
        n_folds: usize,
        random_state: Option<u64>,
    ) -> UtilsResult<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_folds < 2 {
            return Err(UtilsError::InvalidParameter(
                "n_folds must be at least 2".to_string(),
            ));
        }

        if n_samples < n_folds {
            return Err(UtilsError::InvalidParameter(
                "n_samples must be >= n_folds".to_string(),
            ));
        }

        // Create shuffled indices
        let mut indices: Vec<usize> = (0..n_samples).collect();
        let mut rng = random_state
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(|| StdRng::seed_from_u64(42));

        // Fisher-Yates shuffle
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        // Distribute indices into folds
        let fold_sizes = Self::compute_fold_sizes(n_samples, n_folds);
        let mut folds = Vec::with_capacity(n_folds);
        let mut start = 0;

        for size in fold_sizes {
            let test_indices = indices[start..start + size].to_vec();
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= start + size)
                .map(|(_, &idx)| idx)
                .collect();

            folds.push((train_indices, test_indices));
            start += size;
        }

        Ok(folds)
    }

    fn compute_fold_sizes(n_samples: usize, n_folds: usize) -> Vec<usize> {
        let base_size = n_samples / n_folds;
        let remainder = n_samples % n_folds;

        (0..n_folds)
            .map(|i| {
                if i < remainder {
                    base_size + 1
                } else {
                    base_size
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_bootstrap_sample() {
        let bootstrap = Bootstrap::new(Some(10), Some(42));
        let (in_bag, out_of_bag) = bootstrap.sample(10).unwrap();

        assert_eq!(in_bag.len(), 10);
        assert!(!out_of_bag.is_empty()); // Typically ~37% are OOB
        assert!(out_of_bag.len() < 10);

        // Check all indices are valid
        for &idx in &in_bag {
            assert!(idx < 10);
        }
        for &idx in &out_of_bag {
            assert!(idx < 10);
        }
    }

    #[test]
    fn test_bootstrap_multiple() {
        let bootstrap = Bootstrap::new(None, Some(42));
        let samples = bootstrap.sample_multiple(10, 5).unwrap();

        assert_eq!(samples.len(), 5);

        // Check each sample is valid
        for (in_bag, out_of_bag) in &samples {
            assert_eq!(in_bag.len(), 10);
            assert!(out_of_bag.len() <= 10);
        }
    }

    #[test]
    fn test_bagging_mean_aggregation() {
        let predictor = BaggingPredictor::new(AggregationStrategy::Mean);

        // 3 samples, 4 estimators
        let predictions = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 2.0, 2.0, 2.0],
            [1.0, 3.0, 2.0, 4.0]
        ];

        let result = predictor.aggregate_regression(&predictions, None).unwrap();

        assert_abs_diff_eq!(result[0], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], 2.5, epsilon = 1e-10);
    }

    #[test]
    fn test_bagging_median_aggregation() {
        let predictor = BaggingPredictor::new(AggregationStrategy::Median);

        // Test with 4 estimators per sample
        let predictions = array![
            [1.0, 2.0, 100.0, 3.0], // Median = 2.5 (robust to outlier 100)
            [1.0, 2.0, 3.0, 4.0]    // Median = 2.5
        ];

        let result = predictor.aggregate_regression(&predictions, None).unwrap();

        assert_abs_diff_eq!(result[0], 2.5, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 2.5, epsilon = 1e-10);

        // Test with 5 estimators (odd number)
        let predictions2 = array![
            [1.0, 2.0, 3.0, 4.0, 5.0],    // Median = 3.0
            [10.0, 1.0, 2.0, 3.0, 100.0]  // Median = 3.0 (robust to outliers)
        ];

        let result2 = predictor.aggregate_regression(&predictions2, None).unwrap();
        assert_abs_diff_eq!(result2[0], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result2[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bagging_weighted_mean() {
        let predictor = BaggingPredictor::new(AggregationStrategy::WeightedMean);

        let predictions = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]];
        let weights = array![0.5, 0.3, 0.2]; // Sum = 1.0

        let result = predictor
            .aggregate_regression(&predictions, Some(&weights))
            .unwrap();

        // Sample 0: 1.0*0.5 + 2.0*0.3 + 3.0*0.2 = 0.5 + 0.6 + 0.6 = 1.7
        assert_abs_diff_eq!(result[0], 1.7, epsilon = 1e-10);
        // Sample 1: 4.0*0.5 + 5.0*0.3 + 6.0*0.2 = 2.0 + 1.5 + 1.2 = 4.7
        assert_abs_diff_eq!(result[1], 4.7, epsilon = 1e-10);
    }

    #[test]
    fn test_majority_vote() {
        let predictor = BaggingPredictor::new(AggregationStrategy::MajorityVote);

        let predictions = array![
            [0, 0, 1, 0, 0], // Majority: 0 (4 votes)
            [1, 1, 0, 1, 1], // Majority: 1 (4 votes)
            [2, 2, 2, 0, 1]  // Majority: 2 (3 votes)
        ];

        let result = predictor.aggregate_classification(&predictions).unwrap();

        assert_eq!(result[0], 0);
        assert_eq!(result[1], 1);
        assert_eq!(result[2], 2);
    }

    #[test]
    fn test_aggregate_probabilities() {
        let predictor = BaggingPredictor::default();

        let probs1 = array![[0.8, 0.2], [0.3, 0.7]];
        let probs2 = array![[0.6, 0.4], [0.4, 0.6]];

        let result = predictor
            .aggregate_probabilities(&[probs1, probs2])
            .unwrap();

        assert_abs_diff_eq!(result[[0, 0]], 0.7, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[0, 1]], 0.3, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 0]], 0.35, epsilon = 1e-10);
        assert_abs_diff_eq!(result[[1, 1]], 0.65, epsilon = 1e-10);
    }

    #[test]
    fn test_oob_score_regression() {
        let y_true = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y_pred = array![1.1, 1.9, 3.1, 3.9, 5.1];

        let score = OOBScoreEstimator::oob_score_regression(&y_true, &y_pred).unwrap();

        // Should be close to 1.0 (perfect predictions)
        assert!(score > 0.95);
    }

    #[test]
    fn test_oob_accuracy() {
        let y_true = array![0, 1, 2, 0, 1];
        let y_pred = array![0, 1, 2, 0, 2]; // 4/5 correct

        let accuracy = OOBScoreEstimator::oob_accuracy(&y_true, &y_pred).unwrap();

        assert_abs_diff_eq!(accuracy, 0.8, epsilon = 1e-10);
    }

    #[test]
    fn test_stacking_cv_folds() {
        let folds = StackingHelper::generate_cv_folds(10, 3, Some(42)).unwrap();

        assert_eq!(folds.len(), 3);

        // Check all samples are covered exactly once in test sets
        let mut all_test_indices: Vec<usize> = Vec::new();
        for (train, test) in &folds {
            assert!(!train.is_empty());
            assert!(!test.is_empty());
            assert_eq!(train.len() + test.len(), 10);
            all_test_indices.extend(test);
        }

        all_test_indices.sort_unstable();
        assert_eq!(all_test_indices, vec![0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
    }
}
