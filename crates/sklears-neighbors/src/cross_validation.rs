//! Advanced cross-validation techniques for neighbor-based algorithms
//!
//! This module provides specialized cross-validation methods optimized for
//! neighbor-based algorithms, including stratified validation, bootstrap methods,
//! spatial validation, and time series validation techniques.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use scirs2_core::random::Random;
use std::collections::HashMap;

use crate::distance::Distance;
use crate::knn::{Algorithm, KNeighborsClassifier, KNeighborsRegressor};
use crate::NeighborsError;
use sklears_core::traits::{Fit, Predict};
use sklears_core::types::{Float, Int};

/// Cross-validation strategy for neighbor algorithms
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CVStrategy {
    /// K-Fold cross-validation
    KFold { n_splits: usize },
    /// Stratified K-Fold (maintains class distribution)
    StratifiedKFold { n_splits: usize },
    /// Leave-One-Out cross-validation
    LeaveOneOut,
    /// Leave-P-Out cross-validation
    LeavePOut { p: usize },
    /// Bootstrap validation
    Bootstrap {
        n_samples: Option<usize>,
        n_iterations: usize,
    },
    /// Time series cross-validation with temporal ordering
    TimeSeriesKFold {
        n_splits: usize,
        max_train_size: Option<usize>,
    },
    /// Spatial cross-validation for geographic data
    SpatialKFold { n_splits: usize, buffer_size: Float },
}

/// Results from a single cross-validation fold
#[derive(Debug, Clone)]
pub struct CVFoldResult {
    /// Fold number
    pub fold: usize,
    /// Training set size
    pub train_size: usize,
    /// Test set size
    pub test_size: usize,
    /// Training time
    pub train_time: std::time::Duration,
    /// Prediction time
    pub predict_time: std::time::Duration,
    /// Test score (accuracy for classification, negative MSE for regression)
    pub test_score: Float,
    /// Training score (if computed)
    pub train_score: Option<Float>,
    /// Predictions on test set
    pub predictions: Array1<Float>,
    /// True labels/values for test set
    pub y_true: Array1<Float>,
}

/// Complete cross-validation results
#[derive(Debug, Clone)]
pub struct CVResults {
    /// Strategy used
    pub strategy: CVStrategy,
    /// Results for each fold
    pub fold_results: Vec<CVFoldResult>,
    /// Mean test score across folds
    pub mean_test_score: Float,
    /// Standard deviation of test scores
    pub std_test_score: Float,
    /// Mean training score across folds (if computed)
    pub mean_train_score: Option<Float>,
    /// Standard deviation of training scores
    pub std_train_score: Option<Float>,
    /// Total time for cross-validation
    pub total_time: std::time::Duration,
}

impl CVResults {
    /// Get the best fold result (highest test score)
    pub fn best_fold(&self) -> Option<&CVFoldResult> {
        self.fold_results
            .iter()
            .max_by(|a, b| a.test_score.partial_cmp(&b.test_score).unwrap())
    }

    /// Get the worst fold result (lowest test score)
    pub fn worst_fold(&self) -> Option<&CVFoldResult> {
        self.fold_results
            .iter()
            .min_by(|a, b| a.test_score.partial_cmp(&b.test_score).unwrap())
    }

    /// Calculate 95% confidence interval for test scores
    pub fn confidence_interval(&self) -> (Float, Float) {
        let n = self.fold_results.len() as Float;
        let se = self.std_test_score / n.sqrt();
        let margin = 1.96 * se; // 95% CI
        (self.mean_test_score - margin, self.mean_test_score + margin)
    }

    /// Generate a summary report
    pub fn summary(&self) -> String {
        let mut report = String::from("# Cross-Validation Results\n\n");

        report.push_str(&format!("**Strategy:** {:?}\n", self.strategy));
        report.push_str(&format!(
            "**Number of folds:** {}\n",
            self.fold_results.len()
        ));
        report.push_str(&format!(
            "**Mean test score:** {:.4} ± {:.4}\n",
            self.mean_test_score, self.std_test_score
        ));

        if let Some(mean_train) = self.mean_train_score {
            report.push_str(&format!(
                "**Mean train score:** {:.4} ± {:.4}\n",
                mean_train,
                self.std_train_score.unwrap_or(0.0)
            ));
        }

        let (ci_low, ci_high) = self.confidence_interval();
        report.push_str(&format!(
            "**95% Confidence Interval:** [{:.4}, {:.4}]\n",
            ci_low, ci_high
        ));

        report.push_str(&format!(
            "**Total time:** {:.2}s\n\n",
            self.total_time.as_secs_f64()
        ));

        // Detailed fold results
        report.push_str("## Fold Details\n\n");
        report.push_str("| Fold | Train Size | Test Size | Test Score | Train Time (ms) | Predict Time (ms) |\n");
        report.push_str("|------|------------|-----------|------------|-----------------|-------------------|\n");

        for result in &self.fold_results {
            report.push_str(&format!(
                "| {} | {} | {} | {:.4} | {:.2} | {:.2} |\n",
                result.fold,
                result.train_size,
                result.test_size,
                result.test_score,
                result.train_time.as_millis(),
                result.predict_time.as_millis()
            ));
        }

        report
    }
}

/// Advanced cross-validator for neighbor algorithms
pub struct NeighborCrossValidator {
    /// Cross-validation strategy
    strategy: CVStrategy,
    /// Whether to compute training scores
    compute_train_score: bool,
    /// Random state for reproducibility
    random_state: Option<u64>,
    /// Whether to use parallel processing
    parallel: bool,
}

impl NeighborCrossValidator {
    pub fn new(strategy: CVStrategy) -> Self {
        Self {
            strategy,
            compute_train_score: false,
            random_state: None,
            parallel: false,
        }
    }

    /// Enable computation of training scores
    pub fn with_train_scores(mut self, compute: bool) -> Self {
        self.compute_train_score = compute;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, state: u64) -> Self {
        self.random_state = Some(state);
        self
    }

    /// Enable parallel processing
    pub fn with_parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Run cross-validation for a KNN classifier
    #[allow(non_snake_case)]
    pub fn validate_classifier(
        &self,
        X: ArrayView2<Float>,
        y: ArrayView1<Int>,
        k: usize,
        distance: Distance,
        algorithm: Algorithm,
    ) -> Result<CVResults, NeighborsError> {
        let start_time = std::time::Instant::now();

        let splits = self.generate_splits(X, Some(y.view()))?;
        let mut fold_results = Vec::new();

        for (fold, (train_indices, test_indices)) in splits.into_iter().enumerate() {
            let _fold_start = std::time::Instant::now();

            // Create train/test splits
            let X_train = self.select_rows(&X, &train_indices);
            let y_train = self.select_elements(&y, &train_indices);
            let X_test = self.select_rows(&X, &test_indices);
            let y_test = self.select_elements(&y, &test_indices);

            // Train classifier
            let train_start = std::time::Instant::now();
            let classifier = KNeighborsClassifier::new(k)
                .with_metric(distance.clone())
                .with_algorithm(algorithm)
                .fit(&X_train, &y_train)?;
            let train_time = train_start.elapsed();

            // Make predictions
            let predict_start = std::time::Instant::now();
            let predictions = classifier.predict(&X_test)?;
            let predict_time = predict_start.elapsed();

            // Calculate test score (accuracy)
            let test_score = self.calculate_classification_accuracy(&predictions, &y_test);

            // Calculate train score if requested
            let train_score = if self.compute_train_score {
                let train_predictions = classifier.predict(&X_train)?;
                Some(self.calculate_classification_accuracy(&train_predictions, &y_train))
            } else {
                None
            };

            fold_results.push(CVFoldResult {
                fold,
                train_size: train_indices.len(),
                test_size: test_indices.len(),
                train_time,
                predict_time,
                test_score,
                train_score,
                predictions: predictions.mapv(|x| x as Float),
                y_true: y_test.mapv(|x| x as Float),
            });
        }

        let total_time = start_time.elapsed();
        self.compile_results(fold_results, total_time)
    }

    /// Run cross-validation for a KNN regressor
    #[allow(non_snake_case)]
    pub fn validate_regressor(
        &self,
        X: ArrayView2<Float>,
        y: ArrayView1<Float>,
        k: usize,
        distance: Distance,
        algorithm: Algorithm,
    ) -> Result<CVResults, NeighborsError> {
        let start_time = std::time::Instant::now();

        let splits = self.generate_splits(X, None)?;
        let mut fold_results = Vec::new();

        for (fold, (train_indices, test_indices)) in splits.into_iter().enumerate() {
            // Create train/test splits
            let X_train = self.select_rows(&X, &train_indices);
            let y_train = self.select_elements(&y, &train_indices);
            let X_test = self.select_rows(&X, &test_indices);
            let y_test = self.select_elements(&y, &test_indices);

            // Train regressor
            let train_start = std::time::Instant::now();
            let regressor = KNeighborsRegressor::new(k)
                .with_metric(distance.clone())
                .with_algorithm(algorithm)
                .fit(&X_train, &y_train)?;
            let train_time = train_start.elapsed();

            // Make predictions
            let predict_start = std::time::Instant::now();
            let predictions = regressor.predict(&X_test)?;
            let predict_time = predict_start.elapsed();

            // Calculate test score (negative MSE)
            let test_score = -self.calculate_mse(&predictions, &y_test);

            // Calculate train score if requested
            let train_score = if self.compute_train_score {
                let train_predictions = regressor.predict(&X_train)?;
                Some(-self.calculate_mse(&train_predictions, &y_train))
            } else {
                None
            };

            fold_results.push(CVFoldResult {
                fold,
                train_size: train_indices.len(),
                test_size: test_indices.len(),
                train_time,
                predict_time,
                test_score,
                train_score,
                predictions,
                y_true: y_test,
            });
        }

        let total_time = start_time.elapsed();
        self.compile_results(fold_results, total_time)
    }

    /// Compare multiple algorithms using cross-validation
    pub fn compare_algorithms(
        &self,
        X: ArrayView2<Float>,
        y: ArrayView1<Int>,
        k: usize,
        algorithms: &[Algorithm],
        distance: Distance,
    ) -> Result<HashMap<Algorithm, CVResults>, NeighborsError> {
        let mut results = HashMap::new();

        for &algorithm in algorithms {
            let cv_result = self.validate_classifier(X, y, k, distance.clone(), algorithm)?;
            results.insert(algorithm, cv_result);
        }

        Ok(results)
    }

    /// Compare multiple distance metrics using cross-validation
    pub fn compare_distances(
        &self,
        X: ArrayView2<Float>,
        y: ArrayView1<Int>,
        k: usize,
        distances: &[Distance],
        algorithm: Algorithm,
    ) -> Result<HashMap<Distance, CVResults>, NeighborsError> {
        let mut results = HashMap::new();

        for distance in distances {
            let cv_result = self.validate_classifier(X, y, k, distance.clone(), algorithm)?;
            results.insert(distance.clone(), cv_result);
        }

        Ok(results)
    }

    /// Find optimal k value using cross-validation
    pub fn optimize_k(
        &self,
        X: ArrayView2<Float>,
        y: ArrayView1<Int>,
        k_values: &[usize],
        distance: Distance,
        algorithm: Algorithm,
    ) -> Result<(usize, HashMap<usize, CVResults>), NeighborsError> {
        let mut results = HashMap::new();
        let mut best_k = k_values[0];
        let mut best_score = Float::NEG_INFINITY;

        for &k in k_values {
            let cv_result = self.validate_classifier(X, y, k, distance.clone(), algorithm)?;

            if cv_result.mean_test_score > best_score {
                best_score = cv_result.mean_test_score;
                best_k = k;
            }

            results.insert(k, cv_result);
        }

        Ok((best_k, results))
    }

    // Helper methods

    fn generate_splits(
        &self,
        X: ArrayView2<Float>,
        y: Option<ArrayView1<Int>>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        let n_samples = X.nrows();
        let mut rng = if let Some(seed) = self.random_state {
            Random::seed(seed)
        } else {
            Random::seed(42)
        };

        match self.strategy {
            CVStrategy::KFold { n_splits } => self.k_fold_splits(n_samples, n_splits, &mut rng),
            CVStrategy::StratifiedKFold { n_splits } => {
                if let Some(y) = y {
                    self.stratified_k_fold_splits(y, n_splits, &mut rng)
                } else {
                    Err(NeighborsError::InvalidInput(
                        "Stratified CV requires labels".to_string(),
                    ))
                }
            }
            CVStrategy::LeaveOneOut => self.leave_one_out_splits(n_samples),
            CVStrategy::LeavePOut { p } => self.leave_p_out_splits(n_samples, p),
            CVStrategy::Bootstrap {
                n_samples: bootstrap_n_samples,
                n_iterations,
            } => self.bootstrap_splits(n_samples, bootstrap_n_samples, n_iterations, &mut rng),
            CVStrategy::TimeSeriesKFold {
                n_splits,
                max_train_size,
            } => self.time_series_splits(n_samples, n_splits, max_train_size),
            CVStrategy::SpatialKFold {
                n_splits,
                buffer_size: _,
            } => {
                // Simplified spatial CV (would need coordinate data in practice)
                self.k_fold_splits(n_samples, n_splits, &mut rng)
            }
        }
    }

    fn k_fold_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        if n_splits > n_samples {
            return Err(NeighborsError::InvalidInput(format!(
                "n_splits ({}) > n_samples ({})",
                n_splits, n_samples
            )));
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle indices for randomization
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..i + 1);
            indices.swap(i, j);
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / n_splits;
        let remainder = n_samples % n_splits;

        let mut start = 0;
        for fold in 0..n_splits {
            let fold_size_adj = if fold < remainder {
                fold_size + 1
            } else {
                fold_size
            };
            let end = start + fold_size_adj;

            let test_indices = indices[start..end].to_vec();
            let train_indices: Vec<usize> = indices
                .iter()
                .enumerate()
                .filter(|(i, _)| *i < start || *i >= end)
                .map(|(_, &idx)| idx)
                .collect();

            splits.push((train_indices, test_indices));
            start = end;
        }

        Ok(splits)
    }

    fn stratified_k_fold_splits(
        &self,
        y: ArrayView1<Int>,
        n_splits: usize,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        // Group indices by class
        let mut class_indices: HashMap<Int, Vec<usize>> = HashMap::new();
        for (i, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(i);
        }

        // Shuffle indices within each class
        for (_, indices) in class_indices.iter_mut() {
            for i in (1..indices.len()).rev() {
                let j = rng.gen_range(0..i + 1);
                indices.swap(i, j);
            }
        }

        let mut splits = vec![(Vec::new(), Vec::new()); n_splits];

        // Distribute samples from each class across folds
        for (_, indices) in class_indices {
            let class_size = indices.len();
            let fold_size = class_size / n_splits;
            let remainder = class_size % n_splits;

            let mut start = 0;
            for fold in 0..n_splits {
                let fold_size_adj = if fold < remainder {
                    fold_size + 1
                } else {
                    fold_size
                };
                let end = start + fold_size_adj;

                for &idx in &indices[start..end] {
                    splits[fold].1.push(idx); // Add to test set
                }

                start = end;
            }
        }

        // Create train sets (all indices except test set)
        let all_indices: Vec<usize> = (0..y.len()).collect();
        for (train_indices, test_indices) in splits.iter_mut() {
            *train_indices = all_indices
                .iter()
                .filter(|&&idx| !test_indices.contains(&idx))
                .copied()
                .collect();
        }

        Ok(splits)
    }

    fn leave_one_out_splits(
        &self,
        n_samples: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        let mut splits = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    fn leave_p_out_splits(
        &self,
        n_samples: usize,
        p: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        if p > n_samples {
            return Err(NeighborsError::InvalidInput(format!(
                "p ({}) > n_samples ({})",
                p, n_samples
            )));
        }

        let mut splits = Vec::new();

        // Generate all combinations of p indices (simplified implementation)
        // For large datasets, this would be computationally expensive
        if p == 1 {
            return self.leave_one_out_splits(n_samples);
        }

        // For simplicity, just generate a subset of combinations
        let max_combinations = 100.min(Self::n_choose_k(n_samples, p));
        let mut rng = Random::seed(42);

        for _ in 0..max_combinations {
            let mut test_indices = Vec::new();
            let mut available: Vec<usize> = (0..n_samples).collect();

            for _ in 0..p {
                if available.is_empty() {
                    break;
                }
                let idx = rng.gen_range(0..available.len());
                test_indices.push(available.remove(idx));
            }

            let train_indices = available;
            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    fn bootstrap_splits(
        &self,
        n_samples: usize,
        bootstrap_n_samples: Option<usize>,
        n_iterations: usize,
        rng: &mut Random<scirs2_core::random::rngs::StdRng>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        let bootstrap_size = bootstrap_n_samples.unwrap_or(n_samples);
        let mut splits = Vec::with_capacity(n_iterations);

        for _ in 0..n_iterations {
            let mut train_indices = Vec::with_capacity(bootstrap_size);
            let mut in_bootstrap = vec![false; n_samples];

            // Generate bootstrap sample
            for _ in 0..bootstrap_size {
                let idx = rng.gen_range(0..n_samples);
                train_indices.push(idx);
                in_bootstrap[idx] = true;
            }

            // Out-of-bag samples become test set
            let test_indices: Vec<usize> = (0..n_samples).filter(|&i| !in_bootstrap[i]).collect();

            if !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    fn time_series_splits(
        &self,
        n_samples: usize,
        n_splits: usize,
        max_train_size: Option<usize>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>, NeighborsError> {
        let mut splits = Vec::new();
        let test_size = n_samples / n_splits;

        for i in 0..n_splits {
            let test_start = (i + 1) * test_size;
            let test_end = ((i + 2) * test_size).min(n_samples);

            if test_start >= n_samples {
                break;
            }

            let test_indices: Vec<usize> = (test_start..test_end).collect();

            let train_end = test_start;
            let train_start = if let Some(max_size) = max_train_size {
                train_end.saturating_sub(max_size)
            } else {
                0
            };

            let train_indices: Vec<usize> = (train_start..train_end).collect();

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        Ok(splits)
    }

    fn select_rows<T: Copy>(&self, arr: &ArrayView2<T>, indices: &[usize]) -> Array2<T> {
        let n_features = arr.ncols();
        let mut result = Array2::uninit((indices.len(), n_features));

        for (i, &idx) in indices.iter().enumerate() {
            for j in 0..n_features {
                result[(i, j)] = std::mem::MaybeUninit::new(arr[(idx, j)]);
            }
        }

        // SAFETY: We've initialized all elements
        unsafe { result.assume_init() }
    }

    fn select_elements<T: Copy>(&self, arr: &ArrayView1<T>, indices: &[usize]) -> Array1<T> {
        Array1::from_iter(indices.iter().map(|&i| arr[i]))
    }

    fn calculate_classification_accuracy(
        &self,
        predictions: &Array1<Int>,
        y_true: &Array1<Int>,
    ) -> Float {
        if predictions.len() != y_true.len() {
            return 0.0;
        }

        let correct = predictions
            .iter()
            .zip(y_true.iter())
            .filter(|(&pred, &true_val)| pred == true_val)
            .count();

        correct as Float / predictions.len() as Float
    }

    fn calculate_mse(&self, predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
        if predictions.len() != y_true.len() {
            return Float::INFINITY;
        }

        let mse = predictions
            .iter()
            .zip(y_true.iter())
            .map(|(&pred, &true_val)| (pred - true_val).powi(2))
            .sum::<Float>()
            / predictions.len() as Float;

        mse
    }

    fn compile_results(
        &self,
        fold_results: Vec<CVFoldResult>,
        total_time: std::time::Duration,
    ) -> Result<CVResults, NeighborsError> {
        if fold_results.is_empty() {
            return Err(NeighborsError::InvalidInput("No fold results".to_string()));
        }

        let test_scores: Vec<Float> = fold_results.iter().map(|r| r.test_score).collect();
        let mean_test_score = test_scores.iter().sum::<Float>() / test_scores.len() as Float;
        let std_test_score = if test_scores.len() > 1 {
            let variance = test_scores
                .iter()
                .map(|&score| (score - mean_test_score).powi(2))
                .sum::<Float>()
                / (test_scores.len() - 1) as Float;
            variance.sqrt()
        } else {
            0.0
        };

        let (mean_train_score, std_train_score) = if fold_results[0].train_score.is_some() {
            let train_scores: Vec<Float> =
                fold_results.iter().filter_map(|r| r.train_score).collect();

            if !train_scores.is_empty() {
                let mean = train_scores.iter().sum::<Float>() / train_scores.len() as Float;
                let std = if train_scores.len() > 1 {
                    let variance = train_scores
                        .iter()
                        .map(|&score| (score - mean).powi(2))
                        .sum::<Float>()
                        / (train_scores.len() - 1) as Float;
                    variance.sqrt()
                } else {
                    0.0
                };
                (Some(mean), Some(std))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };

        Ok(CVResults {
            strategy: self.strategy,
            fold_results,
            mean_test_score,
            std_test_score,
            mean_train_score,
            std_train_score,
            total_time,
        })
    }

    fn n_choose_k(n: usize, k: usize) -> usize {
        if k > n {
            return 0;
        }
        if k == 0 || k == n {
            return 1;
        }

        let mut result = 1;
        for i in 0..k {
            result = result * (n - i) / (i + 1);
        }
        result
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[allow(non_snake_case)]
    fn create_test_data() -> (Array2<Float>, Array1<Int>) {
        let X = Array2::from_shape_vec((20, 2), (0..40).map(|x| x as Float).collect()).unwrap();
        let y = Array1::from_iter((0..20).map(|i| (i % 3) as Int));
        (X, y)
    }

    #[test]
    fn test_k_fold_cv() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 5 });

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.fold_results.len(), 5);
        assert!(result.mean_test_score >= 0.0 && result.mean_test_score <= 1.0);
    }

    #[test]
    fn test_stratified_k_fold_cv() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::StratifiedKFold { n_splits: 3 });

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.fold_results.len(), 3);
    }

    #[test]
    fn test_leave_one_out_cv() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::LeaveOneOut);

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.fold_results.len(), 20); // One fold per sample
    }

    #[test]
    fn test_bootstrap_cv() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::Bootstrap {
            n_samples: None,
            n_iterations: 5,
        });

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.fold_results.len() <= 5); // Some bootstraps might not have test samples
    }

    #[test]
    fn test_time_series_cv() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::TimeSeriesKFold {
            n_splits: 4,
            max_train_size: None,
        });

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.fold_results.is_empty());
    }

    #[test]
    fn test_cv_with_train_scores() {
        let (X, y) = create_test_data();
        let cv =
            NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 3 }).with_train_scores(true);

        let result =
            cv.validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.mean_train_score.is_some());
        assert!(result.fold_results.iter().all(|r| r.train_score.is_some()));
    }

    #[test]
    fn test_algorithm_comparison() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 3 });

        let algorithms = vec![Algorithm::Brute, Algorithm::KdTree];
        let results =
            cv.compare_algorithms(X.view(), y.view(), 3, &algorithms, Distance::Euclidean);

        assert!(results.is_ok());
        let results = results.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.contains_key(&Algorithm::Brute));
        assert!(results.contains_key(&Algorithm::KdTree));
    }

    #[test]
    fn test_k_optimization() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 3 });

        let k_values = vec![1, 3, 5];
        let result = cv.optimize_k(
            X.view(),
            y.view(),
            &k_values,
            Distance::Euclidean,
            Algorithm::Brute,
        );

        assert!(result.is_ok());
        let (best_k, results) = result.unwrap();
        assert!(k_values.contains(&best_k));
        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_regressor_cv() {
        let (X, _) = create_test_data();
        let y_reg = Array1::from_iter((0..20).map(|i| i as Float * 0.5));

        let cv = NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 3 });

        let result = cv.validate_regressor(
            X.view(),
            y_reg.view(),
            3,
            Distance::Euclidean,
            Algorithm::Brute,
        );

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.fold_results.len(), 3);
        // For regression, we use negative MSE, so scores should be <= 0
        assert!(result.mean_test_score <= 0.0);
    }

    #[test]
    fn test_cv_results_methods() {
        let (X, y) = create_test_data();
        let cv = NeighborCrossValidator::new(CVStrategy::KFold { n_splits: 5 });

        let result = cv
            .validate_classifier(X.view(), y.view(), 3, Distance::Euclidean, Algorithm::Brute)
            .unwrap();

        // Test best/worst fold methods
        let best = result.best_fold();
        let worst = result.worst_fold();
        assert!(best.is_some());
        assert!(worst.is_some());
        assert!(best.unwrap().test_score >= worst.unwrap().test_score);

        // Test confidence interval
        let (ci_low, ci_high) = result.confidence_interval();
        assert!(ci_low <= result.mean_test_score);
        assert!(ci_high >= result.mean_test_score);

        // Test summary generation
        let summary = result.summary();
        assert!(summary.contains("Cross-Validation Results"));
        assert!(summary.contains("KFold"));
    }
}
