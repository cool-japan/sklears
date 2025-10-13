//! Parallel feature evaluation utilities
//!
//! This module provides parallel implementations for feature evaluation
//! to improve performance of feature selection algorithms.

use rayon::prelude::*;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{error::Result as SklResult, types::Float};

/// Parallel feature evaluation trait
///
/// This trait provides methods for evaluating features in parallel
/// to improve performance on multi-core systems.
pub trait ParallelFeatureEvaluator {
    /// Evaluate all features in parallel
    fn evaluate_features_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> SklResult<Array1<Float>>;

    /// Evaluate specific features in parallel
    fn evaluate_features_subset_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        feature_indices: &[usize],
    ) -> SklResult<Array1<Float>>;
}

/// Parallel feature scorer for univariate statistical tests
#[derive(Debug, Clone)]
pub struct ParallelUnivariateScorer<F>
where
    F: Fn(&Array1<Float>, &Array1<i32>) -> SklResult<Float> + Sync + Send,
{
    score_func: F,
}

impl<F> ParallelUnivariateScorer<F>
where
    F: Fn(&Array1<Float>, &Array1<i32>) -> SklResult<Float> + Sync + Send,
{
    /// Create a new parallel univariate scorer
    pub fn new(score_func: F) -> Self {
        Self { score_func }
    }

    /// Evaluate features in parallel using the provided scoring function
    pub fn evaluate_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> SklResult<Array1<Float>> {
        let n_features = x.ncols();

        // Use parallel iterator to evaluate features
        let scores: SklResult<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = x.column(feature_idx);
                let feature_owned = feature_column.to_owned();
                (self.score_func)(&feature_owned, y)
            })
            .collect();

        scores.map(Array1::from_vec)
    }

    /// Evaluate specific features in parallel
    pub fn evaluate_subset_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<i32>,
        feature_indices: &[usize],
    ) -> SklResult<Array1<Float>> {
        let scores: SklResult<Vec<Float>> = feature_indices
            .par_iter()
            .map(|&feature_idx| {
                let feature_column = x.column(feature_idx);
                let feature_owned = feature_column.to_owned();
                (self.score_func)(&feature_owned, y)
            })
            .collect();

        scores.map(Array1::from_vec)
    }
}

/// Parallel feature scorer for regression problems
#[derive(Debug, Clone)]
pub struct ParallelUnivariateRegressionScorer<F>
where
    F: Fn(&Array1<Float>, &Array1<Float>) -> SklResult<Float> + Sync + Send,
{
    score_func: F,
}

impl<F> ParallelUnivariateRegressionScorer<F>
where
    F: Fn(&Array1<Float>, &Array1<Float>) -> SklResult<Float> + Sync + Send,
{
    /// Create a new parallel univariate regression scorer
    pub fn new(score_func: F) -> Self {
        Self { score_func }
    }

    /// Evaluate features in parallel for regression
    pub fn evaluate_parallel(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = x.ncols();

        let scores: SklResult<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = x.column(feature_idx);
                let feature_owned = feature_column.to_owned();
                (self.score_func)(&feature_owned, y)
            })
            .collect();

        scores.map(Array1::from_vec)
    }
}

/// Parallel feature ranking utilities
pub struct ParallelFeatureRanker;

impl ParallelFeatureRanker {
    /// Rank features in parallel based on scores
    pub fn rank_features_parallel(scores: &Array1<Float>) -> Vec<usize> {
        let mut indexed_scores: Vec<(usize, Float)> = scores
            .iter()
            .enumerate()
            .map(|(idx, &score)| (idx, score))
            .collect();

        // Sort by score descending (parallel sort for large arrays)
        if indexed_scores.len() > 1000 {
            indexed_scores.par_sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        } else {
            indexed_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        }

        indexed_scores.into_iter().map(|(idx, _)| idx).collect()
    }

    /// Select top k features in parallel
    pub fn select_top_k_parallel(scores: &Array1<Float>, k: usize) -> Vec<usize> {
        let ranked_features = Self::rank_features_parallel(scores);
        ranked_features.into_iter().take(k).collect()
    }
}

/// Parallel correlation computation
pub struct ParallelCorrelationComputer;

impl ParallelCorrelationComputer {
    /// Compute correlation between features and target in parallel
    pub fn compute_feature_target_correlation_parallel(
        x: &Array2<Float>,
        y: &Array1<i32>,
    ) -> SklResult<Array1<Float>> {
        let n_features = x.ncols();
        let y_float: Array1<Float> = y.mapv(|v| v as Float);

        let correlations: SklResult<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = x.column(feature_idx);
                let feature_owned = feature_column.to_owned();
                Self::compute_correlation(&feature_owned, &y_float)
            })
            .collect();

        correlations.map(Array1::from_vec)
    }

    /// Compute correlation between features and continuous target in parallel
    pub fn compute_feature_target_correlation_regression_parallel(
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = x.ncols();

        let correlations: SklResult<Vec<Float>> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = x.column(feature_idx);
                let feature_owned = feature_column.to_owned();
                Self::compute_correlation(&feature_owned, y)
            })
            .collect();

        correlations.map(Array1::from_vec)
    }

    /// Compute correlation matrix between features in parallel
    pub fn compute_feature_correlation_matrix_parallel(
        x: &Array2<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_features = x.ncols();
        let mut correlation_matrix = Array2::zeros((n_features, n_features));

        // Compute upper triangular part in parallel
        let upper_triangular: SklResult<Vec<((usize, usize), Float)>> = (0..n_features)
            .into_par_iter()
            .flat_map(|i| {
                (i..n_features).into_par_iter().map(move |j| {
                    let corr = if i == j {
                        Ok(1.0)
                    } else {
                        let feature_i = x.column(i).to_owned();
                        let feature_j = x.column(j).to_owned();
                        Self::compute_correlation(&feature_i, &feature_j)
                    };
                    corr.map(|c| ((i, j), c))
                })
            })
            .collect();

        // Fill the correlation matrix
        for ((i, j), corr) in upper_triangular? {
            correlation_matrix[[i, j]] = corr;
            if i != j {
                correlation_matrix[[j, i]] = corr;
            }
        }

        Ok(correlation_matrix)
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(x: &Array1<Float>, y: &Array1<Float>) -> SklResult<Float> {
        let n = x.len();
        if n != y.len() {
            return Err(sklears_core::error::SklearsError::InvalidInput(
                "Arrays must have the same length".to_string(),
            ));
        }

        if n < 2 {
            return Ok(0.0);
        }

        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();

        let mut sum_xy = 0.0;
        let mut sum_xx = 0.0;
        let mut sum_yy = 0.0;

        for i in 0..n {
            let dx = x[i] - mean_x;
            let dy = y[i] - mean_y;
            sum_xy += dx * dy;
            sum_xx += dx * dx;
            sum_yy += dy * dy;
        }

        let denominator = (sum_xx * sum_yy).sqrt();
        if denominator < 1e-10 {
            Ok(0.0)
        } else {
            Ok(sum_xy / denominator)
        }
    }
}

/// Parallel variance computation
pub struct ParallelVarianceComputer;

impl ParallelVarianceComputer {
    /// Compute variance of features in parallel
    pub fn compute_feature_variances_parallel(x: &Array2<Float>) -> Array1<Float> {
        let n_features = x.ncols();

        let variances: Vec<Float> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let feature_column = x.column(feature_idx);
                Self::compute_variance(&feature_column.to_owned())
            })
            .collect();

        Array1::from_vec(variances)
    }

    /// Compute variance of a single feature
    fn compute_variance(x: &Array1<Float>) -> Float {
        let n = x.len();
        if n < 2 {
            return 0.0;
        }

        let mean = x.mean().unwrap();
        let sum_sq_diff: Float = x.iter().map(|&val| (val - mean).powi(2)).sum();
        sum_sq_diff / (n - 1) as Float
    }
}

/// Parallel feature selection utilities
pub struct ParallelSelectionUtils;

impl ParallelSelectionUtils {
    /// Apply threshold-based selection in parallel
    pub fn apply_threshold_parallel(scores: &Array1<Float>, threshold: Float) -> Vec<usize> {
        (0..scores.len())
            .into_par_iter()
            .filter(|&i| scores[i] >= threshold)
            .collect()
    }

    /// Apply percentile-based selection in parallel
    pub fn apply_percentile_parallel(scores: &Array1<Float>, percentile: Float) -> Vec<usize> {
        let mut sorted_scores = scores.to_vec();
        sorted_scores.par_sort_by(|a, b| b.partial_cmp(a).unwrap());

        let threshold_idx = ((1.0 - percentile / 100.0) * sorted_scores.len() as Float) as usize;
        let threshold = if threshold_idx < sorted_scores.len() {
            sorted_scores[threshold_idx]
        } else {
            Float::NEG_INFINITY
        };

        Self::apply_threshold_parallel(scores, threshold)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    #[test]
    fn test_parallel_univariate_scorer() {
        let x = Array2::from_shape_vec(
            (5, 3),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0,
            ],
        )
        .unwrap();
        let y = Array1::from_vec(vec![0, 1, 0, 1, 0]);

        let scorer =
            ParallelUnivariateScorer::new(|feature: &Array1<Float>, target: &Array1<i32>| {
                Ok(feature
                    .iter()
                    .zip(target.iter())
                    .map(|(f, t)| f * (*t as Float))
                    .sum())
            });

        let scores = scorer.evaluate_parallel(&x, &y).unwrap();
        assert_eq!(scores.len(), 3);
    }

    #[test]
    fn test_parallel_feature_ranker() {
        let scores = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);
        let ranked = ParallelFeatureRanker::rank_features_parallel(&scores);

        assert_eq!(ranked[0], 3); // Highest score (0.8)
        assert_eq!(ranked[1], 1); // Second highest (0.5)
        assert_eq!(ranked[2], 2); // Third highest (0.3)
    }

    #[test]
    fn test_parallel_correlation_computer() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();
        let y = Array1::from_vec(vec![0, 1, 0, 1]);

        let correlations =
            ParallelCorrelationComputer::compute_feature_target_correlation_parallel(&x, &y)
                .unwrap();
        assert_eq!(correlations.len(), 2);
    }

    #[test]
    fn test_parallel_variance_computer() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]).unwrap();

        let variances = ParallelVarianceComputer::compute_feature_variances_parallel(&x);
        assert_eq!(variances.len(), 2);
        assert!(variances[0] > 0.0);
        assert!(variances[1] > 0.0);
    }

    #[test]
    fn test_parallel_selection_utils() {
        let scores = Array1::from_vec(vec![0.1, 0.5, 0.3, 0.8, 0.2]);

        let threshold_selected = ParallelSelectionUtils::apply_threshold_parallel(&scores, 0.4);
        assert_eq!(threshold_selected.len(), 2); // 0.5 and 0.8

        let percentile_selected = ParallelSelectionUtils::apply_percentile_parallel(&scores, 60.0);
        assert_eq!(percentile_selected.len(), 3); // Top 60% = 3 features (0.8, 0.5, 0.3)
    }
}
