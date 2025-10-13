//! Parallel processing utilities for tree algorithms
//!
//! This module provides parallel implementations of tree construction,
//! prediction, and ensemble methods using rayon.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::thread_rng;
use scirs2_core::SliceRandomExt; // For shuffle method
use sklears_core::error::Result;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

/// Parallel utilities for tree algorithms
pub struct ParallelUtils;

impl ParallelUtils {
    /// Execute a function in parallel if the parallel feature is enabled
    #[cfg(feature = "parallel")]
    pub fn maybe_parallel_map<T, U, F>(items: Vec<T>, f: F) -> Vec<U>
    where
        T: Send + Sync,
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        items.into_par_iter().map(f).collect()
    }

    /// Execute a function sequentially if parallel feature is disabled
    #[cfg(not(feature = "parallel"))]
    pub fn maybe_parallel_map<T, U, F>(items: Vec<T>, f: F) -> Vec<U>
    where
        F: Fn(T) -> U,
    {
        items.into_iter().map(f).collect()
    }

    /// Parallel bootstrap sampling for ensemble methods
    #[cfg(feature = "parallel")]
    pub fn parallel_bootstrap_samples(
        x: &Array2<f64>,
        y: &Array1<i32>,
        n_estimators: usize,
        n_samples: usize,
        random_seeds: &[u64],
    ) -> Vec<(Array2<f64>, Array1<i32>)> {
        (0..n_estimators)
            .into_par_iter()
            .map(|i| {
                let seed = random_seeds.get(i).copied().unwrap_or(42 + i as u64);
                Self::bootstrap_sample(x, y, n_samples, seed)
            })
            .collect()
    }

    /// Sequential bootstrap sampling fallback
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_bootstrap_samples(
        x: &Array2<f64>,
        y: &Array1<i32>,
        n_estimators: usize,
        n_samples: usize,
        random_seeds: &[u64],
    ) -> Vec<(Array2<f64>, Array1<i32>)> {
        (0..n_estimators)
            .map(|i| {
                let seed = random_seeds.get(i).copied().unwrap_or(42 + i as u64);
                Self::bootstrap_sample(x, y, n_samples, seed)
            })
            .collect()
    }

    /// Create a bootstrap sample from the data
    fn bootstrap_sample(
        x: &Array2<f64>,
        y: &Array1<i32>,
        n_samples: usize,
        seed: u64,
    ) -> (Array2<f64>, Array1<i32>) {
        let mut rng = thread_rng();
        let original_n_samples = x.nrows();
        let n_features = x.ncols();

        let mut bootstrap_x = Array2::zeros((n_samples, n_features));
        let mut bootstrap_y = Array1::zeros(n_samples);

        for i in 0..n_samples {
            let idx = rng.gen_range(0..original_n_samples);
            bootstrap_x.row_mut(i).assign(&x.row(idx));
            bootstrap_y[i] = y[idx];
        }

        (bootstrap_x, bootstrap_y)
    }

    /// Parallel prediction aggregation for ensemble methods
    #[cfg(feature = "parallel")]
    pub fn parallel_predict_proba_aggregate(predictions: Vec<Array2<f64>>) -> Result<Array2<f64>> {
        if predictions.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidData {
                reason: "No predictions to aggregate".to_string(),
            });
        }

        let n_samples = predictions[0].nrows();
        let n_classes = predictions[0].ncols();

        // Parallel aggregation of predictions
        let aggregated: Vec<Vec<f64>> = (0..n_samples)
            .into_par_iter()
            .map(|sample_idx| {
                let mut class_votes = vec![0.0; n_classes];
                for pred_matrix in &predictions {
                    let row = pred_matrix.row(sample_idx);
                    for (class_idx, &prob) in row.iter().enumerate() {
                        class_votes[class_idx] += prob;
                    }
                }

                // Normalize by number of estimators
                let n_estimators = predictions.len() as f64;
                class_votes
                    .iter()
                    .map(|&vote| vote / n_estimators)
                    .collect()
            })
            .collect();

        // Convert back to Array2
        let mut result = Array2::zeros((n_samples, n_classes));
        for (i, row) in aggregated.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                result[[i, j]] = val;
            }
        }

        Ok(result)
    }

    /// Sequential prediction aggregation fallback
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_predict_proba_aggregate(predictions: Vec<Array2<f64>>) -> Result<Array2<f64>> {
        if predictions.is_empty() {
            return Err(sklears_core::error::SklearsError::InvalidData {
                reason: "No predictions to aggregate".to_string(),
            });
        }

        let n_samples = predictions[0].nrows();
        let n_classes = predictions[0].ncols();
        let n_estimators = predictions.len() as f64;

        let mut result = Array2::zeros((n_samples, n_classes));

        for pred_matrix in predictions {
            result = result + pred_matrix;
        }

        result = result / n_estimators;
        Ok(result)
    }

    /// Parallel feature importance calculation using permutation
    #[cfg(feature = "parallel")]
    pub fn parallel_permutation_importance<F>(
        x: &Array2<f64>,
        y: &Array1<i32>,
        baseline_score: f64,
        scoring_fn: F,
        n_repeats: usize,
        random_seeds: &[u64],
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array2<f64>, &Array1<i32>) -> Result<f64> + Sync + Send,
    {
        let n_features = x.ncols();

        let importances: Result<Vec<f64>> = (0..n_features)
            .into_par_iter()
            .map(|feature_idx| {
                let mut importance_scores = Vec::new();

                for repeat in 0..n_repeats {
                    let seed = random_seeds
                        .get(repeat)
                        .copied()
                        .unwrap_or(42 + repeat as u64);
                    let mut x_permuted = x.clone();
                    Self::permute_feature(&mut x_permuted, feature_idx, seed)?;

                    let permuted_score = scoring_fn(&x_permuted, y)?;
                    importance_scores.push(baseline_score - permuted_score);
                }

                // Average importance across repeats
                let avg_importance =
                    importance_scores.iter().sum::<f64>() / importance_scores.len() as f64;
                Ok(avg_importance)
            })
            .collect();

        let importances = importances?;
        Ok(Array1::from_vec(importances))
    }

    /// Sequential feature importance calculation fallback
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_permutation_importance<F>(
        x: &Array2<f64>,
        y: &Array1<i32>,
        baseline_score: f64,
        scoring_fn: F,
        n_repeats: usize,
        random_seeds: &[u64],
    ) -> Result<Array1<f64>>
    where
        F: Fn(&Array2<f64>, &Array1<i32>) -> Result<f64>,
    {
        let n_features = x.ncols();
        let mut importances = Vec::with_capacity(n_features);

        for feature_idx in 0..n_features {
            let mut importance_scores = Vec::new();

            for repeat in 0..n_repeats {
                let seed = random_seeds
                    .get(repeat)
                    .copied()
                    .unwrap_or(42 + repeat as u64);
                let mut x_permuted = x.clone();
                Self::permute_feature(&mut x_permuted, feature_idx, seed)?;

                let permuted_score = scoring_fn(&x_permuted, y)?;
                importance_scores.push(baseline_score - permuted_score);
            }

            let avg_importance =
                importance_scores.iter().sum::<f64>() / importance_scores.len() as f64;
            importances.push(avg_importance);
        }

        Ok(Array1::from_vec(importances))
    }

    /// Permute values in a specific feature column
    fn permute_feature(x: &mut Array2<f64>, feature_idx: usize, seed: u64) -> Result<()> {
        let mut rng = thread_rng();
        let mut column_values: Vec<f64> = x.column(feature_idx).to_vec();
        column_values.shuffle(&mut rng);

        for (i, &value) in column_values.iter().enumerate() {
            x[[i, feature_idx]] = value;
        }

        Ok(())
    }

    /// Determine optimal number of threads to use
    pub fn optimal_n_threads(n_jobs: Option<i32>) -> usize {
        match n_jobs {
            Some(n) if n > 0 => n as usize,
            Some(-1) => num_cpus::get(),
            Some(n) if n < -1 => (num_cpus::get() as i32 + n + 1).max(1) as usize,
            _ => 1,
        }
    }

    /// Initialize thread pool with specified number of threads
    #[cfg(feature = "parallel")]
    pub fn with_thread_pool<T, F>(n_threads: usize, f: F) -> T
    where
        F: FnOnce() -> T + Send,
        T: Send,
    {
        if n_threads <= 1 {
            f()
        } else {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(n_threads)
                .build()
                .unwrap_or_else(|_| rayon::ThreadPoolBuilder::new().build().unwrap());
            pool.install(f)
        }
    }

    /// Sequential execution fallback
    #[cfg(not(feature = "parallel"))]
    pub fn with_thread_pool<T, F>(n_threads: usize, f: F) -> T
    where
        F: FnOnce() -> T,
    {
        f() // Always execute sequentially
    }

    /// Parallel evaluation of all features to find the best split
    #[cfg(feature = "parallel")]
    pub fn parallel_find_best_split(
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_indices: &[usize],
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_features = x.ncols();
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        // Select features to evaluate
        let feature_indices = Self::select_features(n_features, config)?;

        // Parallel evaluation of features
        let feature_splits: Vec<Option<FeatureSplit>> = feature_indices
            .into_par_iter()
            .map(|feature_idx| {
                Self::evaluate_feature_split(x, y, sample_indices, feature_idx, config)
                    .unwrap_or(None)
            })
            .collect();

        // Find the best split among all features
        let mut best_split = None;
        let mut best_score = f64::NEG_INFINITY;

        for split in feature_splits.into_iter().flatten() {
            if split.is_valid() && split.quality_score() > best_score {
                best_score = split.quality_score();
                best_split = Some(split);
            }
        }

        Ok(best_split)
    }

    /// Sequential fallback for finding best split
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_find_best_split(
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_indices: &[usize],
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_features = x.ncols();
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        let feature_indices = Self::select_features(n_features, config)?;

        let mut best_split = None;
        let mut best_score = f64::NEG_INFINITY;

        for feature_idx in feature_indices {
            if let Some(split) =
                Self::evaluate_feature_split(x, y, sample_indices, feature_idx, config)?
            {
                if split.is_valid() && split.quality_score() > best_score {
                    best_score = split.quality_score();
                    best_split = Some(split);
                }
            }
        }

        Ok(best_split)
    }

    /// Parallel evaluation of classification splits
    #[cfg(feature = "parallel")]
    pub fn parallel_find_best_classification_split(
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_indices: &[usize],
        n_classes: usize,
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_features = x.ncols();
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        let feature_indices = Self::select_features(n_features, config)?;

        // Parallel evaluation of features for classification
        let feature_splits: Vec<Option<FeatureSplit>> = feature_indices
            .into_par_iter()
            .map(|feature_idx| {
                Self::evaluate_classification_split(
                    x,
                    y,
                    sample_indices,
                    feature_idx,
                    n_classes,
                    config,
                )
                .unwrap_or(None)
            })
            .collect();

        // Find the best split
        let mut best_split = None;
        let mut best_score = f64::NEG_INFINITY;

        for split in feature_splits.into_iter().flatten() {
            if split.is_valid() && split.quality_score() > best_score {
                best_score = split.quality_score();
                best_split = Some(split);
            }
        }

        Ok(best_split)
    }

    /// Sequential fallback for classification splits
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_find_best_classification_split(
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_indices: &[usize],
        n_classes: usize,
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_features = x.ncols();
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        let feature_indices = Self::select_features(n_features, config)?;

        let mut best_split = None;
        let mut best_score = f64::NEG_INFINITY;

        for feature_idx in feature_indices {
            if let Some(split) = Self::evaluate_classification_split(
                x,
                y,
                sample_indices,
                feature_idx,
                n_classes,
                config,
            )? {
                if split.is_valid() && split.quality_score() > best_score {
                    best_score = split.quality_score();
                    best_split = Some(split);
                }
            }
        }

        Ok(best_split)
    }

    /// Parallel computation of feature statistics for multiple features
    #[cfg(feature = "parallel")]
    pub fn parallel_compute_feature_stats(
        x: &Array2<f64>,
        sample_indices: &[usize],
        feature_indices: &[usize],
    ) -> Vec<FeatureStats> {
        feature_indices
            .par_iter()
            .map(|&feature_idx| Self::compute_feature_stats(x, sample_indices, feature_idx))
            .collect()
    }

    /// Sequential fallback for feature statistics
    #[cfg(not(feature = "parallel"))]
    pub fn parallel_compute_feature_stats(
        x: &Array2<f64>,
        sample_indices: &[usize],
        feature_indices: &[usize],
    ) -> Vec<FeatureStats> {
        feature_indices
            .iter()
            .map(|&feature_idx| Self::compute_feature_stats(x, sample_indices, feature_idx))
            .collect()
    }

    /// Select features to evaluate based on configuration
    fn select_features(n_features: usize, config: &ParallelFeatureConfig) -> Result<Vec<usize>> {
        let max_features = config.max_features.unwrap_or(n_features);
        let n_features_to_use = max_features.min(n_features);

        if n_features_to_use >= n_features {
            // Use all features
            Ok((0..n_features).collect())
        } else {
            // Randomly sample features
            let mut rng = thread_rng();

            let mut all_features: Vec<usize> = (0..n_features).collect();
            all_features.shuffle(&mut rng);
            all_features.truncate(n_features_to_use);

            Ok(all_features)
        }
    }

    /// Evaluate a specific feature for regression splits
    fn evaluate_feature_split(
        x: &Array2<f64>,
        y: &Array1<f64>,
        sample_indices: &[usize],
        feature_idx: usize,
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        // Collect feature values and targets for samples
        let mut feature_target_pairs: Vec<(f64, f64)> = sample_indices
            .iter()
            .map(|&idx| (x[[idx, feature_idx]], y[idx]))
            .collect();

        // Sort by feature value
        feature_target_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_threshold = 0.0;
        let mut best_impurity_reduction = f64::NEG_INFINITY;
        let mut best_n_left = 0;
        let mut best_n_right = 0;

        // Calculate initial statistics
        let total_sum: f64 = feature_target_pairs.iter().map(|(_, target)| target).sum();
        let total_sum_sq: f64 = feature_target_pairs
            .iter()
            .map(|(_, target)| target * target)
            .sum();
        let total_variance =
            (total_sum_sq / n_samples as f64) - (total_sum / n_samples as f64).powi(2);

        // Try different split points
        for i in 1..n_samples {
            let current_val = feature_target_pairs[i - 1].0;
            let next_val = feature_target_pairs[i].0;

            if (next_val - current_val).abs() < 1e-10 {
                continue; // Skip if values are the same
            }

            let n_left = i;
            let n_right = n_samples - i;

            if n_left < config.min_samples_leaf || n_right < config.min_samples_leaf {
                continue;
            }

            // Calculate left and right statistics
            let left_sum: f64 = feature_target_pairs[..i]
                .iter()
                .map(|(_, target)| target)
                .sum();
            let left_sum_sq: f64 = feature_target_pairs[..i]
                .iter()
                .map(|(_, target)| target * target)
                .sum();
            let left_variance = (left_sum_sq / n_left as f64) - (left_sum / n_left as f64).powi(2);

            let right_sum = total_sum - left_sum;
            let right_sum_sq = total_sum_sq - left_sum_sq;
            let right_variance =
                (right_sum_sq / n_right as f64) - (right_sum / n_right as f64).powi(2);

            // Calculate weighted variance reduction
            let weighted_variance = (n_left as f64 / n_samples as f64) * left_variance
                + (n_right as f64 / n_samples as f64) * right_variance;
            let impurity_reduction = total_variance - weighted_variance;

            if impurity_reduction > best_impurity_reduction {
                best_impurity_reduction = impurity_reduction;
                best_threshold = (current_val + next_val) / 2.0;
                best_n_left = n_left;
                best_n_right = n_right;
            }
        }

        if best_impurity_reduction > config.min_impurity_decrease {
            Ok(Some(FeatureSplit::new(
                feature_idx,
                best_threshold,
                best_impurity_reduction,
                best_n_left,
                best_n_right,
            )))
        } else {
            Ok(None)
        }
    }

    /// Evaluate a specific feature for classification splits
    fn evaluate_classification_split(
        x: &Array2<f64>,
        y: &Array1<i32>,
        sample_indices: &[usize],
        feature_idx: usize,
        n_classes: usize,
        config: &ParallelFeatureConfig,
    ) -> Result<Option<FeatureSplit>> {
        let n_samples = sample_indices.len();

        if n_samples < config.min_samples_split {
            return Ok(None);
        }

        // Collect feature values and targets
        let mut feature_target_pairs: Vec<(f64, i32)> = sample_indices
            .iter()
            .map(|&idx| (x[[idx, feature_idx]], y[idx]))
            .collect();

        // Sort by feature value
        feature_target_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let mut best_threshold = 0.0;
        let mut best_information_gain = f64::NEG_INFINITY;
        let mut best_n_left = 0;
        let mut best_n_right = 0;

        // Calculate initial entropy
        let mut class_counts = vec![0; n_classes];
        for (_, class) in &feature_target_pairs {
            if *class >= 0 && (*class as usize) < n_classes {
                class_counts[*class as usize] += 1;
            }
        }

        let initial_entropy = Self::calculate_entropy(&class_counts, n_samples);

        // Try different split points
        for i in 1..n_samples {
            let current_val = feature_target_pairs[i - 1].0;
            let next_val = feature_target_pairs[i].0;

            if (next_val - current_val).abs() < 1e-10 {
                continue;
            }

            let n_left = i;
            let n_right = n_samples - i;

            if n_left < config.min_samples_leaf || n_right < config.min_samples_leaf {
                continue;
            }

            // Calculate left and right class distributions
            let mut left_counts = vec![0; n_classes];
            let mut right_counts = vec![0; n_classes];

            for j in 0..i {
                let class = feature_target_pairs[j].1;
                if class >= 0 && (class as usize) < n_classes {
                    left_counts[class as usize] += 1;
                }
            }

            for j in i..n_samples {
                let class = feature_target_pairs[j].1;
                if class >= 0 && (class as usize) < n_classes {
                    right_counts[class as usize] += 1;
                }
            }

            // Calculate entropies
            let left_entropy = Self::calculate_entropy(&left_counts, n_left);
            let right_entropy = Self::calculate_entropy(&right_counts, n_right);

            // Calculate weighted entropy
            let weighted_entropy = (n_left as f64 / n_samples as f64) * left_entropy
                + (n_right as f64 / n_samples as f64) * right_entropy;

            let information_gain = initial_entropy - weighted_entropy;

            if information_gain > best_information_gain {
                best_information_gain = information_gain;
                best_threshold = (current_val + next_val) / 2.0;
                best_n_left = n_left;
                best_n_right = n_right;
            }
        }

        if best_information_gain > config.min_impurity_decrease {
            Ok(Some(
                FeatureSplit::new(
                    feature_idx,
                    best_threshold,
                    best_information_gain, // Use information gain as impurity reduction
                    best_n_left,
                    best_n_right,
                )
                .with_information_gain(best_information_gain),
            ))
        } else {
            Ok(None)
        }
    }

    /// Calculate entropy for classification
    fn calculate_entropy(class_counts: &[usize], total_samples: usize) -> f64 {
        if total_samples == 0 {
            return 0.0;
        }

        let mut entropy = 0.0;
        for &count in class_counts {
            if count > 0 {
                let probability = count as f64 / total_samples as f64;
                entropy -= probability * probability.log2();
            }
        }

        entropy
    }

    /// Compute statistics for a single feature
    fn compute_feature_stats(
        x: &Array2<f64>,
        sample_indices: &[usize],
        feature_idx: usize,
    ) -> FeatureStats {
        let values: Vec<f64> = sample_indices
            .iter()
            .map(|&idx| x[[idx, feature_idx]])
            .collect();

        if values.is_empty() {
            return FeatureStats::default();
        }

        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let n = values.len();
        let sum: f64 = values.iter().sum();
        let mean = sum / n as f64;

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / n as f64;

        let std_dev = variance.sqrt();
        let min = sorted_values[0];
        let max = sorted_values[n - 1];
        let median = if n % 2 == 0 {
            (sorted_values[n / 2 - 1] + sorted_values[n / 2]) / 2.0
        } else {
            sorted_values[n / 2]
        };

        // Count unique values
        let mut unique_values = sorted_values.clone();
        unique_values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        let n_unique = unique_values.len();

        FeatureStats {
            feature_idx,
            mean,
            std_dev,
            min,
            max,
            median,
            n_unique,
            n_samples: n,
        }
    }
}

/// Parallel iterator extensions for tree algorithms
pub trait ParallelTreeExt<T> {
    /// Process items in parallel if the parallel feature is enabled
    fn maybe_parallel_process<U, F>(self, f: F) -> Vec<U>
    where
        T: Send + Sync,
        U: Send,
        F: Fn(T) -> U + Sync + Send;
}

impl<I, T> ParallelTreeExt<T> for I
where
    I: IntoIterator<Item = T>,
    I::IntoIter: Send,
    T: Send + Sync,
{
    #[cfg(feature = "parallel")]
    fn maybe_parallel_process<U, F>(self, f: F) -> Vec<U>
    where
        T: Send + Sync,
        U: Send,
        F: Fn(T) -> U + Sync + Send,
    {
        self.into_iter()
            .collect::<Vec<_>>()
            .into_par_iter()
            .map(f)
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    fn maybe_parallel_process<U, F>(self, f: F) -> Vec<U>
    where
        F: Fn(T) -> U,
    {
        self.into_iter().map(f).collect()
    }
}

/// Parallel feature evaluation for tree construction
#[derive(Debug, Clone)]
pub struct FeatureSplit {
    /// Feature index
    pub feature_idx: usize,
    /// Best threshold for this feature
    pub threshold: f64,
    /// Impurity reduction achieved by this split
    pub impurity_reduction: f64,
    /// Number of samples that would go left
    pub n_left: usize,
    /// Number of samples that would go right
    pub n_right: usize,
    /// Information gain (for classification)
    pub information_gain: Option<f64>,
}

impl FeatureSplit {
    /// Create a new feature split
    pub fn new(
        feature_idx: usize,
        threshold: f64,
        impurity_reduction: f64,
        n_left: usize,
        n_right: usize,
    ) -> Self {
        Self {
            feature_idx,
            threshold,
            impurity_reduction,
            n_left,
            n_right,
            information_gain: None,
        }
    }

    /// Set information gain for classification
    pub fn with_information_gain(mut self, gain: f64) -> Self {
        self.information_gain = Some(gain);
        self
    }

    /// Check if this split is valid (has samples on both sides)
    pub fn is_valid(&self) -> bool {
        self.n_left > 0 && self.n_right > 0
    }

    /// Get the split quality score
    pub fn quality_score(&self) -> f64 {
        self.information_gain.unwrap_or(self.impurity_reduction)
    }
}

/// Configuration for parallel feature evaluation
#[derive(Debug, Clone)]
pub struct ParallelFeatureConfig {
    /// Minimum number of samples required to split
    pub min_samples_split: usize,
    /// Minimum number of samples required in a leaf
    pub min_samples_leaf: usize,
    /// Minimum impurity decrease required for a split
    pub min_impurity_decrease: f64,
    /// Maximum number of features to consider (None = all features)
    pub max_features: Option<usize>,
    /// Random seed for feature sampling
    pub random_state: Option<u64>,
}

impl Default for ParallelFeatureConfig {
    fn default() -> Self {
        Self {
            min_samples_split: 2,
            min_samples_leaf: 1,
            min_impurity_decrease: 0.0,
            max_features: None,
            random_state: None,
        }
    }
}

/// Statistics for a feature
#[derive(Debug, Clone)]
pub struct FeatureStats {
    /// Feature index
    pub feature_idx: usize,
    /// Mean value
    pub mean: f64,
    /// Standard deviation
    pub std_dev: f64,
    /// Minimum value
    pub min: f64,
    /// Maximum value
    pub max: f64,
    /// Median value
    pub median: f64,
    /// Number of unique values
    pub n_unique: usize,
    /// Number of samples
    pub n_samples: usize,
}

impl Default for FeatureStats {
    fn default() -> Self {
        Self {
            feature_idx: 0,
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            median: 0.0,
            n_unique: 0,
            n_samples: 0,
        }
    }
}

impl FeatureStats {
    /// Check if this feature has enough variation to be useful for splitting
    pub fn is_informative(&self) -> bool {
        self.n_unique > 1 && self.std_dev > 1e-10
    }

    /// Get the range of the feature
    pub fn range(&self) -> f64 {
        self.max - self.min
    }

    /// Get coefficient of variation (std_dev / mean)
    pub fn coefficient_of_variation(&self) -> f64 {
        if self.mean.abs() > 1e-10 {
            self.std_dev / self.mean.abs()
        } else {
            f64::INFINITY
        }
    }
}
