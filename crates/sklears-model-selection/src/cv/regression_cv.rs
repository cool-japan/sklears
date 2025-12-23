//! Regression-specific cross-validation iterators

use super::RegressionCrossValidator;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use scirs2_core::SliceRandomExt;
use sklears_core::types::Float;
use std::collections::HashMap;

/// Stratified K-Fold cross-validation for regression tasks
///
/// This cross-validator stratifies continuous target values by binning them into quantiles
/// and then performs stratified sampling based on these bins. This ensures that each fold
/// has a representative distribution of target values, which is particularly useful for
/// regression problems with non-uniform target distributions.
#[derive(Debug, Clone)]
pub struct StratifiedRegressionKFold {
    n_splits: usize,
    n_bins: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedRegressionKFold {
    /// Create a new StratifiedRegressionKFold cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            n_bins: 10, // Default to 10 bins
            shuffle: false,
            random_state: None,
        }
    }

    /// Set the number of bins for stratifying continuous targets
    pub fn n_bins(mut self, n_bins: usize) -> Self {
        assert!(n_bins >= 2, "n_bins must be at least 2");
        self.n_bins = n_bins;
        self
    }

    /// Set whether to shuffle the data before splitting
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random state for shuffling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Convert continuous targets to discrete bins using quantile-based binning
    fn create_bins(&self, y: &Array1<Float>) -> Array1<i32> {
        let n_samples = y.len();
        let mut y_sorted: Vec<(Float, usize)> =
            y.iter().enumerate().map(|(i, &val)| (val, i)).collect();
        y_sorted.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut bins = Array1::<i32>::zeros(n_samples);
        let bin_size = n_samples as f64 / self.n_bins as f64;

        for (rank, &(_val, orig_idx)) in y_sorted.iter().enumerate() {
            let bin = ((rank as f64 / bin_size).floor() as usize).min(self.n_bins - 1);
            bins[orig_idx] = bin as i32;
        }

        bins
    }

    /// Calculate the size of each fold
    fn calculate_fold_sizes(&self, n_samples: usize) -> Vec<usize> {
        let min_fold_size = n_samples / self.n_splits;
        let n_larger_folds = n_samples % self.n_splits;

        let mut fold_sizes = vec![min_fold_size; self.n_splits];
        for fold_size in fold_sizes.iter_mut().take(n_larger_folds) {
            *fold_size += 1;
        }

        fold_sizes
    }
}

impl RegressionCrossValidator for StratifiedRegressionKFold {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split_regression(
        &self,
        n_samples: usize,
        y: &Array1<Float>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            y.len(),
            n_samples,
            "y must have the same length as n_samples"
        );
        assert!(
            self.n_splits <= n_samples,
            "Cannot have number of splits {} greater than the number of samples {}",
            self.n_splits,
            n_samples
        );

        // Convert continuous targets to discrete bins
        let y_binned = self.create_bins(y);

        // Group indices by bin
        let mut bin_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &bin) in y_binned.iter().enumerate() {
            bin_indices.entry(bin).or_default().push(idx);
        }

        // Check we have enough samples in each bin
        for indices in bin_indices.values() {
            assert!(
                indices.len() >= self.n_splits,
                "The least populated bin has only {} members, which is less than n_splits={}",
                indices.len(),
                self.n_splits
            );
        }

        // Shuffle within each bin if requested
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    use scirs2_core::random::thread_rng;
                    StdRng::from_rng(&mut thread_rng())
                }
            };
            for indices in bin_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Create stratified folds
        let mut splits = vec![(Vec::new(), Vec::new()); self.n_splits];

        for (_bin, indices) in bin_indices {
            let fold_sizes = self.calculate_fold_sizes(indices.len());
            let mut current = 0;

            for i in 0..self.n_splits {
                let fold_size = fold_sizes[i];
                let test_end = current + fold_size;

                // Add to test set for this fold
                splits[i].1.extend(&indices[current..test_end]);

                // Add to train sets for other folds
                for (j, split) in splits.iter_mut().enumerate().take(self.n_splits) {
                    if i != j {
                        split.0.extend(&indices[current..test_end]);
                    }
                }

                current = test_end;
            }
        }

        splits
    }
}
