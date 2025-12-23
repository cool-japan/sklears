//! Custom cross-validation implementations
//!
//! This module provides flexible cross-validation strategies that allow users to define
//! custom splitting logic or use predefined splits for specific use cases.

use scirs2_core::ndarray::Array1;

use crate::CrossValidator;

/// Custom cross-validation iterator that allows users to define their own splitting logic
///
/// This provides a flexible way to implement custom cross-validation strategies
/// by allowing users to provide their own splitting function.
///
/// # Example
/// ```rust
/// use sklears_model_selection::{CustomCrossValidator, CrossValidator};
/// use scirs2_core::ndarray::Array1;
///
/// // Example: Simple 2-fold split that alternates samples
/// let custom_cv = CustomCrossValidator::new(
///     2,
///     Box::new(|n_samples, _y: Option<&Array1<i32>>| {
///         let mut splits = Vec::new();
///         
///         // First fold: even indices for train, odd for test
///         let train1: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 0).collect();
///         let test1: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 1).collect();
///         splits.push((train1, test1));
///         
///         // Second fold: odd indices for train, even for test
///         let train2: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 1).collect();
///         let test2: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 0).collect();
///         splits.push((train2, test2));
///         
///         splits
///     })
/// );
/// ```
/// Type alias for the split function
type SplitFn =
    Box<dyn Fn(usize, Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> + Send + Sync>;

pub struct CustomCrossValidator {
    n_splits: usize,
    split_fn: SplitFn,
}

impl CustomCrossValidator {
    /// Create a new custom cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of splits this validator will generate
    /// * `split_fn` - Function that takes n_samples and optional labels and returns train/test index pairs
    pub fn new<F>(n_splits: usize, split_fn: F) -> Self
    where
        F: Fn(usize, Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> + Send + Sync + 'static,
    {
        Self {
            n_splits,
            split_fn: Box::new(split_fn),
        }
    }
}

impl CrossValidator for CustomCrossValidator {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        (self.split_fn)(n_samples, y)
    }
}

/// Block cross-validation for time series or sequential data
///
/// Splits the data into blocks where each fold uses a contiguous block for testing
/// and the preceding data for training. This is useful for time series data where
/// we want to respect temporal order.
///
/// # Example
/// ```rust
/// use sklears_model_selection::{BlockCrossValidator, CrossValidator};
///
/// let block_cv = BlockCrossValidator::new(3);
/// let splits = block_cv.split(12, None);
/// // This would create 3 folds with blocks of 4 samples each
/// ```
#[derive(Debug, Clone)]
pub struct BlockCrossValidator {
    n_splits: usize,
    test_size: Option<usize>,
    gap: usize,
}

impl BlockCrossValidator {
    /// Create a new block cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of blocks to create
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            test_size: None,
            gap: 0,
        }
    }

    /// Set the size of each test block
    pub fn test_size(mut self, test_size: usize) -> Self {
        self.test_size = Some(test_size);
        self
    }

    /// Set the gap between train and test sets to avoid data leakage
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }
}

impl CrossValidator for BlockCrossValidator {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let test_size = self.test_size.unwrap_or(n_samples / self.n_splits);

        assert!(
            test_size * self.n_splits <= n_samples,
            "Test size too large for number of samples"
        );

        let mut splits = Vec::new();

        for fold in 0..self.n_splits {
            let test_start = fold * test_size;
            let test_end = (test_start + test_size).min(n_samples);

            // Train on all data before test set (with gap)
            let train_end = test_start.saturating_sub(self.gap);

            let train_indices: Vec<usize> = (0..train_end).collect();
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            if !train_indices.is_empty() && !test_indices.is_empty() {
                splits.push((train_indices, test_indices));
            }
        }

        splits
    }
}

/// Predefined Split cross-validator
///
/// Uses user-provided split indices to generate train/test splits.
/// This is useful when you have a specific validation strategy or when
/// cross-validation folds are predefined by the problem.
#[derive(Debug, Clone)]
pub struct PredefinedSplit {
    test_fold: Array1<i32>,
}

impl PredefinedSplit {
    /// Create a new PredefinedSplit cross-validator
    ///
    /// # Arguments
    /// * `test_fold` - Array where test_fold\[i\] is the fold number for sample i.
    ///   A value of -1 indicates that the sample should always be in the training set.
    pub fn new(test_fold: Array1<i32>) -> Self {
        // Validate that fold indices are valid (-1 or non-negative)
        for &fold in test_fold.iter() {
            assert!(
                fold >= -1,
                "test_fold values must be -1 or non-negative, got {fold}"
            );
        }
        Self { test_fold }
    }

    /// Get the number of unique folds (excluding -1)
    fn get_n_splits(&self) -> usize {
        let unique_folds: std::collections::HashSet<i32> = self
            .test_fold
            .iter()
            .filter(|&&x| x >= 0)
            .cloned()
            .collect();
        unique_folds.len()
    }
}

impl CrossValidator for PredefinedSplit {
    fn n_splits(&self) -> usize {
        self.get_n_splits()
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            self.test_fold.len(),
            n_samples,
            "test_fold must have the same length as n_samples"
        );

        // Get unique fold numbers (excluding -1)
        let mut unique_folds: Vec<i32> = self
            .test_fold
            .iter()
            .filter(|&&x| x >= 0)
            .cloned()
            .collect::<std::collections::HashSet<_>>()
            .into_iter()
            .collect();
        unique_folds.sort();

        let mut splits = Vec::new();

        for &fold in &unique_folds {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            for (idx, &sample_fold) in self.test_fold.iter().enumerate() {
                if sample_fold == fold {
                    test_indices.push(idx);
                } else if sample_fold == -1 || sample_fold != fold {
                    train_indices.push(idx);
                }
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_custom_cross_validator() {
        // Create a custom CV that alternates samples
        let custom_cv = CustomCrossValidator::new(2, |n_samples, _y| {
            let mut splits = Vec::new();

            // First fold: even indices for train, odd for test
            let train1: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 0).collect();
            let test1: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 1).collect();
            splits.push((train1, test1));

            // Second fold: odd indices for train, even for test
            let train2: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 1).collect();
            let test2: Vec<usize> = (0..n_samples).filter(|&i| i % 2 == 0).collect();
            splits.push((train2, test2));

            splits
        });

        let splits = custom_cv.split(6, None);
        assert_eq!(splits.len(), 2);

        // Check first fold
        assert_eq!(splits[0].0, vec![0, 2, 4]); // even indices for train
        assert_eq!(splits[0].1, vec![1, 3, 5]); // odd indices for test

        // Check second fold
        assert_eq!(splits[1].0, vec![1, 3, 5]); // odd indices for train
        assert_eq!(splits[1].1, vec![0, 2, 4]); // even indices for test
    }

    #[test]
    fn test_block_cross_validator() {
        let block_cv = BlockCrossValidator::new(3).test_size(2);
        let splits = block_cv.split(8, None);

        // Should create 2 folds (first fold is skipped because no training data)
        assert_eq!(splits.len(), 2);

        // First fold: train on [0, 1], test on [2, 3]
        assert_eq!(splits[0].0, vec![0, 1]);
        assert_eq!(splits[0].1, vec![2, 3]);

        // Second fold: train on [0, 1, 2, 3], test on [4, 5]
        assert_eq!(splits[1].0, vec![0, 1, 2, 3]);
        assert_eq!(splits[1].1, vec![4, 5]);
    }

    #[test]
    fn test_block_cross_validator_with_gap() {
        let block_cv = BlockCrossValidator::new(3).test_size(2).gap(1);
        let splits = block_cv.split(8, None);

        // Should create folds with gap between train and test
        assert_eq!(splits.len(), 2); // First fold will be skipped due to gap

        // Second fold: train on [0], test on [2, 3] (gap of 1 at index 1)
        assert_eq!(splits[0].0, vec![0]);
        assert_eq!(splits[0].1, vec![2, 3]);

        // Third fold: train on [0, 1, 2], test on [4, 5] (gap of 1 at index 3)
        assert_eq!(splits[1].0, vec![0, 1, 2]);
        assert_eq!(splits[1].1, vec![4, 5]);
    }

    #[test]
    fn test_predefined_split() {
        // Define custom folds: -1 means always in train, 0/1/2 are fold numbers
        let test_fold = array![-1, 0, 0, 1, 1, 2, 2, -1];
        let cv = PredefinedSplit::new(test_fold);

        assert_eq!(cv.n_splits(), 3);

        let splits = cv.split(8, None::<&Array1<i32>>);
        assert_eq!(splits.len(), 3);

        // Check first fold (test fold 0)
        assert_eq!(splits[0].1, vec![1, 2]); // indices with fold 0
        assert!(splits[0].0.contains(&0)); // index 0 (fold -1) should be in train
        assert!(splits[0].0.contains(&7)); // index 7 (fold -1) should be in train

        // Check second fold (test fold 1)
        assert_eq!(splits[1].1, vec![3, 4]); // indices with fold 1

        // Check third fold (test fold 2)
        assert_eq!(splits[2].1, vec![5, 6]); // indices with fold 2

        // Verify that samples with fold -1 are always in training sets
        for (train, _) in &splits {
            assert!(train.contains(&0));
            assert!(train.contains(&7));
        }
    }

    #[test]
    fn test_predefined_split_edge_cases() {
        // Test with all samples in training (all -1)
        let test_fold = array![-1, -1, -1, -1];
        let cv = PredefinedSplit::new(test_fold);
        assert_eq!(cv.n_splits(), 0);
        let splits = cv.split(4, None::<&Array1<i32>>);
        assert_eq!(splits.len(), 0);

        // Test with single fold
        let test_fold = array![0, 0, -1, -1];
        let cv = PredefinedSplit::new(test_fold);
        assert_eq!(cv.n_splits(), 1);
        let splits = cv.split(4, None::<&Array1<i32>>);
        assert_eq!(splits.len(), 1);
        assert_eq!(splits[0].1, vec![0, 1]);
        assert_eq!(splits[0].0, vec![2, 3]);
    }
}
