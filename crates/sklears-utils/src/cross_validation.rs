//! Advanced cross-validation utilities for machine learning
//!
//! This module provides sophisticated cross-validation techniques including
//! stratified k-fold, time series cross-validation, group-based CV, and more.

use crate::{UtilsError, UtilsResult};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use std::collections::HashMap;

/// Cross-validation split information
#[derive(Clone, Debug)]
pub struct CVSplit {
    /// Training indices
    pub train: Vec<usize>,
    /// Test indices
    pub test: Vec<usize>,
}

/// Stratified K-Fold cross-validation generator
///
/// Generates k-fold splits while maintaining class distribution in each fold.
/// Useful for classification tasks with imbalanced classes.
#[derive(Clone, Debug)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new stratified k-fold splitter
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds (must be >= 2)
    /// * `shuffle` - Whether to shuffle data before splitting
    /// * `random_state` - Random seed for reproducibility
    pub fn new(n_splits: usize, shuffle: bool, random_state: Option<u64>) -> UtilsResult<Self> {
        if n_splits < 2 {
            return Err(UtilsError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }

        Ok(Self {
            n_splits,
            shuffle,
            random_state,
        })
    }

    /// Generate stratified k-fold splits for given labels
    ///
    /// # Arguments
    /// * `y` - Class labels (0-indexed integers)
    ///
    /// # Returns
    /// Vector of CVSplit structures containing train/test indices for each fold
    pub fn split(&self, y: &[usize]) -> UtilsResult<Vec<CVSplit>> {
        if y.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Cannot split empty label array".to_string(),
            ));
        }

        // Group indices by class
        let mut class_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        // Check that each class has at least n_splits samples
        for (class, indices) in &class_indices {
            if indices.len() < self.n_splits {
                return Err(UtilsError::InvalidParameter(format!(
                    "Class {class} has only {} samples, need at least {} for {}-fold CV",
                    indices.len(),
                    self.n_splits,
                    self.n_splits
                )));
            }
        }

        // Shuffle class indices if requested
        let mut rng = self
            .random_state
            .map(StdRng::seed_from_u64)
            .unwrap_or_else(|| StdRng::seed_from_u64(42));

        if self.shuffle {
            for indices in class_indices.values_mut() {
                Self::shuffle_indices(indices, &mut rng);
            }
        }

        // Create folds by distributing each class's samples
        let mut fold_indices: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];

        for indices in class_indices.values() {
            let fold_sizes = Self::distribute_samples(indices.len(), self.n_splits);
            let mut current_idx = 0;

            for (fold_id, size) in fold_sizes.iter().enumerate() {
                fold_indices[fold_id].extend(&indices[current_idx..current_idx + size]);
                current_idx += size;
            }
        }

        // Generate train/test splits
        let mut splits = Vec::with_capacity(self.n_splits);
        for test_fold_id in 0..self.n_splits {
            let mut train = Vec::new();
            for (fold_id, indices) in fold_indices.iter().enumerate() {
                if fold_id != test_fold_id {
                    train.extend(indices);
                }
            }

            splits.push(CVSplit {
                train,
                test: fold_indices[test_fold_id].clone(),
            });
        }

        Ok(splits)
    }

    fn shuffle_indices(indices: &mut [usize], rng: &mut StdRng) {
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    fn distribute_samples(n_samples: usize, n_folds: usize) -> Vec<usize> {
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

/// Time Series Cross-Validation
///
/// Generates splits suitable for time series data where temporal order must be preserved.
/// Uses expanding window approach where training set grows with each split.
#[derive(Clone, Debug)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    test_size: Option<usize>,
    gap: usize,
}

impl TimeSeriesSplit {
    /// Create a new time series cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of splits to generate
    /// * `test_size` - Size of test set (None = automatic sizing)
    /// * `gap` - Number of samples to exclude between train and test sets
    pub fn new(n_splits: usize, test_size: Option<usize>, gap: usize) -> UtilsResult<Self> {
        if n_splits < 2 {
            return Err(UtilsError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }

        Ok(Self {
            n_splits,
            test_size,
            gap,
        })
    }

    /// Generate time series splits
    ///
    /// # Arguments
    /// * `n_samples` - Total number of samples
    ///
    /// # Returns
    /// Vector of CVSplit structures with expanding training windows
    pub fn split(&self, n_samples: usize) -> UtilsResult<Vec<CVSplit>> {
        let test_size = self.test_size.unwrap_or_else(|| {
            // Default: divide remaining samples after first split
            (n_samples - (n_samples / (self.n_splits + 1))) / self.n_splits
        });

        let min_train_size = n_samples / (self.n_splits + 1);

        if min_train_size + self.gap + test_size > n_samples {
            return Err(UtilsError::InvalidParameter(
                "Not enough samples for requested split configuration".to_string(),
            ));
        }

        let mut splits = Vec::with_capacity(self.n_splits);

        for i in 0..self.n_splits {
            let train_end = min_train_size + i * test_size;
            let test_start = train_end + self.gap;
            let test_end = test_start + test_size;

            if test_end > n_samples {
                break;
            }

            splits.push(CVSplit {
                train: (0..train_end).collect(),
                test: (test_start..test_end).collect(),
            });
        }

        if splits.len() < self.n_splits {
            return Err(UtilsError::InvalidParameter(
                "Cannot generate requested number of splits with given parameters".to_string(),
            ));
        }

        Ok(splits)
    }
}

/// Group K-Fold Cross-Validation
///
/// Ensures that samples from the same group are not in both train and test sets.
/// Useful for preventing data leakage in scenarios like patient data, user data, etc.
#[derive(Clone, Debug)]
pub struct GroupKFold {
    n_splits: usize,
}

impl GroupKFold {
    /// Create a new group k-fold splitter
    pub fn new(n_splits: usize) -> UtilsResult<Self> {
        if n_splits < 2 {
            return Err(UtilsError::InvalidParameter(
                "n_splits must be at least 2".to_string(),
            ));
        }

        Ok(Self { n_splits })
    }

    /// Generate group-based k-fold splits
    ///
    /// # Arguments
    /// * `groups` - Group identifier for each sample
    ///
    /// # Returns
    /// Vector of CVSplit structures ensuring group separation
    pub fn split(&self, groups: &[usize]) -> UtilsResult<Vec<CVSplit>> {
        if groups.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Cannot split empty groups array".to_string(),
            ));
        }

        // Map groups to sample indices
        let mut group_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_to_indices.entry(group).or_default().push(idx);
        }

        let unique_groups: Vec<usize> = group_to_indices.keys().copied().collect();

        if unique_groups.len() < self.n_splits {
            return Err(UtilsError::InvalidParameter(format!(
                "Number of unique groups ({}) must be >= n_splits ({})",
                unique_groups.len(),
                self.n_splits
            )));
        }

        // Distribute groups into folds
        let fold_sizes = Self::distribute_groups(unique_groups.len(), self.n_splits);
        let mut group_folds: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];
        let mut current_idx = 0;

        for (fold_id, size) in fold_sizes.iter().enumerate() {
            group_folds[fold_id].extend(&unique_groups[current_idx..current_idx + size]);
            current_idx += size;
        }

        // Generate splits
        let mut splits = Vec::with_capacity(self.n_splits);

        for test_fold_id in 0..self.n_splits {
            let mut train = Vec::new();
            let mut test = Vec::new();

            for (fold_id, groups_in_fold) in group_folds.iter().enumerate() {
                let indices: Vec<usize> = groups_in_fold
                    .iter()
                    .flat_map(|g| group_to_indices.get(g).unwrap())
                    .copied()
                    .collect();

                if fold_id == test_fold_id {
                    test.extend(indices);
                } else {
                    train.extend(indices);
                }
            }

            splits.push(CVSplit { train, test });
        }

        Ok(splits)
    }

    fn distribute_groups(n_groups: usize, n_folds: usize) -> Vec<usize> {
        let base_size = n_groups / n_folds;
        let remainder = n_groups % n_folds;

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

/// Leave-One-Group-Out Cross-Validation
///
/// Creates one split for each unique group, using that group as test set.
#[derive(Clone, Debug)]
pub struct LeaveOneGroupOut;

impl LeaveOneGroupOut {
    /// Create a new leave-one-group-out splitter
    pub fn new() -> Self {
        Self
    }

    /// Generate leave-one-group-out splits
    ///
    /// # Arguments
    /// * `groups` - Group identifier for each sample
    ///
    /// # Returns
    /// Vector of CVSplit structures, one per unique group
    pub fn split(&self, groups: &[usize]) -> UtilsResult<Vec<CVSplit>> {
        if groups.is_empty() {
            return Err(UtilsError::InvalidParameter(
                "Cannot split empty groups array".to_string(),
            ));
        }

        // Map groups to indices
        let mut group_to_indices: HashMap<usize, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_to_indices.entry(group).or_default().push(idx);
        }

        let unique_groups: Vec<usize> = group_to_indices.keys().copied().collect();
        let mut splits = Vec::with_capacity(unique_groups.len());

        for &test_group in &unique_groups {
            let mut train = Vec::new();
            let test = group_to_indices.get(&test_group).unwrap().clone();

            for &group in &unique_groups {
                if group != test_group {
                    train.extend(group_to_indices.get(&group).unwrap());
                }
            }

            splits.push(CVSplit { train, test });
        }

        Ok(splits)
    }
}

impl Default for LeaveOneGroupOut {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stratified_kfold_basic() {
        let y = vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let skf = StratifiedKFold::new(3, false, Some(42)).unwrap();
        let splits = skf.split(&y).unwrap();

        assert_eq!(splits.len(), 3);

        // Check that each fold has samples from all classes
        for split in &splits {
            let test_labels: Vec<usize> = split.test.iter().map(|&i| y[i]).collect();
            assert!(test_labels.contains(&0));
            assert!(test_labels.contains(&1));
            assert!(test_labels.contains(&2));
        }
    }

    #[test]
    fn test_stratified_kfold_all_samples_used() {
        let y = vec![0, 0, 1, 1, 2, 2];
        let skf = StratifiedKFold::new(2, false, None).unwrap();
        let splits = skf.split(&y).unwrap();

        assert_eq!(splits.len(), 2);

        let mut all_test_indices: Vec<usize> = Vec::new();
        for split in &splits {
            all_test_indices.extend(&split.test);
        }
        all_test_indices.sort_unstable();

        assert_eq!(all_test_indices, vec![0, 1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_time_series_split_basic() {
        let tscv = TimeSeriesSplit::new(3, Some(2), 0).unwrap();
        let splits = tscv.split(10).unwrap();

        assert_eq!(splits.len(), 3);

        // Check expanding window property
        for (i, split) in splits.iter().enumerate() {
            assert!(split.train.len() > 0);
            assert_eq!(split.test.len(), 2);
            if i > 0 {
                assert!(split.train.len() > splits[i - 1].train.len());
            }
        }
    }

    #[test]
    fn test_time_series_split_with_gap() {
        let tscv = TimeSeriesSplit::new(2, Some(2), 1).unwrap();
        let splits = tscv.split(10).unwrap();

        for split in &splits {
            // Check gap is maintained
            if !split.train.is_empty() && !split.test.is_empty() {
                let train_max = *split.train.iter().max().unwrap();
                let test_min = *split.test.iter().min().unwrap();
                assert!(test_min > train_max); // Gap of at least 1
            }
        }
    }

    #[test]
    fn test_group_kfold_basic() {
        let groups = vec![0, 0, 1, 1, 2, 2, 3, 3];
        let gkf = GroupKFold::new(2).unwrap();
        let splits = gkf.split(&groups).unwrap();

        assert_eq!(splits.len(), 2);

        // Check group separation
        for split in &splits {
            let train_groups: Vec<usize> = split.train.iter().map(|&i| groups[i]).collect();
            let test_groups: Vec<usize> = split.test.iter().map(|&i| groups[i]).collect();

            // No group should appear in both train and test
            for &test_group in &test_groups {
                assert!(!train_groups.contains(&test_group));
            }
        }
    }

    #[test]
    fn test_leave_one_group_out() {
        let groups = vec![0, 0, 1, 1, 2, 2];
        let logo = LeaveOneGroupOut::new();
        let splits = logo.split(&groups).unwrap();

        assert_eq!(splits.len(), 3); // 3 unique groups

        // Check each split leaves out exactly one group
        for split in &splits {
            let test_groups: Vec<usize> = split.test.iter().map(|&idx| groups[idx]).collect();
            let unique_test_groups: std::collections::HashSet<usize> =
                test_groups.into_iter().collect();
            assert_eq!(unique_test_groups.len(), 1);
        }
    }

    #[test]
    fn test_stratified_kfold_error_too_few_samples() {
        let y = vec![0, 1]; // Only 2 samples
        let skf = StratifiedKFold::new(3, false, None).unwrap();
        assert!(skf.split(&y).is_err());
    }

    #[test]
    fn test_time_series_split_error_insufficient_samples() {
        let tscv = TimeSeriesSplit::new(5, Some(10), 0).unwrap();
        assert!(tscv.split(20).is_err());
    }
}
