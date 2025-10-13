//! Repeated cross-validation methods for robust model evaluation
//!
//! This module provides repeated cross-validation methods that run multiple rounds
//! of K-fold or stratified K-fold with different random states to obtain more robust
//! estimates of model performance. These methods are particularly useful when you
//! want to reduce the variance in your cross-validation estimates.

use scirs2_core::ndarray::Array1;
use scirs2_core::SliceRandomExt;

use crate::cross_validation::{CrossValidator, KFold, StratifiedKFold};

/// Repeated K-Fold cross-validator
///
/// Repeats K-Fold n times with different randomization in each repetition.
/// This provides more robust estimates by reducing the variance that comes
/// from a single random split of the data.
///
/// # Examples
///
/// ```
/// use sklears_model_selection::{RepeatedKFold, CrossValidator};
///
/// let cv = RepeatedKFold::new(5, 3)  // 5-fold repeated 3 times
///     .random_state(42);
/// let splits = cv.split(100, None);
/// assert_eq!(splits.len(), 15);  // 5 * 3 = 15 splits
/// ```
#[derive(Debug, Clone)]
pub struct RepeatedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

impl RepeatedKFold {
    /// Create a new RepeatedKFold cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds per repetition (must be >= 2)
    /// * `n_repeats` - Number of repetitions (must be >= 1)
    ///
    /// # Panics
    /// Panics if `n_splits` < 2 or `n_repeats` < 1
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        assert!(n_repeats >= 1, "n_repeats must be at least 1");
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    ///
    /// # Arguments
    /// * `seed` - Random seed value
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the number of splits per repetition
    pub fn n_splits_per_repeat(&self) -> usize {
        self.n_splits
    }

    /// Get the number of repetitions
    pub fn n_repeats(&self) -> usize {
        self.n_repeats
    }
}

impl CrossValidator for RepeatedKFold {
    fn n_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut all_splits = Vec::new();

        let base_seed = self.random_state.unwrap_or(42);

        for repeat in 0..self.n_repeats {
            // Create a KFold with shuffling and a different seed for each repeat
            let kfold = KFold::new(self.n_splits)
                .shuffle(true)
                .random_state(base_seed + repeat as u64);

            let splits = kfold.split(n_samples, None);
            all_splits.extend(splits);
        }

        all_splits
    }
}

/// Repeated Stratified K-Fold cross-validator
///
/// Repeats Stratified K-Fold n times with different randomization in each repetition.
/// This provides more robust estimates while maintaining the class distribution
/// in each fold. This is particularly useful for imbalanced datasets where you
/// want both stratification and robust estimates.
///
/// # Examples
///
/// ```
/// use sklears_model_selection::{RepeatedStratifiedKFold, CrossValidator};
/// use scirs2_core::ndarray::array;
///
/// let cv = RepeatedStratifiedKFold::new(2, 2)  // 2-fold repeated 2 times
///     .random_state(42);
/// let y = array![0, 0, 1, 1, 2, 2];
/// let splits = cv.split(6, Some(&y));
/// assert_eq!(splits.len(), 4);  // 2 * 2 = 4 splits
/// ```
#[derive(Debug, Clone)]
pub struct RepeatedStratifiedKFold {
    n_splits: usize,
    n_repeats: usize,
    random_state: Option<u64>,
}

impl RepeatedStratifiedKFold {
    /// Create a new RepeatedStratifiedKFold cross-validator
    ///
    /// # Arguments
    /// * `n_splits` - Number of folds per repetition (must be >= 2)
    /// * `n_repeats` - Number of repetitions (must be >= 1)
    ///
    /// # Panics
    /// Panics if `n_splits` < 2 or `n_repeats` < 1
    pub fn new(n_splits: usize, n_repeats: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        assert!(n_repeats >= 1, "n_repeats must be at least 1");
        Self {
            n_splits,
            n_repeats,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    ///
    /// # Arguments
    /// * `seed` - Random seed value
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Get the number of splits per repetition
    pub fn n_splits_per_repeat(&self) -> usize {
        self.n_splits
    }

    /// Get the number of repetitions
    pub fn n_repeats(&self) -> usize {
        self.n_repeats
    }
}

impl CrossValidator for RepeatedStratifiedKFold {
    fn n_splits(&self) -> usize {
        self.n_splits * self.n_repeats
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let y = y.expect("RepeatedStratifiedKFold requires y to be provided");
        let mut all_splits = Vec::new();

        let base_seed = self.random_state.unwrap_or(42);

        for repeat in 0..self.n_repeats {
            // Create a StratifiedKFold with shuffling and a different seed for each repeat
            let stratified_kfold = StratifiedKFold::new(self.n_splits)
                .shuffle(true)
                .random_state(base_seed + repeat as u64);

            let splits = stratified_kfold.split(n_samples, Some(y));
            all_splits.extend(splits);
        }

        all_splits
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;
    use std::collections::HashMap;

    #[test]
    fn test_repeated_kfold_basic() {
        let cv = RepeatedKFold::new(3, 2).random_state(42);
        let splits = cv.split(9, None);

        // Should have n_splits * n_repeats splits
        assert_eq!(splits.len(), 6);
        assert_eq!(cv.n_splits(), 6);
        assert_eq!(cv.n_splits_per_repeat(), 3);
        assert_eq!(cv.n_repeats(), 2);

        // Check that each sample appears in test sets the expected number of times
        let mut test_count = vec![0; 9];
        for (_, test) in &splits {
            for &idx in test {
                test_count[idx] += 1;
            }
        }

        // Each sample should appear in test sets exactly n_repeats times
        for count in test_count {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_repeated_kfold_no_overlap() {
        let cv = RepeatedKFold::new(3, 2).random_state(42);
        let splits = cv.split(9, None);

        // Verify no overlap between train and test in each split
        for (train, test) in &splits {
            for &test_idx in test {
                assert!(!train.contains(&test_idx));
            }
        }
    }

    #[test]
    fn test_repeated_kfold_different_seeds() {
        let cv1 = RepeatedKFold::new(3, 2).random_state(42);
        let cv2 = RepeatedKFold::new(3, 2).random_state(123);

        let splits1 = cv1.split(9, None);
        let splits2 = cv2.split(9, None);

        // Different seeds should produce different splits
        assert_ne!(splits1, splits2);
    }

    #[test]
    fn test_repeated_stratified_kfold_basic() {
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cv = RepeatedStratifiedKFold::new(3, 2).random_state(42);
        let splits = cv.split(9, Some(&y));

        // Should have n_splits * n_repeats splits
        assert_eq!(splits.len(), 6);
        assert_eq!(cv.n_splits(), 6);
        assert_eq!(cv.n_splits_per_repeat(), 3);
        assert_eq!(cv.n_repeats(), 2);

        // Check stratification in each split
        for (_, test) in &splits {
            let mut class_counts = HashMap::new();
            for &idx in test {
                *class_counts.entry(y[idx]).or_insert(0) += 1;
            }

            // Each class should be represented
            assert_eq!(class_counts.len(), 3);
            // Each class should have exactly 1 sample in test set
            for count in class_counts.values() {
                assert_eq!(*count, 1);
            }
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_requires_y() {
        let cv = RepeatedStratifiedKFold::new(3, 2);

        // Should panic when y is not provided
        let result = std::panic::catch_unwind(|| cv.split(9, None));
        assert!(result.is_err());
    }

    #[test]
    fn test_repeated_stratified_kfold_class_distribution() {
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cv = RepeatedStratifiedKFold::new(3, 2).random_state(42);
        let splits = cv.split(9, Some(&y));

        // Each sample should appear in test sets exactly n_repeats times
        let mut test_count = vec![0; 9];
        for (_, test) in &splits {
            for &idx in test {
                test_count[idx] += 1;
            }
        }

        for count in test_count {
            assert_eq!(count, 2);
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_different_seeds() {
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cv1 = RepeatedStratifiedKFold::new(3, 2).random_state(42);
        let cv2 = RepeatedStratifiedKFold::new(3, 2).random_state(123);

        let splits1 = cv1.split(9, Some(&y));
        let splits2 = cv2.split(9, Some(&y));

        // Different seeds should produce different splits
        assert_ne!(splits1, splits2);
    }

    #[test]
    #[should_panic(expected = "n_splits must be at least 2")]
    fn test_repeated_kfold_invalid_n_splits() {
        RepeatedKFold::new(1, 2);
    }

    #[test]
    #[should_panic(expected = "n_repeats must be at least 1")]
    fn test_repeated_kfold_invalid_n_repeats() {
        RepeatedKFold::new(3, 0);
    }

    #[test]
    #[should_panic(expected = "n_splits must be at least 2")]
    fn test_repeated_stratified_kfold_invalid_n_splits() {
        RepeatedStratifiedKFold::new(1, 2);
    }

    #[test]
    #[should_panic(expected = "n_repeats must be at least 1")]
    fn test_repeated_stratified_kfold_invalid_n_repeats() {
        RepeatedStratifiedKFold::new(3, 0);
    }

    #[test]
    fn test_repeated_kfold_single_repeat() {
        let cv = RepeatedKFold::new(3, 1).random_state(42);
        let splits = cv.split(9, None);

        // Should have exactly n_splits splits
        assert_eq!(splits.len(), 3);

        // Each sample should appear exactly once in test sets
        let mut test_count = vec![0; 9];
        for (_, test) in &splits {
            for &idx in test {
                test_count[idx] += 1;
            }
        }

        for count in test_count {
            assert_eq!(count, 1);
        }
    }

    #[test]
    fn test_repeated_stratified_kfold_single_repeat() {
        let y = array![0, 0, 0, 1, 1, 1, 2, 2, 2];
        let cv = RepeatedStratifiedKFold::new(3, 1).random_state(42);
        let splits = cv.split(9, Some(&y));

        // Should have exactly n_splits splits
        assert_eq!(splits.len(), 3);

        // Each sample should appear exactly once in test sets
        let mut test_count = vec![0; 9];
        for (_, test) in &splits {
            for &idx in test {
                test_count[idx] += 1;
            }
        }

        for count in test_count {
            assert_eq!(count, 1);
        }
    }
}
