//! Basic cross-validation iterators

use super::CrossValidator;
use scirs2_core::ndarray::Array1;
use scirs2_core::random::{rngs::StdRng, SeedableRng};
use scirs2_core::SliceRandomExt;
use std::collections::HashMap;

/// K-Fold cross-validation iterator
#[derive(Debug, Clone)]
pub struct KFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl KFold {
    /// Create a new KFold cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
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

impl CrossValidator for KFold {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(
            self.n_splits <= n_samples,
            "Cannot have number of splits {} greater than the number of samples {}",
            self.n_splits,
            n_samples
        );

        // Create indices
        let mut indices: Vec<usize> = (0..n_samples).collect();

        // Shuffle if requested
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    use scirs2_core::random::thread_rng;
                    StdRng::from_rng(&mut thread_rng())
                }
            };
            indices.shuffle(&mut rng);
        }

        // Generate train/test splits
        let mut splits = Vec::new();
        let fold_sizes = self.calculate_fold_sizes(n_samples);
        let mut current = 0;

        for fold_size in fold_sizes.iter().take(self.n_splits) {
            let test_start = current;
            let test_end = current + *fold_size;

            let test_indices: Vec<usize> = indices[test_start..test_end].to_vec();
            let train_indices: Vec<usize> = indices[..test_start]
                .iter()
                .chain(indices[test_end..].iter())
                .cloned()
                .collect();

            splits.push((train_indices, test_indices));
            current = test_end;
        }

        splits
    }
}

/// Stratified K-Fold cross-validation iterator
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new StratifiedKFold cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
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

impl CrossValidator for StratifiedKFold {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let y = y.expect("StratifiedKFold requires y to be provided");
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

        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        // Check we have enough samples in each class
        for indices in class_indices.values() {
            assert!(
                indices.len() >= self.n_splits,
                "The least populated class has only {} members, which is less than n_splits={}",
                indices.len(),
                self.n_splits
            );
        }

        // Shuffle within each class if requested
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    use scirs2_core::random::thread_rng;
                    StdRng::from_rng(&mut thread_rng())
                }
            };
            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Create stratified folds
        let mut splits = vec![(Vec::new(), Vec::new()); self.n_splits];

        for (_class, indices) in class_indices {
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

/// Leave-One-Out cross-validator
#[derive(Debug, Clone)]
pub struct LeaveOneOut;

impl Default for LeaveOneOut {
    fn default() -> Self {
        Self::new()
    }
}

impl LeaveOneOut {
    /// Create a new LeaveOneOut cross-validator
    pub fn new() -> Self {
        LeaveOneOut
    }
}

impl CrossValidator for LeaveOneOut {
    fn n_splits(&self) -> usize {
        // This is dynamic based on the number of samples
        0 // Will be determined during split
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let test_indices = vec![i];
            let train_indices: Vec<usize> = (0..i).chain(i + 1..n_samples).collect();
            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Leave-P-Out cross-validator
///
/// Provides test sets by taking all possible combinations of p samples
#[derive(Debug, Clone)]
pub struct LeavePOut {
    p: usize,
}

impl LeavePOut {
    /// Create a new LeavePOut cross-validator
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "p must be at least 1");
        Self { p }
    }

    /// Get the value of p
    pub fn p(&self) -> usize {
        self.p
    }
}

impl CrossValidator for LeavePOut {
    fn n_splits(&self) -> usize {
        // This is dynamic based on the number of samples
        0 // Will be determined during split
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert!(
            self.p <= n_samples,
            "p ({}) cannot be greater than the number of samples ({})",
            self.p,
            n_samples
        );

        let mut splits = Vec::new();
        let all_indices: Vec<usize> = (0..n_samples).collect();

        // Generate all combinations of p indices
        for test_indices in combinations(&all_indices, self.p) {
            let test_set: std::collections::HashSet<usize> = test_indices.iter().cloned().collect();
            let train_indices: Vec<usize> = all_indices
                .iter()
                .cloned()
                .filter(|&i| !test_set.contains(&i))
                .collect();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Generate all combinations of k elements from a vector
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > items.len() {
        return vec![];
    }
    if k == items.len() {
        return vec![items.to_vec()];
    }

    let mut result = Vec::new();

    // Include first item
    let with_first = combinations(&items[1..], k - 1);
    for mut combo in with_first {
        combo.insert(0, items[0].clone());
        result.push(combo);
    }

    // Exclude first item
    let without_first = combinations(&items[1..], k);
    result.extend(without_first);

    result
}
