//! Shuffle-based cross-validation iterators
//!
//! This module provides cross-validation iterators that use random shuffling
//! to create train/test splits. These methods are particularly useful when
//! you want to control the exact size of training and test sets while
//! maintaining randomness in the splits.

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use scirs2_core::SliceRandomExt;
use std::collections::HashMap;

use crate::cross_validation::CrossValidator;

/// Utility function to generate combinations
fn combinations<T: Clone>(items: &[T], k: usize) -> Vec<Vec<T>> {
    if k == 0 {
        return vec![vec![]];
    }
    if items.is_empty() {
        return vec![];
    }

    let first = &items[0];
    let rest = &items[1..];

    let mut result = Vec::new();

    // Include first element
    for mut combo in combinations(rest, k - 1) {
        combo.insert(0, first.clone());
        result.push(combo);
    }

    // Exclude first element
    result.extend(combinations(rest, k));

    result
}

/// Shuffle Split cross-validator
///
/// Generates random train/test splits independent of the number of iterations
#[derive(Debug, Clone)]
pub struct ShuffleSplit {
    n_splits: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

impl ShuffleSplit {
    /// Create a new ShuffleSplit cross-validator
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: Some(0.1),
            train_size: None,
            random_state: None,
        }
    }

    /// Set the test size as a proportion (0.0 to 1.0) of the dataset
    pub fn test_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "test_size must be between 0.0 and 1.0"
        );
        self.test_size = Some(size);
        self
    }

    /// Set the train size as a proportion (0.0 to 1.0) of the dataset
    pub fn train_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "train_size must be between 0.0 and 1.0"
        );
        self.train_size = Some(size);
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for ShuffleSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let test_size = self.test_size.unwrap_or(0.1);
        let train_size = self.train_size.unwrap_or(1.0 - test_size);

        assert!(
            train_size + test_size <= 1.0,
            "train_size + test_size cannot exceed 1.0"
        );

        let n_test = (n_samples as f64 * test_size).round() as usize;
        let n_train = (n_samples as f64 * train_size).round() as usize;

        assert!(
            n_train + n_test <= n_samples,
            "train_size + test_size results in more samples than available"
        );

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut splits = Vec::new();

        for _ in 0..self.n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let test_indices = indices[..n_test].to_vec();
            let train_indices = indices[n_test..n_test + n_train].to_vec();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Stratified Shuffle Split cross-validator
///
/// Combines stratified sampling with shuffle split for balanced random splits
#[derive(Debug, Clone)]
pub struct StratifiedShuffleSplit {
    n_splits: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

impl StratifiedShuffleSplit {
    /// Create a new StratifiedShuffleSplit cross-validator
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: Some(0.1),
            train_size: None,
            random_state: None,
        }
    }

    /// Set the test size as a proportion (0.0 to 1.0) of the dataset
    pub fn test_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "test_size must be between 0.0 and 1.0"
        );
        self.test_size = Some(size);
        self
    }

    /// Set the train size as a proportion (0.0 to 1.0) of the dataset
    pub fn train_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "train_size must be between 0.0 and 1.0"
        );
        self.train_size = Some(size);
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for StratifiedShuffleSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let y = y.expect("StratifiedShuffleSplit requires y to be provided");
        assert_eq!(
            y.len(),
            n_samples,
            "y must have the same length as n_samples"
        );

        let test_size = self.test_size.unwrap_or(0.1);
        let train_size = self.train_size.unwrap_or(1.0 - test_size);

        assert!(
            train_size + test_size <= 1.0,
            "train_size + test_size cannot exceed 1.0"
        );

        // Group indices by class
        let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &label) in y.iter().enumerate() {
            class_indices.entry(label).or_default().push(idx);
        }

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut splits = Vec::new();

        for _ in 0..self.n_splits {
            let mut train_indices = Vec::new();
            let mut test_indices = Vec::new();

            // Stratified sampling within each class
            for (_class, mut indices) in class_indices.clone() {
                indices.shuffle(&mut rng);

                let n_test_class = ((indices.len() as f64) * test_size).round() as usize;
                let n_train_class = ((indices.len() as f64) * train_size).round() as usize;

                test_indices.extend(&indices[..n_test_class]);
                train_indices.extend(&indices[n_test_class..n_test_class + n_train_class]);
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Bootstrap cross-validator with confidence interval estimation
///
/// Bootstrap cross-validation uses sampling with replacement to create training sets
/// of the same size as the original dataset. The out-of-bag (OOB) samples serve as
/// the test set. This provides bootstrap estimates of model performance with built-in
/// confidence intervals.
#[derive(Debug, Clone)]
pub struct BootstrapCV {
    n_splits: usize,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

impl BootstrapCV {
    /// Create a new Bootstrap cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 1, "n_splits must be at least 1");
        Self {
            n_splits,
            train_size: None, // Use same size as original dataset by default
            random_state: None,
        }
    }

    /// Set the size of the training set as a proportion (0.0 to 1.0) of the dataset
    pub fn train_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "train_size must be between 0.0 and 1.0"
        );
        self.train_size = Some(size);
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for BootstrapCV {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let train_size = match self.train_size {
            Some(frac) => (frac * n_samples as f64).round() as usize,
            None => n_samples, // Bootstrap typically uses same size as original
        };

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut splits = Vec::with_capacity(self.n_splits);

        for _ in 0..self.n_splits {
            // Bootstrap sampling with replacement for training set
            let mut train_indices = Vec::with_capacity(train_size);
            let mut sampled_indices = std::collections::HashSet::new();

            for _ in 0..train_size {
                let idx = rng.gen_range(0..n_samples);
                train_indices.push(idx);
                sampled_indices.insert(idx);
            }

            // Out-of-bag samples for test set
            let test_indices: Vec<usize> = (0..n_samples)
                .filter(|idx| !sampled_indices.contains(idx))
                .collect();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Monte Carlo Cross-Validation with random subsampling
///
/// Monte Carlo CV repeatedly randomly splits the data into training and test sets,
/// unlike K-fold CV which ensures each sample appears exactly once in a test set.
/// This allows for more flexible control over train/test sizes and provides
/// bootstrap-like estimates of model performance.
#[derive(Debug, Clone)]
pub struct MonteCarloCV {
    n_splits: usize,
    test_size: f64,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

impl MonteCarloCV {
    /// Create a new Monte Carlo cross-validator
    pub fn new(n_splits: usize, test_size: f64) -> Self {
        assert!(n_splits >= 1, "n_splits must be at least 1");
        assert!(
            (0.0..=1.0).contains(&test_size),
            "test_size must be between 0.0 and 1.0"
        );
        Self {
            n_splits,
            test_size,
            train_size: None,
            random_state: None,
        }
    }

    /// Set the training set size as a proportion (0.0 to 1.0) of the dataset
    pub fn train_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "train_size must be between 0.0 and 1.0"
        );
        self.train_size = Some(size);
        self
    }

    /// Set the random state for reproducible results
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl CrossValidator for MonteCarloCV {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let train_size = match self.train_size {
            Some(frac) => (frac * n_samples as f64).round() as usize,
            None => n_samples - (self.test_size * n_samples as f64).round() as usize,
        };
        let test_size = (self.test_size * n_samples as f64).round() as usize;

        assert!(
            train_size + test_size <= n_samples,
            "train_size + test_size cannot exceed the number of samples"
        );

        let mut rng = match self.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => {
                use scirs2_core::random::thread_rng;
                StdRng::from_rng(&mut thread_rng())
            }
        };

        let mut splits = Vec::with_capacity(self.n_splits);

        for _ in 0..self.n_splits {
            let mut indices: Vec<usize> = (0..n_samples).collect();
            indices.shuffle(&mut rng);

            let test_indices = indices[..test_size].to_vec();
            let train_indices = indices[test_size..test_size + train_size].to_vec();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::{array, Array1};

    #[test]
    fn test_shuffle_split() {
        let cv = ShuffleSplit::new(3)
            .test_size(0.2)
            .train_size(0.6)
            .random_state(42);

        let splits = cv.split(100, None);
        assert_eq!(splits.len(), 3);

        for (train, test) in splits {
            assert_eq!(test.len(), 20); // 20% of 100
            assert_eq!(train.len(), 60); // 60% of 100

            // Check no overlap
            let train_set: std::collections::HashSet<_> = train.iter().collect();
            let test_set: std::collections::HashSet<_> = test.iter().collect();
            assert!(train_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_shuffle_split_basic() {
        let cv = ShuffleSplit::new(3).test_size(0.2).random_state(42);
        let splits = cv.split(10, None::<&Array1<i32>>);

        assert_eq!(splits.len(), 3);

        for (train, test) in &splits {
            assert_eq!(test.len(), 2); // 20% of 10
            assert_eq!(train.len(), 8); // 80% of 10

            // No overlap between train and test
            for &idx in test {
                assert!(!train.contains(&idx));
            }
        }
    }

    #[test]
    fn test_stratified_shuffle_split() {
        let y = Array1::from_vec(vec![0, 0, 0, 0, 1, 1, 1, 1, 2, 2]);
        let cv = StratifiedShuffleSplit::new(2)
            .test_size(0.3)
            .random_state(42);

        let splits = cv.split(10, Some(&y));
        assert_eq!(splits.len(), 2);

        for (train, test) in splits {
            assert_eq!(test.len(), 3); // 30% of 10
            assert_eq!(train.len(), 7); // 70% of 10

            // Check no overlap
            let train_set: std::collections::HashSet<_> = train.iter().collect();
            let test_set: std::collections::HashSet<_> = test.iter().collect();
            assert!(train_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_stratified_shuffle_split_basic() {
        let y = array![0, 0, 0, 0, 1, 1, 1, 1];
        let cv = StratifiedShuffleSplit::new(2)
            .test_size(0.25)
            .random_state(42);
        let splits = cv.split(8, Some(&y));

        assert_eq!(splits.len(), 2);

        for (train, test) in &splits {
            // Check stratification in test set
            let mut class_counts = HashMap::new();
            for &idx in test {
                *class_counts.entry(y[idx]).or_insert(0) += 1;
            }

            // Both classes should be represented
            assert_eq!(class_counts.len(), 2);

            // Each class should have roughly equal representation
            assert_eq!(class_counts[&0], 1); // 25% of 4 samples of class 0
            assert_eq!(class_counts[&1], 1); // 25% of 4 samples of class 1
        }
    }

    #[test]
    fn test_bootstrap_cv() {
        let cv = BootstrapCV::new(3).random_state(42);

        let splits = cv.split(50, None);
        assert_eq!(splits.len(), 3);

        for (train, test) in splits {
            assert_eq!(train.len(), 50); // Bootstrap uses same size as original
            assert!(test.len() > 0); // Out-of-bag samples
            assert!(test.len() < 50); // Should be less than original
        }
    }

    #[test]
    fn test_monte_carlo_cv() {
        let cv = MonteCarloCV::new(4, 0.25).random_state(42);

        let splits = cv.split(80, None);
        assert_eq!(splits.len(), 4);

        for (train, test) in splits {
            assert_eq!(test.len(), 20); // 25% of 80
            assert_eq!(train.len(), 60); // 75% of 80

            // Check no overlap
            let train_set: std::collections::HashSet<_> = train.iter().collect();
            let test_set: std::collections::HashSet<_> = test.iter().collect();
            assert!(train_set.is_disjoint(&test_set));
        }
    }

    #[test]
    fn test_combinations() {
        let items = vec![1, 2, 3, 4];
        let combos = combinations(&items, 2);
        assert_eq!(combos.len(), 6); // C(4,2) = 6

        let expected = vec![
            vec![1, 2],
            vec![1, 3],
            vec![1, 4],
            vec![2, 3],
            vec![2, 4],
            vec![3, 4],
        ];

        for combo in combos {
            assert_eq!(combo.len(), 2);
            assert!(expected.contains(&combo));
        }
    }
}
