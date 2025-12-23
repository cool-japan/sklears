//! Group-based cross-validation iterators
//!
//! This module provides cross-validation iterators that work with group labels to ensure
//! no data leakage between groups. These are particularly useful when samples are not
//! independent and can be grouped together (e.g., patients in medical studies, time series
//! from the same entity, etc.).

use scirs2_core::ndarray::Array1;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use scirs2_core::SliceRandomExt;
use std::collections::HashMap;

use crate::cross_validation::CrossValidator;

/// Strategy for defining groups in GroupKFold
#[derive(Debug, Clone)]
pub enum GroupStrategy {
    /// Use provided group labels directly
    Direct,
    /// Use balanced distribution of groups across folds
    Balanced,
    /// Use size-aware distribution (larger groups get separate folds)
    SizeAware { max_group_size: usize },
}

/// Group K-Fold cross-validator with custom group definitions
///
/// Ensures that samples from the same group are not in both training and test sets.
/// Supports custom grouping strategies for advanced use cases.
#[derive(Debug, Clone)]
pub struct GroupKFold {
    n_splits: usize,
    group_strategy: GroupStrategy,
}

impl GroupKFold {
    /// Create a new GroupKFold cross-validator with direct grouping strategy
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            group_strategy: GroupStrategy::Direct,
        }
    }

    /// Create a GroupKFold with balanced group distribution strategy
    pub fn new_balanced(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            group_strategy: GroupStrategy::Balanced,
        }
    }

    /// Create a GroupKFold with size-aware group distribution strategy
    pub fn new_size_aware(n_splits: usize, max_group_size: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            group_strategy: GroupStrategy::SizeAware { max_group_size },
        }
    }

    /// Set the group strategy
    pub fn group_strategy(mut self, strategy: GroupStrategy) -> Self {
        self.group_strategy = strategy;
        self
    }

    /// Split based on groups using the configured strategy to ensure no leakage
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            groups.len(),
            n_samples,
            "groups must have the same length as n_samples"
        );

        // Group indices by group labels
        let mut group_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        let mut unique_groups: Vec<i32> = group_indices.keys().cloned().collect();
        unique_groups.sort();

        assert!(
            unique_groups.len() >= self.n_splits,
            "The number of groups ({}) must be at least equal to the number of splits ({})",
            unique_groups.len(),
            self.n_splits
        );

        match &self.group_strategy {
            GroupStrategy::Direct => self.split_direct(&unique_groups, &group_indices),
            GroupStrategy::Balanced => self.split_balanced(&unique_groups, &group_indices),
            GroupStrategy::SizeAware { max_group_size } => {
                self.split_size_aware(&unique_groups, &group_indices, *max_group_size)
            }
        }
    }

    fn split_direct(
        &self,
        unique_groups: &[i32],
        group_indices: &HashMap<i32, Vec<usize>>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Original distribution strategy - evenly distribute groups
        let n_groups = unique_groups.len();
        let groups_per_fold = n_groups / self.n_splits;
        let n_larger_folds = n_groups % self.n_splits;

        let mut splits = Vec::new();
        let mut current_group_idx = 0;

        for i in 0..self.n_splits {
            let fold_size = if i < n_larger_folds {
                groups_per_fold + 1
            } else {
                groups_per_fold
            };

            let test_groups = &unique_groups[current_group_idx..current_group_idx + fold_size];
            let train_groups: Vec<i32> = unique_groups
                .iter()
                .filter(|&group| !test_groups.contains(group))
                .cloned()
                .collect();

            let mut test_indices = Vec::new();
            for &group in test_groups {
                test_indices.extend(&group_indices[&group]);
            }

            let mut train_indices = Vec::new();
            for &group in &train_groups {
                train_indices.extend(&group_indices[&group]);
            }

            splits.push((train_indices, test_indices));
            current_group_idx += fold_size;
        }

        splits
    }

    fn split_balanced(
        &self,
        unique_groups: &[i32],
        group_indices: &HashMap<i32, Vec<usize>>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Balanced strategy - try to balance the number of samples in each fold
        let mut group_sizes: Vec<(i32, usize)> = unique_groups
            .iter()
            .map(|&group| (group, group_indices[&group].len()))
            .collect();

        // Sort by size to distribute large groups first
        group_sizes.sort_by(|a, b| b.1.cmp(&a.1));

        let mut fold_assignments: Vec<Vec<i32>> = vec![Vec::new(); self.n_splits];
        let mut fold_sizes: Vec<usize> = vec![0; self.n_splits];

        // Assign each group to the fold with the smallest current size
        for (group, size) in group_sizes {
            let min_fold = fold_sizes
                .iter()
                .enumerate()
                .min_by_key(|(_, &size)| size)
                .map(|(idx, _)| idx)
                .unwrap();

            fold_assignments[min_fold].push(group);
            fold_sizes[min_fold] += size;
        }

        let mut splits = Vec::new();
        for test_groups in fold_assignments.iter().take(self.n_splits) {
            let train_groups: Vec<i32> = unique_groups
                .iter()
                .filter(|&group| !test_groups.contains(group))
                .cloned()
                .collect();

            let mut test_indices = Vec::new();
            for &group in test_groups {
                test_indices.extend(&group_indices[&group]);
            }

            let mut train_indices = Vec::new();
            for &group in &train_groups {
                train_indices.extend(&group_indices[&group]);
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }

    fn split_size_aware(
        &self,
        unique_groups: &[i32],
        group_indices: &HashMap<i32, Vec<usize>>,
        max_group_size: usize,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        // Size-aware strategy - large groups get their own folds
        let mut large_groups = Vec::new();
        let mut small_groups = Vec::new();

        for &group in unique_groups {
            if group_indices[&group].len() > max_group_size {
                large_groups.push(group);
            } else {
                small_groups.push(group);
            }
        }

        let mut fold_assignments: Vec<Vec<i32>> = vec![Vec::new(); self.n_splits];
        let mut fold_index = 0;

        // Assign large groups to individual folds
        for group in large_groups {
            if fold_index < self.n_splits {
                fold_assignments[fold_index].push(group);
                fold_index += 1;
            } else {
                // If we have more large groups than folds, add to existing folds
                fold_assignments[fold_index % self.n_splits].push(group);
            }
        }

        // Distribute small groups among remaining folds
        let mut fold_sizes: Vec<usize> = fold_assignments
            .iter()
            .map(|groups| groups.iter().map(|&g| group_indices[&g].len()).sum())
            .collect();

        for group in small_groups {
            let min_fold = fold_sizes
                .iter()
                .enumerate()
                .min_by_key(|(_, &size)| size)
                .map(|(idx, _)| idx)
                .unwrap();

            fold_assignments[min_fold].push(group);
            fold_sizes[min_fold] += group_indices[&group].len();
        }

        let mut splits = Vec::new();
        for test_groups in fold_assignments.iter().take(self.n_splits) {
            let train_groups: Vec<i32> = unique_groups
                .iter()
                .filter(|&group| !test_groups.contains(group))
                .cloned()
                .collect();

            let mut test_indices = Vec::new();
            for &group in test_groups {
                test_indices.extend(&group_indices[&group]);
            }

            let mut train_indices = Vec::new();
            for &group in &train_groups {
                train_indices.extend(&group_indices[&group]);
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl CrossValidator for GroupKFold {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        // For the generic interface, we assume y contains group labels
        let groups = y.expect("GroupKFold requires group labels to be provided in y parameter");
        self.split_with_groups(n_samples, groups)
    }
}

/// Stratified Group K-Fold cross-validator
///
/// This is a variation of GroupKFold that attempts to preserve the percentage of samples
/// for each class while ensuring that the same group is not in both training and testing sets.
#[derive(Debug, Clone)]
pub struct StratifiedGroupKFold {
    n_splits: usize,
    shuffle: bool,
    random_state: Option<u64>,
}

impl StratifiedGroupKFold {
    /// Create a new StratifiedGroupKFold cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            shuffle: false,
            random_state: None,
        }
    }

    /// Set whether to shuffle the groups before splitting
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set the random state for shuffling
    pub fn random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }

    /// Split data into train/test sets based on stratification and groups
    pub fn split_with_groups_and_labels(
        &self,
        n_samples: usize,
        y: &Array1<i32>,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            n_samples,
            groups.len(),
            "n_samples and groups must have the same length"
        );
        assert_eq!(
            n_samples,
            y.len(),
            "n_samples and y must have the same length"
        );

        // Get unique groups and their class distributions
        let mut group_class_counts: HashMap<i32, HashMap<i32, usize>> = HashMap::new();
        let mut group_indices: HashMap<i32, Vec<usize>> = HashMap::new();

        for (idx, (&group, &label)) in groups.iter().zip(y.iter()).enumerate() {
            group_indices.entry(group).or_default().push(idx);

            *group_class_counts
                .entry(group)
                .or_default()
                .entry(label)
                .or_insert(0) += 1;
        }

        let mut unique_groups: Vec<i32> = group_indices.keys().cloned().collect();
        let n_groups = unique_groups.len();

        assert!(
            self.n_splits <= n_groups,
            "Cannot have number of splits {} greater than the number of groups {}",
            self.n_splits,
            n_groups
        );

        // Sort groups by size (descending) for better distribution
        unique_groups.sort_by_key(|&g| {
            let size: usize = group_class_counts[&g].values().sum();
            std::cmp::Reverse(size)
        });

        // Shuffle if requested
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => {
                    use scirs2_core::random::thread_rng;
                    StdRng::from_rng(&mut thread_rng())
                }
            };
            unique_groups.shuffle(&mut rng);
        }

        // Distribute groups to folds to maintain class balance
        let mut fold_groups: Vec<Vec<i32>> = vec![Vec::new(); self.n_splits];
        let mut fold_class_counts: Vec<HashMap<i32, usize>> = vec![HashMap::new(); self.n_splits];

        // Assign groups to folds using a greedy approach
        for group in unique_groups {
            // Find the fold with the smallest total size
            let mut best_fold = 0;
            let mut min_size = usize::MAX;

            for (fold_idx, fold_counts) in fold_class_counts.iter().enumerate() {
                let fold_size: usize = fold_counts.values().sum();
                if fold_size < min_size {
                    min_size = fold_size;
                    best_fold = fold_idx;
                }
            }

            // Add group to the selected fold
            fold_groups[best_fold].push(group);

            // Update fold class counts
            for (&class, &count) in &group_class_counts[&group] {
                *fold_class_counts[best_fold].entry(class).or_insert(0) += count;
            }
        }

        // Generate train/test splits
        let mut splits = Vec::new();

        for test_fold_idx in 0..self.n_splits {
            let mut test_indices = Vec::new();
            let mut train_indices = Vec::new();

            for (fold_idx, groups_in_fold) in fold_groups.iter().enumerate() {
                for &group in groups_in_fold {
                    if fold_idx == test_fold_idx {
                        test_indices.extend(&group_indices[&group]);
                    } else {
                        train_indices.extend(&group_indices[&group]);
                    }
                }
            }

            // Sort indices for consistency
            test_indices.sort_unstable();
            train_indices.sort_unstable();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl CrossValidator for StratifiedGroupKFold {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, _n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        panic!("StratifiedGroupKFold requires both groups and labels. Use split_with_groups_and_labels method instead.");
    }
}

/// Group Shuffle Split cross-validator
///
/// Generates random train/test splits that respect group constraints.
/// Ensures that the same group is not in both training and test sets.
#[derive(Debug, Clone)]
pub struct GroupShuffleSplit {
    n_splits: usize,
    test_size: Option<f64>,
    train_size: Option<f64>,
    random_state: Option<u64>,
}

impl GroupShuffleSplit {
    /// Create a new GroupShuffleSplit cross-validator
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            test_size: Some(0.2),
            train_size: None,
            random_state: None,
        }
    }

    /// Set the test size as a proportion (0.0 to 1.0) of the groups
    pub fn test_size(mut self, size: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&size),
            "test_size must be between 0.0 and 1.0"
        );
        self.test_size = Some(size);
        self
    }

    /// Set the train size as a proportion (0.0 to 1.0) of the groups
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

    /// Split data based on groups
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            groups.len(),
            n_samples,
            "groups must have the same length as n_samples"
        );

        // Group indices by group labels
        let mut group_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        let unique_groups: Vec<i32> = group_indices.keys().cloned().collect();
        let n_groups = unique_groups.len();

        let test_size = self.test_size.unwrap_or(0.2);
        let train_size = self.train_size.unwrap_or(1.0 - test_size);

        assert!(
            train_size + test_size <= 1.0,
            "train_size + test_size cannot exceed 1.0"
        );

        let n_test_groups = ((n_groups as f64) * test_size).round() as usize;
        let n_train_groups = ((n_groups as f64) * train_size).round() as usize;

        assert!(
            n_train_groups + n_test_groups <= n_groups,
            "train_size + test_size results in more groups than available"
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
            let mut shuffled_groups = unique_groups.clone();
            shuffled_groups.shuffle(&mut rng);

            let test_groups = &shuffled_groups[..n_test_groups];
            let train_groups = &shuffled_groups[n_test_groups..n_test_groups + n_train_groups];

            let mut test_indices = Vec::new();
            for &group in test_groups {
                test_indices.extend(&group_indices[&group]);
            }

            let mut train_indices = Vec::new();
            for &group in train_groups {
                train_indices.extend(&group_indices[&group]);
            }

            // Sort for consistency
            test_indices.sort_unstable();
            train_indices.sort_unstable();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl CrossValidator for GroupShuffleSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        // For the generic interface, we assume y contains group labels
        let groups =
            y.expect("GroupShuffleSplit requires group labels to be provided in y parameter");
        self.split_with_groups(n_samples, groups)
    }
}

/// Leave One Group Out cross-validator
///
/// Provides train/test splits where each split leaves out one unique group.
#[derive(Debug, Clone)]
pub struct LeaveOneGroupOut;

impl Default for LeaveOneGroupOut {
    fn default() -> Self {
        Self::new()
    }
}

impl LeaveOneGroupOut {
    /// Create a new LeaveOneGroupOut cross-validator
    pub fn new() -> Self {
        LeaveOneGroupOut
    }

    /// Split data based on groups
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            groups.len(),
            n_samples,
            "groups must have the same length as n_samples"
        );

        // Group indices by group labels
        let mut group_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        let mut unique_groups: Vec<i32> = group_indices.keys().cloned().collect();
        unique_groups.sort();

        let mut splits = Vec::new();

        // Leave out each group one at a time
        for &test_group in &unique_groups {
            let test_indices = group_indices[&test_group].clone();
            let mut train_indices = Vec::new();

            for &train_group in &unique_groups {
                if train_group != test_group {
                    train_indices.extend(&group_indices[&train_group]);
                }
            }

            // Sort for consistency
            train_indices.sort_unstable();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl CrossValidator for LeaveOneGroupOut {
    fn n_splits(&self) -> usize {
        // This is dynamic based on the number of unique groups
        0 // Will be determined during split
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        // For the generic interface, we assume y contains group labels
        let groups =
            y.expect("LeaveOneGroupOut requires group labels to be provided in y parameter");
        self.split_with_groups(n_samples, groups)
    }
}

/// Leave P Groups Out cross-validator
///
/// Provides train/test splits where each split leaves out P groups.
#[derive(Debug, Clone)]
pub struct LeavePGroupsOut {
    p: usize,
}

impl LeavePGroupsOut {
    /// Create a new LeavePGroupsOut cross-validator
    pub fn new(p: usize) -> Self {
        assert!(p >= 1, "p must be at least 1");
        Self { p }
    }

    /// Split data based on groups
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            groups.len(),
            n_samples,
            "groups must have the same length as n_samples"
        );

        // Group indices by group labels
        let mut group_indices: HashMap<i32, Vec<usize>> = HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_indices.entry(group).or_default().push(idx);
        }

        let unique_groups: Vec<i32> = group_indices.keys().cloned().collect();
        let n_groups = unique_groups.len();

        assert!(
            self.p <= n_groups,
            "p ({}) cannot be greater than number of groups ({})",
            self.p,
            n_groups
        );

        let mut splits = Vec::new();

        // Generate all combinations of p groups for test sets
        let group_combinations = combinations(&unique_groups, self.p);

        for test_groups in group_combinations {
            let mut test_indices = Vec::new();
            for &group in &test_groups {
                test_indices.extend(&group_indices[&group]);
            }

            let mut train_indices = Vec::new();
            for &group in &unique_groups {
                if !test_groups.contains(&group) {
                    train_indices.extend(&group_indices[&group]);
                }
            }

            // Sort for consistency
            test_indices.sort_unstable();
            train_indices.sort_unstable();

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

impl CrossValidator for LeavePGroupsOut {
    fn n_splits(&self) -> usize {
        // This is dynamic based on the number of groups and p
        0 // Will be determined during split
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        // For the generic interface, we assume y contains group labels
        let groups =
            y.expect("LeavePGroupsOut requires group labels to be provided in y parameter");
        self.split_with_groups(n_samples, groups)
    }
}

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

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_group_kfold() {
        let groups = array![0, 0, 1, 1, 2, 2];
        let cv = GroupKFold::new(2);
        let splits = cv.split_with_groups(6, &groups);

        assert_eq!(splits.len(), 2);

        for (train, test) in &splits {
            // Check that groups don't overlap between train and test
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();

            // No group should appear in both train and test
            for &test_group in &test_groups {
                assert!(!train_groups.contains(&test_group));
            }
        }
    }

    #[test]
    fn test_group_kfold_custom_strategies() {
        // Test balanced strategy
        let groups = array![0, 0, 1, 1, 1, 2, 2, 2, 2, 3, 3, 4]; // Varying group sizes
        let cv_balanced = GroupKFold::new_balanced(3);
        let splits = cv_balanced.split_with_groups(12, &groups);

        assert_eq!(splits.len(), 3);

        // Check group separation
        for (train, test) in &splits {
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();

            for &test_group in &test_groups {
                assert!(!train_groups.contains(&test_group));
            }
        }

        // Test size-aware strategy
        let cv_size_aware = GroupKFold::new_size_aware(3, 3); // max_group_size = 3
        let splits = cv_size_aware.split_with_groups(12, &groups);

        assert_eq!(splits.len(), 3);

        // Check group separation
        for (train, test) in &splits {
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();

            for &test_group in &test_groups {
                assert!(!train_groups.contains(&test_group));
            }
        }

        // Test custom strategy assignment
        let cv_custom = GroupKFold::new(3).group_strategy(GroupStrategy::Balanced);
        let splits = cv_custom.split_with_groups(12, &groups);

        assert_eq!(splits.len(), 3);
    }

    #[test]
    fn test_stratified_group_kfold() {
        let groups = array![1, 1, 2, 2, 3, 3, 4, 4];
        let y = array![0, 0, 1, 1, 0, 1, 0, 1];
        let cv = StratifiedGroupKFold::new(2);
        let splits = cv.split_with_groups_and_labels(8, &y, &groups);

        assert_eq!(splits.len(), 2);

        // Check that groups don't overlap between train and test
        for (train_idx, test_idx) in &splits {
            let train_groups: std::collections::HashSet<i32> =
                train_idx.iter().map(|&i| groups[i]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test_idx.iter().map(|&i| groups[i]).collect();

            assert!(train_groups.is_disjoint(&test_groups));

            // Check that both classes are represented in training set
            let train_class_0 = train_idx.iter().filter(|&&i| y[i] == 0).count();
            let train_class_1 = train_idx.iter().filter(|&&i| y[i] == 1).count();

            assert!(train_class_0 > 0);
            assert!(train_class_1 > 0);
        }
    }

    #[test]
    fn test_group_shuffle_split() {
        let groups = array![0, 0, 1, 1, 2, 2, 3, 3];
        let cv = GroupShuffleSplit::new(3).test_size(0.25).random_state(42);
        let splits = cv.split_with_groups(8, &groups);

        assert_eq!(splits.len(), 3);

        for (train, test) in &splits {
            // Check that groups don't overlap between train and test
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();

            // No group should appear in both train and test
            assert!(train_groups.is_disjoint(&test_groups));

            // Check we have the expected number of groups in test
            assert_eq!(test_groups.len(), 1); // 25% of 4 groups
        }
    }

    #[test]
    fn test_leave_one_group_out() {
        let groups = array![0, 0, 1, 1, 2, 2];
        let cv = LeaveOneGroupOut::new();
        let splits = cv.split_with_groups(6, &groups);

        // Should have as many splits as unique groups
        assert_eq!(splits.len(), 3);

        for (train, test) in splits.iter() {
            // Each test set should contain samples from exactly one group
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();
            assert_eq!(test_groups.len(), 1);

            // Train set should contain samples from all other groups
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            assert_eq!(train_groups.len(), 2);

            // No overlap between train and test groups
            assert!(train_groups.is_disjoint(&test_groups));
        }
    }

    #[test]
    fn test_leave_p_groups_out() {
        let groups = array![0, 0, 1, 1, 2, 2, 3, 3];
        let cv = LeavePGroupsOut::new(2);
        let splits = cv.split_with_groups(8, &groups);

        // C(4,2) = 6 combinations
        assert_eq!(splits.len(), 6);

        for (train, test) in &splits {
            // Each test set should contain samples from exactly 2 groups
            let test_groups: std::collections::HashSet<i32> =
                test.iter().map(|&idx| groups[idx]).collect();
            assert_eq!(test_groups.len(), 2);

            // Train set should contain samples from the other 2 groups
            let train_groups: std::collections::HashSet<i32> =
                train.iter().map(|&idx| groups[idx]).collect();
            assert_eq!(train_groups.len(), 2);

            // No overlap between train and test groups
            assert!(train_groups.is_disjoint(&test_groups));
        }
    }
}
