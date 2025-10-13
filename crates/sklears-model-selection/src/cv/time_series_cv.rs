//! Time series cross-validation iterators

use super::CrossValidator;
use scirs2_core::ndarray::Array1;

/// Time Series Split cross-validator with gap and overlapping support
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    n_splits: usize,
    max_train_size: Option<usize>,
    test_size: Option<usize>,
    gap: usize,
    overlap: usize,
}

impl TimeSeriesSplit {
    /// Create a new TimeSeriesSplit cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            max_train_size: None,
            test_size: None,
            gap: 0,
            overlap: 0,
        }
    }

    /// Set the maximum size for a single training set
    pub fn max_train_size(mut self, size: usize) -> Self {
        self.max_train_size = Some(size);
        self
    }

    /// Set the size of the test set
    pub fn test_size(mut self, size: usize) -> Self {
        self.test_size = Some(size);
        self
    }

    /// Set the gap between train and test set
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Set the overlap between consecutive training sets
    /// When overlap > 0, training sets will include overlapping data from previous splits
    pub fn overlap(mut self, overlap: usize) -> Self {
        self.overlap = overlap;
        self
    }
}

impl CrossValidator for TimeSeriesSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let n_splits = self.n_splits;
        let n_folds = n_splits + 1;
        let test_size = self.test_size.unwrap_or_else(|| n_samples / n_folds);

        assert!(
            n_folds * test_size <= n_samples,
            "Too many splits {n_splits} for number of samples {n_samples}"
        );

        let mut splits = Vec::new();
        let test_starts = (0..n_splits)
            .map(|i| n_samples - (n_splits - i) * test_size)
            .collect::<Vec<_>>();

        for (split_idx, &test_start) in test_starts.iter().enumerate() {
            let train_end = test_start - self.gap;
            let test_end = test_start + test_size;

            // Calculate training set with potential overlap
            let mut train_start = 0;
            if self.overlap > 0 && split_idx > 0 {
                // For overlapping, start training from the overlap amount before previous test
                let prev_test_start = test_starts[split_idx - 1];
                train_start = prev_test_start.saturating_sub(self.overlap);
            }

            let mut train_indices: Vec<usize> = (train_start..train_end).collect();

            // Apply max_train_size if set
            if let Some(max_size) = self.max_train_size {
                if train_indices.len() > max_size {
                    let start_idx = train_indices.len() - max_size;
                    train_indices = train_indices[start_idx..].to_vec();
                }
            }

            let test_indices: Vec<usize> = (test_start..test_end).collect();
            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Blocked Time Series Cross-Validation
///
/// This cross-validator provides multiple non-contiguous training blocks
/// for time series data, with gap control to prevent data leakage.
#[derive(Debug, Clone)]
pub struct BlockedTimeSeriesCV {
    n_splits: usize,
    n_blocks: usize,
    gap: usize,
    test_size: Option<usize>,
}

impl BlockedTimeSeriesCV {
    /// Create a new BlockedTimeSeriesCV cross-validator
    pub fn new(n_splits: usize, n_blocks: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        assert!(n_blocks >= 1, "n_blocks must be at least 1");
        Self {
            n_splits,
            n_blocks,
            gap: 0,
            test_size: None,
        }
    }

    /// Set the gap between blocks and test sets
    pub fn gap(mut self, gap: usize) -> Self {
        self.gap = gap;
        self
    }

    /// Set the size of the test set
    pub fn test_size(mut self, size: usize) -> Self {
        self.test_size = Some(size);
        self
    }
}

impl CrossValidator for BlockedTimeSeriesCV {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, _y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let test_size = self.test_size.unwrap_or(n_samples / (self.n_splits + 1));
        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let test_start = n_samples - (self.n_splits - i) * test_size;
            let test_end = test_start + test_size;
            let test_indices: Vec<usize> = (test_start..test_end).collect();

            // Create multiple training blocks before the test set
            let mut train_indices = Vec::new();
            let available_train_space = test_start.saturating_sub(self.gap);
            let block_size = available_train_space / (self.n_blocks + self.n_blocks - 1); // Include gaps between blocks

            for block in 0..self.n_blocks {
                let block_start = block * 2 * block_size; // 2x for block + gap
                let block_end = block_start + block_size;

                if block_end <= available_train_space {
                    train_indices.extend(block_start..block_end);
                }
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }
}

/// Purged Group Time Series Split for financial data
///
/// Advanced time series cross-validation with purging and embargo periods
/// to prevent data leakage in financial modeling.
#[derive(Debug, Clone)]
pub struct PurgedGroupTimeSeriesSplit {
    n_splits: usize,
    max_train_group_size: Option<usize>,
    group_gap: usize,
    purge_length: usize,
    embargo_length: usize,
}

impl PurgedGroupTimeSeriesSplit {
    /// Create a new PurgedGroupTimeSeriesSplit cross-validator
    pub fn new(n_splits: usize) -> Self {
        assert!(n_splits >= 2, "n_splits must be at least 2");
        Self {
            n_splits,
            max_train_group_size: None,
            group_gap: 0,
            purge_length: 0,
            embargo_length: 0,
        }
    }

    /// Set the maximum size for training groups
    pub fn max_train_group_size(mut self, size: usize) -> Self {
        self.max_train_group_size = Some(size);
        self
    }

    /// Set the gap between groups
    pub fn group_gap(mut self, gap: usize) -> Self {
        self.group_gap = gap;
        self
    }

    /// Set the purge length (remove samples before test set)
    pub fn purge_length(mut self, length: usize) -> Self {
        self.purge_length = length;
        self
    }

    /// Set the embargo length (remove samples after test set)
    pub fn embargo_length(mut self, length: usize) -> Self {
        self.embargo_length = length;
        self
    }

    /// Split with group information for financial time series
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        assert_eq!(
            groups.len(),
            n_samples,
            "groups must have same length as n_samples"
        );

        // Get unique groups in order
        let mut group_positions: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_positions.entry(group).or_default().push(idx);
        }

        let mut unique_groups: Vec<i32> = group_positions.keys().cloned().collect();
        unique_groups.sort();

        let groups_per_test = unique_groups.len() / self.n_splits;
        let mut splits = Vec::new();

        for i in 0..self.n_splits {
            let test_group_start = i * groups_per_test;
            let test_group_end = if i == self.n_splits - 1 {
                unique_groups.len()
            } else {
                (i + 1) * groups_per_test
            };

            let test_groups = &unique_groups[test_group_start..test_group_end];

            // Collect test indices
            let mut test_indices = Vec::new();
            for &group in test_groups {
                test_indices.extend(&group_positions[&group]);
            }

            // Collect train indices with purging and embargo
            let mut train_indices = Vec::new();
            for &group in &unique_groups {
                if !test_groups.contains(&group) {
                    let group_indices = &group_positions[&group];

                    // Check if this group should be purged or embargoed
                    let should_include =
                        self.should_include_group(group, test_groups, &unique_groups);

                    if should_include {
                        train_indices.extend(group_indices);
                    }
                }
            }

            splits.push((train_indices, test_indices));
        }

        splits
    }

    fn should_include_group(&self, group: i32, test_groups: &[i32], all_groups: &[i32]) -> bool {
        let group_pos = all_groups.iter().position(|&g| g == group).unwrap();
        let test_start = all_groups
            .iter()
            .position(|&g| g == test_groups[0])
            .unwrap();
        let test_end = all_groups
            .iter()
            .position(|&g| g == test_groups[test_groups.len() - 1])
            .unwrap();

        // Check purge period (before test)
        if group_pos + self.purge_length > test_start && group_pos < test_start {
            return false;
        }

        // Check embargo period (after test)
        if group_pos > test_end && group_pos <= test_end + self.embargo_length {
            return false;
        }

        true
    }
}

impl CrossValidator for PurgedGroupTimeSeriesSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        let groups = y.expect("PurgedGroupTimeSeriesSplit requires group labels in y parameter");
        self.split_with_groups(n_samples, groups)
    }
}
