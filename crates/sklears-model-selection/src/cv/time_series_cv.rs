//! Time series cross-validation iterators

use super::CrossValidator;
use scirs2_core::ndarray::Array1;
use sklears_core::error::{Result, SklearsError};

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
/// Walk-forward cross-validation over *groups* (a trading day, a bar, an event
/// block — anything whose label horizon can straddle several samples), with
/// gap / purge / embargo controls to keep overlapping labels out of the
/// training set.
///
/// # Walk-forward by default
/// Groups are ordered by their sorted label and that order *is* time. The
/// `n_splits` test windows are the last `n_splits` blocks of
/// `n_groups / (n_splits + 1)` groups each, which reserves the head of the
/// series as training data for the first fold and makes every fold train
/// strictly in the past of its own test window:
///
/// ```text
/// groups:   0  1  2 | 3  4  5 | 6  7  8 | 9 10 11
/// fold 0:  [ train ] [ test  ]
/// fold 1:  [   train         ] [ test  ]
/// fold 2:  [       train              ] [ test  ]
/// ```
///
/// Before this fix a fold's training set was *every* non-test group, including
/// the groups after the test window — so fold 0 trained entirely on data from
/// the future of its test set. That leak is gone;
/// [`Self::allow_future_train_groups`] re-enables the old non-causal layout
/// explicitly for callers who genuinely want K-fold-style grouped validation.
///
/// # Gap, purge and embargo
/// * `group_gap` and `purge_length` both drop groups immediately *before* the
///   test window and stack: the training window ends
///   `group_gap + purge_length` groups before the first test group.
/// * `embargo_length` drops groups immediately *after* the test window. In the
///   default walk-forward mode nothing is trained on after the test window, so
///   the embargo only has an effect together with
///   [`Self::allow_future_train_groups`].
/// * `max_train_group_size` caps how many groups of history the (past)
///   training window keeps.
#[derive(Debug, Clone)]
pub struct PurgedGroupTimeSeriesSplit {
    n_splits: usize,
    max_train_group_size: Option<usize>,
    group_gap: usize,
    purge_length: usize,
    embargo_length: usize,
    allow_future_train_groups: bool,
}

impl PurgedGroupTimeSeriesSplit {
    /// Create a new PurgedGroupTimeSeriesSplit cross-validator
    ///
    /// # Panics
    /// Panics if `n_splits < 2`. Use [`Self::try_new`] for a fallible
    /// constructor. (The panic is retained so the existing signature keeps
    /// working for downstream callers.)
    pub fn new(n_splits: usize) -> Self {
        match Self::try_new(n_splits) {
            Ok(cv) => cv,
            Err(err) => panic!("{err}"),
        }
    }

    /// Fallible form of [`Self::new`]: `n_splits` must be at least 2.
    pub fn try_new(n_splits: usize) -> Result<Self> {
        if n_splits < 2 {
            return Err(SklearsError::InvalidParameter {
                name: "n_splits".to_string(),
                reason: format!("n_splits must be at least 2, got {n_splits}"),
            });
        }
        Ok(Self {
            n_splits,
            max_train_group_size: None,
            group_gap: 0,
            purge_length: 0,
            embargo_length: 0,
            allow_future_train_groups: false,
        })
    }

    /// Cap the number of *past* groups kept in each training set (a rolling
    /// window instead of an expanding one)
    pub fn max_train_group_size(mut self, size: usize) -> Self {
        self.max_train_group_size = Some(size);
        self
    }

    /// Set the gap, in groups, between the end of the training window and the
    /// start of the test window
    pub fn group_gap(mut self, gap: usize) -> Self {
        self.group_gap = gap;
        self
    }

    /// Set the purge length: this many groups immediately before the test
    /// window are dropped from training (their labels may overlap the test
    /// period). Stacks with [`Self::group_gap`].
    pub fn purge_length(mut self, length: usize) -> Self {
        self.purge_length = length;
        self
    }

    /// Set the embargo length: this many groups immediately after the test
    /// window are dropped from training.
    ///
    /// Only has an effect together with [`Self::allow_future_train_groups`] —
    /// the default walk-forward mode never trains after the test window.
    pub fn embargo_length(mut self, length: usize) -> Self {
        self.embargo_length = length;
        self
    }

    /// Opt in to the legacy, non-causal behaviour: test windows tile *all*
    /// groups (as in K-fold) and each fold also trains on the groups after its
    /// test window, beyond [`Self::embargo_length`].
    ///
    /// Defaults to `false` (leak-free walk-forward). Only enable this for data
    /// where training on the future is genuinely acceptable — for forecasting
    /// or backtesting it is a look-ahead leak.
    pub fn allow_future_train_groups(mut self, allow: bool) -> Self {
        self.allow_future_train_groups = allow;
        self
    }

    /// Split with group information for financial time series
    ///
    /// # Panics
    /// Panics if `groups.len() != n_samples`, or if there are too few distinct
    /// groups for `n_splits`. Use [`Self::try_split_with_groups`] to receive
    /// those as errors instead. (The panic is retained so the existing
    /// signature keeps working for downstream callers.)
    pub fn split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        match self.try_split_with_groups(n_samples, groups) {
            Ok(splits) => splits,
            Err(err) => panic!("{err}"),
        }
    }

    /// Fallible form of [`Self::split_with_groups`]
    pub fn try_split_with_groups(
        &self,
        n_samples: usize,
        groups: &Array1<i32>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if groups.len() != n_samples {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("n_samples={n_samples}"),
                actual: format!("groups.len()={}", groups.len()),
            });
        }

        // Sample positions per group; the sorted group label order is time order.
        let mut group_positions: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();
        for (idx, &group) in groups.iter().enumerate() {
            group_positions.entry(group).or_default().push(idx);
        }
        let mut unique_groups: Vec<i32> = group_positions.keys().copied().collect();
        unique_groups.sort_unstable();
        let n_groups = unique_groups.len();

        // Walk-forward reserves the first block as training data for fold 0, so
        // it needs one more block than it has folds; the non-causal layout tiles
        // every group with a test window instead.
        let blocks_needed = if self.allow_future_train_groups {
            self.n_splits
        } else {
            self.n_splits + 1
        };
        let groups_per_test = n_groups / blocks_needed;
        if groups_per_test == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "n_splits".to_string(),
                reason: format!(
                    "{n_groups} distinct groups are too few for {} splits \
                     (at least {blocks_needed} groups are required)",
                    self.n_splits
                ),
            });
        }

        let collect_groups = |positions: &[i32]| -> Vec<usize> {
            let mut indices = Vec::new();
            for group in positions {
                indices.extend(group_positions[group].iter().copied());
            }
            indices
        };

        let mut splits = Vec::with_capacity(self.n_splits);
        for fold in 0..self.n_splits {
            let (test_start, test_end) = if self.allow_future_train_groups {
                // Contiguous blocks covering every group; the last one absorbs
                // the remainder.
                let start = fold * groups_per_test;
                let end = if fold == self.n_splits - 1 {
                    n_groups
                } else {
                    (fold + 1) * groups_per_test
                };
                (start, end)
            } else {
                let start = n_groups - (self.n_splits - fold) * groups_per_test;
                (start, start + groups_per_test)
            };

            let test_indices = collect_groups(&unique_groups[test_start..test_end]);

            // Past side: everything strictly before the test window, minus the
            // gap and the purge horizon, optionally windowed.
            let past_end = test_start.saturating_sub(self.group_gap + self.purge_length);
            let past_start = match self.max_train_group_size {
                Some(max_groups) => past_end.saturating_sub(max_groups),
                None => 0,
            };
            let mut train_indices = collect_groups(&unique_groups[past_start..past_end]);

            // Future side: only when explicitly opted in, and only past the
            // embargo that follows the test window.
            if self.allow_future_train_groups {
                let future_start = test_end.saturating_add(self.embargo_length).min(n_groups);
                train_indices.extend(collect_groups(&unique_groups[future_start..]));
            }

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }
}

impl CrossValidator for PurgedGroupTimeSeriesSplit {
    fn n_splits(&self) -> usize {
        self.n_splits
    }

    /// # Panics
    /// The [`CrossValidator`] signature cannot report failures, so this panics
    /// if `y` is `None` (group labels are mandatory here) or if
    /// [`Self::try_split_with_groups`] fails. Call that method directly to
    /// handle either case as an error.
    fn split(&self, n_samples: usize, y: Option<&Array1<i32>>) -> Vec<(Vec<usize>, Vec<usize>)> {
        match y {
            Some(groups) => self.split_with_groups(n_samples, groups),
            None => panic!("PurgedGroupTimeSeriesSplit requires group labels in the y parameter"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `n_groups` groups of `per_group` consecutive samples, laid out in time
    /// order: group `g` owns indices `g * per_group .. (g + 1) * per_group`.
    fn make_groups(n_groups: usize, per_group: usize) -> Array1<i32> {
        let mut values = Vec::with_capacity(n_groups * per_group);
        for group in 0..n_groups {
            for _ in 0..per_group {
                values.push(group as i32);
            }
        }
        Array1::from_vec(values)
    }

    /// Group label of every sample index in `indices`.
    fn groups_of(groups: &Array1<i32>, indices: &[usize]) -> Vec<i32> {
        indices.iter().map(|&i| groups[i]).collect()
    }

    /// The regression test for the walk-forward leak: in every fold, every
    /// training group must be strictly earlier than every test group, and no
    /// fold may end up with an empty training set.
    #[test]
    fn test_purged_group_split_is_walk_forward() {
        for n_splits in 2..=4usize {
            for n_groups in [12usize, 13, 20] {
                let groups = make_groups(n_groups, 3);
                let n_samples = groups.len();
                let splits =
                    PurgedGroupTimeSeriesSplit::new(n_splits).split_with_groups(n_samples, &groups);
                assert_eq!(splits.len(), n_splits);

                for (fold, (train, test)) in splits.iter().enumerate() {
                    let context = format!("fold {fold} (n_splits={n_splits}, n_groups={n_groups})");
                    assert!(!test.is_empty(), "{context}: empty test set");
                    assert!(!train.is_empty(), "{context}: empty train set");

                    let max_train = groups_of(&groups, train)
                        .into_iter()
                        .max()
                        .expect("train set is non-empty");
                    let min_test = groups_of(&groups, test)
                        .into_iter()
                        .min()
                        .expect("test set is non-empty");
                    assert!(
                        max_train < min_test,
                        "{context}: train group {max_train} is not strictly before \
                         test group {min_test}"
                    );

                    let test_set: std::collections::HashSet<usize> = test.iter().copied().collect();
                    assert!(
                        train.iter().all(|idx| !test_set.contains(idx)),
                        "{context}: train and test overlap"
                    );
                }
            }
        }
    }

    /// `group_gap` and `purge_length` stack and shrink the training window from
    /// the right-hand (most recent) end.
    #[test]
    fn test_purged_group_gap_and_purge_shrink_training_window() {
        let groups = make_groups(20, 2);
        let n_samples = groups.len();

        // groups_per_test = 20 / 5 = 4, so fold 3 tests groups 16..20.
        let plain = PurgedGroupTimeSeriesSplit::new(4).split_with_groups(n_samples, &groups);
        assert_eq!(
            groups_of(&groups, &plain[3].0).into_iter().max(),
            Some(15),
            "without gap/purge the training window should run right up to the test window"
        );

        let purged = PurgedGroupTimeSeriesSplit::new(4)
            .group_gap(1)
            .purge_length(2)
            .split_with_groups(n_samples, &groups);
        assert_eq!(
            groups_of(&groups, &purged[3].0).into_iter().max(),
            Some(12),
            "gap(1) + purge(2) should drop groups 13, 14 and 15"
        );
    }

    /// `max_train_group_size` turns the expanding window into a rolling one.
    #[test]
    fn test_purged_group_max_train_group_size_limits_history() {
        let groups = make_groups(20, 2);
        let splits = PurgedGroupTimeSeriesSplit::new(4)
            .max_train_group_size(3)
            .split_with_groups(groups.len(), &groups);

        let train_groups = groups_of(&groups, &splits[3].0);
        assert_eq!(
            (
                train_groups.iter().copied().min(),
                train_groups.iter().copied().max()
            ),
            (Some(13), Some(15))
        );
    }

    /// Training on the future is available, but only on explicit opt-in — and
    /// the embargo then applies to the groups just after the test window.
    #[test]
    fn test_purged_group_future_training_is_opt_in() {
        let groups = make_groups(12, 2);
        let n_samples = groups.len();

        let splits = PurgedGroupTimeSeriesSplit::new(3)
            .allow_future_train_groups(true)
            .split_with_groups(n_samples, &groups);
        assert_eq!(splits.len(), 3);

        // Non-causal layout tiles every group: fold 0 tests groups 0..4.
        let fold0_test = groups_of(&groups, &splits[0].1);
        assert_eq!(fold0_test.iter().copied().min(), Some(0));
        assert_eq!(fold0_test.iter().copied().max(), Some(3));
        // ...and its training data then legitimately comes from the future.
        assert_eq!(
            groups_of(&groups, &splits[0].0).into_iter().min(),
            Some(4),
            "opted-in mode should train on the groups after the test window"
        );

        let embargoed = PurgedGroupTimeSeriesSplit::new(3)
            .allow_future_train_groups(true)
            .embargo_length(2)
            .split_with_groups(n_samples, &groups);
        assert_eq!(
            groups_of(&groups, &embargoed[0].0).into_iter().min(),
            Some(6),
            "embargo(2) should drop groups 4 and 5 from the future side"
        );
    }

    /// Invalid configurations surface as errors through the `try_*` API.
    #[test]
    fn test_purged_group_try_api_reports_errors() {
        assert!(PurgedGroupTimeSeriesSplit::try_new(1).is_err());
        assert!(PurgedGroupTimeSeriesSplit::try_new(2).is_ok());

        let cv = PurgedGroupTimeSeriesSplit::new(2);
        let groups = make_groups(6, 2);
        assert!(cv.try_split_with_groups(groups.len() + 1, &groups).is_err());
        assert!(cv.try_split_with_groups(groups.len(), &groups).is_ok());

        // Two distinct groups cannot support two walk-forward folds.
        let tiny = make_groups(2, 2);
        assert!(cv.try_split_with_groups(tiny.len(), &tiny).is_err());
    }

    /// The `CrossValidator` entry point routes through the same walk-forward
    /// logic.
    #[test]
    fn test_purged_group_cross_validator_trait_split() {
        let groups = make_groups(12, 2);
        let cv = PurgedGroupTimeSeriesSplit::new(3);
        assert_eq!(CrossValidator::n_splits(&cv), 3);

        let splits = cv.split(groups.len(), Some(&groups));
        assert_eq!(splits.len(), 3);
        for (train, test) in &splits {
            let max_train = groups_of(&groups, train)
                .into_iter()
                .max()
                .expect("train set is non-empty");
            let min_test = groups_of(&groups, test)
                .into_iter()
                .min()
                .expect("test set is non-empty");
            assert!(max_train < min_test);
        }
    }
}
