use scirs2_core::ndarray::Array1;
use scirs2_core::random::Rng;
use sklears_core::error::{Result, SklearsError};
use sklears_core::types::{Float, Int};
use std::collections::HashMap;

use super::validation_core::create_rng;

/// Create cross-validation folds from indices
pub fn create_cv_folds(indices: &[usize], n_folds: usize) -> Vec<(Vec<usize>, Vec<usize>)> {
    let fold_size = indices.len() / n_folds;
    let mut folds = Vec::with_capacity(n_folds);

    for fold in 0..n_folds {
        let start_idx = fold * fold_size;
        let end_idx = if fold == n_folds - 1 {
            indices.len()
        } else {
            (fold + 1) * fold_size
        };

        let mut train_indices = Vec::new();
        let mut test_indices = Vec::new();

        for (i, &idx) in indices.iter().enumerate() {
            if i >= start_idx && i < end_idx {
                test_indices.push(idx);
            } else {
                train_indices.push(idx);
            }
        }

        folds.push((train_indices, test_indices));
    }

    folds
}

/// Create stratified cross-validation folds
pub fn create_stratified_folds(
    y: &Array1<Int>,
    n_folds: usize,
    random_state: Option<u64>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_folds < 2 {
        return Err(SklearsError::InvalidInput(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);

    // Group indices by class
    let mut class_indices: HashMap<Int, Vec<usize>> = HashMap::new();
    for (i, &class) in y.iter().enumerate() {
        class_indices.entry(class).or_default().push(i);
    }

    // Shuffle indices within each class
    for indices in class_indices.values_mut() {
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }
    }

    // Create stratified folds
    let mut folds: Vec<(Vec<usize>, Vec<usize>)> = Vec::with_capacity(n_folds);
    for _ in 0..n_folds {
        folds.push((Vec::new(), Vec::new()));
    }

    // Distribute samples from each class across folds
    for (_, indices) in class_indices {
        let fold_size = indices.len() / n_folds;
        let remainder = indices.len() % n_folds;

        let mut start = 0;
        for fold_idx in 0..n_folds {
            let current_fold_size = fold_size + if fold_idx < remainder { 1 } else { 0 };
            let end = start + current_fold_size;

            for &idx in &indices[start..end] {
                folds[fold_idx].1.push(idx);
            }

            // Add to train sets of other folds
            for other_fold_idx in 0..n_folds {
                if other_fold_idx != fold_idx {
                    for &idx in &indices[start..end] {
                        folds[other_fold_idx].0.push(idx);
                    }
                }
            }

            start = end;
        }
    }

    Ok(folds)
}

/// Create shuffled indices
pub fn create_shuffled_indices(n_samples: usize, random_state: Option<u64>) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..n_samples).collect();
    let mut rng = create_rng(random_state);

    for i in (1..indices.len()).rev() {
        let j = rng.gen_range(0..=i);
        indices.swap(i, j);
    }

    indices
}

/// Train-test split with optional stratification
pub fn train_test_split(
    n_samples: usize,
    test_size: Option<Float>,
    train_size: Option<Float>,
    random_state: Option<u64>,
    shuffle: bool,
    stratify: Option<&Array1<Int>>,
) -> Result<TrainTestSplit> {
    // Determine split sizes
    let test_size = match (test_size, train_size) {
        (Some(ts), None) => ts,
        (None, Some(tr_s)) => 1.0 - tr_s,
        (Some(ts), Some(tr_s)) => {
            if (ts + tr_s - 1.0).abs() > 1e-6 {
                return Err(SklearsError::InvalidInput(
                    "test_size and train_size must sum to 1.0".to_string(),
                ));
            }
            ts
        }
        (None, None) => 0.25, // Default 25% test size
    };

    if test_size <= 0.0 || test_size >= 1.0 {
        return Err(SklearsError::InvalidInput(
            "test_size must be between 0 and 1".to_string(),
        ));
    }

    let test_samples = (n_samples as Float * test_size).round() as usize;
    let train_samples = n_samples - test_samples;

    if test_samples == 0 || train_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "Both train and test sets must have at least one sample".to_string(),
        ));
    }

    let indices = if shuffle {
        create_shuffled_indices(n_samples, random_state)
    } else {
        (0..n_samples).collect()
    };

    let (train_indices, test_indices) = if let Some(y) = stratify {
        create_stratified_train_test_split(y, test_size, random_state)?
    } else {
        let train_indices = indices[..train_samples].to_vec();
        let test_indices = indices[train_samples..].to_vec();
        (train_indices, test_indices)
    };

    Ok(TrainTestSplit {
        train_indices: train_indices.clone(),
        test_indices: test_indices.clone(),
        train_size: train_indices.len(),
        test_size: test_indices.len(),
    })
}

/// Create stratified train-test split
fn create_stratified_train_test_split(
    y: &Array1<Int>,
    test_size: Float,
    random_state: Option<u64>,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let mut rng = create_rng(random_state);

    // Group indices by class
    let mut class_indices: HashMap<Int, Vec<usize>> = HashMap::new();
    for (i, &class) in y.iter().enumerate() {
        class_indices.entry(class).or_default().push(i);
    }

    let mut train_indices = Vec::new();
    let mut test_indices = Vec::new();

    // Split each class proportionally
    for (_, mut indices) in class_indices {
        // Shuffle indices within class
        for i in (1..indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        let n_test = (indices.len() as Float * test_size).round() as usize;
        let n_train = indices.len() - n_test;

        test_indices.extend(&indices[..n_test]);
        train_indices.extend(&indices[n_test..]);
    }

    Ok((train_indices, test_indices))
}

/// Create time series cross-validation splits
pub fn create_time_series_splits(
    n_samples: usize,
    n_splits: usize,
    test_size: usize,
    gap: usize,
) -> Result<Vec<TimeSeriesSplit>> {
    if n_splits < 1 {
        return Err(SklearsError::InvalidInput(
            "Number of splits must be at least 1".to_string(),
        ));
    }

    if test_size < 1 {
        return Err(SklearsError::InvalidInput(
            "Test size must be at least 1".to_string(),
        ));
    }

    if n_samples < n_splits + test_size + gap {
        return Err(SklearsError::InvalidInput(
            "Insufficient samples for time series cross-validation".to_string(),
        ));
    }

    let mut splits = Vec::with_capacity(n_splits);

    for i in 0..n_splits {
        let train_end = n_samples - (n_splits - i) * test_size - gap;
        let test_start = train_end + gap;
        let test_end = test_start + test_size;

        if train_end < 1 || test_end > n_samples {
            continue;
        }

        let train_indices: Vec<usize> = (0..train_end).collect();
        let test_indices: Vec<usize> = (test_start..test_end).collect();

        splits.push(TimeSeriesSplit {
            fold_index: i,
            train_indices,
            test_indices,
            train_size: train_end,
            test_size,
            gap,
        });
    }

    if splits.is_empty() {
        return Err(SklearsError::InvalidInput(
            "No valid splits created for time series cross-validation".to_string(),
        ));
    }

    Ok(splits)
}

/// Create group-based cross-validation folds
pub fn create_group_folds(
    groups: &Array1<Int>,
    n_folds: usize,
    random_state: Option<u64>,
) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
    if n_folds < 2 {
        return Err(SklearsError::InvalidInput(
            "Number of folds must be at least 2".to_string(),
        ));
    }

    let mut rng = create_rng(random_state);

    // Get unique groups and their indices
    let mut group_indices: HashMap<Int, Vec<usize>> = HashMap::new();
    for (i, &group) in groups.iter().enumerate() {
        group_indices.entry(group).or_default().push(i);
    }

    let mut unique_groups: Vec<Int> = group_indices.keys().copied().collect();

    // Shuffle groups
    for i in (1..unique_groups.len()).rev() {
        let j = rng.gen_range(0..=i);
        unique_groups.swap(i, j);
    }

    if unique_groups.len() < n_folds {
        return Err(SklearsError::InvalidInput(
            "Number of groups must be at least equal to number of folds".to_string(),
        ));
    }

    // Assign groups to folds
    let groups_per_fold = unique_groups.len() / n_folds;
    let remainder = unique_groups.len() % n_folds;

    let mut folds = Vec::with_capacity(n_folds);
    let mut group_start = 0;

    for fold_idx in 0..n_folds {
        let current_fold_groups = groups_per_fold + if fold_idx < remainder { 1 } else { 0 };
        let group_end = group_start + current_fold_groups;

        let test_groups = &unique_groups[group_start..group_end];
        let mut test_indices = Vec::new();
        let mut train_indices = Vec::new();

        for &group in test_groups {
            test_indices.extend(&group_indices[&group]);
        }

        for &group in &unique_groups {
            if !test_groups.contains(&group) {
                train_indices.extend(&group_indices[&group]);
            }
        }

        folds.push((train_indices, test_indices));
        group_start = group_end;
    }

    Ok(folds)
}

/// Result of train-test split
#[derive(Debug, Clone)]
pub struct TrainTestSplit {
    /// train_indices
    pub train_indices: Vec<usize>,
    /// test_indices
    pub test_indices: Vec<usize>,
    /// train_size
    pub train_size: usize,
    /// test_size
    pub test_size: usize,
}

/// Time series cross-validation split
#[derive(Debug, Clone)]
pub struct TimeSeriesSplit {
    /// fold_index
    pub fold_index: usize,
    /// train_indices
    pub train_indices: Vec<usize>,
    /// test_indices
    pub test_indices: Vec<usize>,
    /// train_size
    pub train_size: usize,
    /// test_size
    pub test_size: usize,
    /// gap
    pub gap: usize,
}

/// Cross-validation strategy
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// KFold
    KFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// StratifiedKFold
    StratifiedKFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// GroupKFold
    GroupKFold { n_splits: usize },
    /// TimeSeriesSplit
    TimeSeriesSplit {
        n_splits: usize,
        test_size: usize,
        gap: usize,
    },
    /// LeaveOneOut
    LeaveOneOut,
    /// LeavePOut
    LeavePOut { p: usize },
}

impl CVStrategy {
    pub fn split(
        &self,
        n_samples: usize,
        y: Option<&Array1<Int>>,
        groups: Option<&Array1<Int>>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        match self {
            CVStrategy::KFold {
                n_splits,
                shuffle,
                random_state,
            } => {
                let indices = if *shuffle {
                    create_shuffled_indices(n_samples, *random_state)
                } else {
                    (0..n_samples).collect()
                };
                Ok(create_cv_folds(&indices, *n_splits))
            }
            CVStrategy::StratifiedKFold {
                n_splits,
                shuffle: _,
                random_state,
            } => match y {
                Some(y_array) => create_stratified_folds(y_array, *n_splits, *random_state),
                None => Err(SklearsError::InvalidInput(
                    "Stratified K-Fold requires target variable y".to_string(),
                )),
            },
            CVStrategy::GroupKFold { n_splits } => match groups {
                Some(group_array) => create_group_folds(group_array, *n_splits, None),
                None => Err(SklearsError::InvalidInput(
                    "Group K-Fold requires groups".to_string(),
                )),
            },
            CVStrategy::TimeSeriesSplit {
                n_splits,
                test_size,
                gap,
            } => {
                let ts_splits = create_time_series_splits(n_samples, *n_splits, *test_size, *gap)?;
                Ok(ts_splits
                    .into_iter()
                    .map(|split| (split.train_indices, split.test_indices))
                    .collect())
            }
            CVStrategy::LeaveOneOut => {
                let mut folds = Vec::with_capacity(n_samples);
                for i in 0..n_samples {
                    let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
                    let test_indices = vec![i];
                    folds.push((train_indices, test_indices));
                }
                Ok(folds)
            }
            CVStrategy::LeavePOut { p } => {
                if *p >= n_samples {
                    return Err(SklearsError::InvalidInput(
                        "p must be less than number of samples".to_string(),
                    ));
                }

                let combinations = generate_combinations(n_samples, *p);
                let mut folds = Vec::with_capacity(combinations.len());

                for test_indices in combinations {
                    let train_indices: Vec<usize> = (0..n_samples)
                        .filter(|i| !test_indices.contains(i))
                        .collect();
                    folds.push((train_indices, test_indices));
                }

                Ok(folds)
            }
        }
    }
}

/// Generate all combinations of size k from 0..n
fn generate_combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }
    if k == n {
        return vec![(0..n).collect()];
    }

    let mut result = Vec::new();

    // Recursive combination generation
    fn generate_recursive(
        start: usize,
        n: usize,
        k: usize,
        current: &mut Vec<usize>,
        result: &mut Vec<Vec<usize>>,
    ) {
        if k == 0 {
            result.push(current.clone());
            return;
        }
        if start >= n {
            return;
        }

        for i in start..=(n - k) {
            current.push(i);
            generate_recursive(i + 1, n, k - 1, current, result);
            current.pop();
        }
    }

    let mut current = Vec::new();
    generate_recursive(0, n, k, &mut current, &mut result);
    result
}
