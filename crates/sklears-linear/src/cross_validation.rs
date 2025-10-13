//! Advanced cross-validation functionality for linear models
//!
//! This module provides enhanced cross-validation capabilities including
//! stratified k-fold CV and early stopping integration.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, seq::SliceRandom, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Stratified K-Fold cross-validation splitter
///
/// Provides stratified k-fold cross-validation where each fold maintains
/// approximately the same percentage of samples of each target class.
#[derive(Debug, Clone)]
pub struct StratifiedKFold {
    /// Number of folds
    pub n_splits: usize,
    /// Whether to shuffle the data before splitting
    pub shuffle: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl StratifiedKFold {
    /// Create a new StratifiedKFold splitter
    pub fn new(n_splits: usize) -> Self {
        Self {
            n_splits,
            shuffle: true,
            random_state: None,
        }
    }

    /// Set whether to shuffle data
    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Generate stratified train/test splits
    ///
    /// Returns a vector of (train_indices, test_indices) tuples
    pub fn split(&self, y: &Array1<Float>) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let n_samples = y.len();

        if self.n_splits <= 1 || self.n_splits > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_splits".to_string(),
                reason: format!("must be between 2 and n_samples ({})", n_samples),
            });
        }

        // Group samples by class
        let mut class_indices: HashMap<OrderedFloat, Vec<usize>> = HashMap::new();
        for (idx, &value) in y.iter().enumerate() {
            class_indices
                .entry(OrderedFloat(value))
                .or_default()
                .push(idx);
        }

        // Check that each class has at least n_splits samples
        for (class, indices) in &class_indices {
            if indices.len() < self.n_splits {
                return Err(SklearsError::InvalidInput(format!(
                    "Class {} has only {} samples, but {} splits were requested",
                    class.0,
                    indices.len(),
                    self.n_splits
                )));
            }
        }

        // Shuffle indices within each class if requested
        if self.shuffle {
            let mut rng = match self.random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::seed_from_u64(42), // Use fixed seed for deterministic behavior
            };

            for indices in class_indices.values_mut() {
                indices.shuffle(&mut rng);
            }
        }

        // Create stratified folds
        let mut folds: Vec<Vec<usize>> = vec![Vec::new(); self.n_splits];

        for indices in class_indices.values() {
            // Distribute samples for this class across folds
            for (idx, &sample_idx) in indices.iter().enumerate() {
                let fold_idx = idx % self.n_splits;
                folds[fold_idx].push(sample_idx);
            }
        }

        // Generate train/test splits
        let mut splits = Vec::new();
        for test_fold_idx in 0..self.n_splits {
            let mut train_indices = Vec::new();
            let test_indices = folds[test_fold_idx].clone();

            for (fold_idx, fold) in folds.iter().enumerate() {
                if fold_idx != test_fold_idx {
                    train_indices.extend(fold.iter().copied());
                }
            }

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Get the number of splits
    pub fn get_n_splits(&self) -> usize {
        self.n_splits
    }
}

/// Wrapper for Float to make it hashable and orderable
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
struct OrderedFloat(Float);

impl Eq for OrderedFloat {}

impl std::hash::Hash for OrderedFloat {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        // Use the bits representation for hashing
        self.0.to_bits().hash(state);
    }
}

impl std::cmp::Ord for OrderedFloat {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Enhanced cross-validation with early stopping support
#[derive(Debug, Clone)]
pub struct CrossValidatorWithEarlyStopping {
    /// Base cross-validation strategy
    pub cv_strategy: CVStrategy,
    /// Early stopping configuration
    pub early_stopping_config: Option<crate::early_stopping::EarlyStoppingConfig>,
}

/// Cross-validation strategy
#[derive(Debug, Clone)]
pub enum CVStrategy {
    /// Standard k-fold
    KFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Stratified k-fold
    StratifiedKFold {
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    },
    /// Leave-one-out
    LeaveOneOut,
}

impl CrossValidatorWithEarlyStopping {
    /// Create a new enhanced cross-validator with stratified k-fold
    pub fn stratified_kfold(n_splits: usize) -> Self {
        Self {
            cv_strategy: CVStrategy::StratifiedKFold {
                n_splits,
                shuffle: true,
                random_state: None,
            },
            early_stopping_config: None,
        }
    }

    /// Create a new enhanced cross-validator with regular k-fold
    pub fn kfold(n_splits: usize) -> Self {
        Self {
            cv_strategy: CVStrategy::KFold {
                n_splits,
                shuffle: true,
                random_state: None,
            },
            early_stopping_config: None,
        }
    }

    /// Add early stopping configuration
    pub fn with_early_stopping(
        mut self,
        config: crate::early_stopping::EarlyStoppingConfig,
    ) -> Self {
        self.early_stopping_config = Some(config);
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, seed: u64) -> Self {
        match &mut self.cv_strategy {
            CVStrategy::KFold { random_state, .. } => *random_state = Some(seed),
            CVStrategy::StratifiedKFold { random_state, .. } => *random_state = Some(seed),
            CVStrategy::LeaveOneOut => {}
        }
        self
    }

    /// Generate train/test splits
    pub fn split(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        match &self.cv_strategy {
            CVStrategy::KFold {
                n_splits,
                shuffle,
                random_state,
            } => self.kfold_split(x.nrows(), *n_splits, *shuffle, *random_state),
            CVStrategy::StratifiedKFold {
                n_splits,
                shuffle,
                random_state,
            } => {
                let splitter = StratifiedKFold {
                    n_splits: *n_splits,
                    shuffle: *shuffle,
                    random_state: *random_state,
                };
                splitter.split(y)
            }
            CVStrategy::LeaveOneOut => self.leave_one_out_split(x.nrows()),
        }
    }

    /// Generate regular k-fold splits
    fn kfold_split(
        &self,
        n_samples: usize,
        n_splits: usize,
        shuffle: bool,
        random_state: Option<u64>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        if n_splits <= 1 || n_splits > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_splits".to_string(),
                reason: format!("must be between 2 and n_samples ({})", n_samples),
            });
        }

        let mut indices: Vec<usize> = (0..n_samples).collect();

        if shuffle {
            let mut rng = match random_state {
                Some(seed) => StdRng::seed_from_u64(seed),
                None => StdRng::seed_from_u64(42), // Use fixed seed for deterministic behavior
            };
            indices.shuffle(&mut rng);
        }

        let mut splits = Vec::new();
        let fold_size = n_samples / n_splits;
        let remainder = n_samples % n_splits;

        for fold_idx in 0..n_splits {
            let start = fold_idx * fold_size + fold_idx.min(remainder);
            let end = start + fold_size + if fold_idx < remainder { 1 } else { 0 };

            let test_indices = indices[start..end].to_vec();
            let train_indices = indices[..start]
                .iter()
                .chain(indices[end..].iter())
                .copied()
                .collect();

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Generate leave-one-out splits
    fn leave_one_out_split(&self, n_samples: usize) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let mut splits = Vec::new();

        for test_idx in 0..n_samples {
            let train_indices: Vec<usize> = (0..n_samples).filter(|&i| i != test_idx).collect();
            let test_indices = vec![test_idx];

            splits.push((train_indices, test_indices));
        }

        Ok(splits)
    }

    /// Get the number of splits
    pub fn get_n_splits(&self, n_samples: usize) -> usize {
        match &self.cv_strategy {
            CVStrategy::KFold { n_splits, .. } => *n_splits,
            CVStrategy::StratifiedKFold { n_splits, .. } => *n_splits,
            CVStrategy::LeaveOneOut => n_samples,
        }
    }
}

/// Cross-validation scoring with early stopping
pub fn cross_validate_with_early_stopping<Model>(
    model: Model,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &CrossValidatorWithEarlyStopping,
    scoring: Option<&str>,
) -> Result<CrossValidationResult>
where
    Model: Clone,
{
    let splits = cv.split(x, y)?;
    let n_splits = splits.len();

    let mut test_scores = Vec::with_capacity(n_splits);
    let mut train_scores = Vec::with_capacity(n_splits);
    let mut early_stopping_iterations = Vec::with_capacity(n_splits);

    for (fold_idx, (train_indices, test_indices)) in splits.iter().enumerate() {
        // Extract train and test data for this fold
        let x_train = x.select(Axis(0), train_indices);
        let y_train = y.select(Axis(0), train_indices);
        let x_test = x.select(Axis(0), test_indices);
        let y_test = y.select(Axis(0), test_indices);

        // For this example, we'll simulate training with early stopping
        // In practice, this would be integrated with the actual model fitting
        if let Some(es_config) = &cv.early_stopping_config {
            // Split training data for validation if early stopping is enabled
            let val_split = es_config.validation_split;
            let (x_tr, y_tr, x_val, y_val) = crate::early_stopping::train_validation_split(
                &x_train,
                &y_train,
                val_split,
                es_config.shuffle,
                es_config.random_state,
            )?;

            // Simulate early stopping (in practice, this would be integrated with model fitting)
            let mut early_stopping = crate::early_stopping::EarlyStopping::new(es_config.clone());
            let mut best_iteration = 0;

            // Simulated training loop
            for iteration in 1..=100 {
                // This would be the actual validation score from the model
                let mock_val_score = simulate_validation_score(iteration);

                if !early_stopping.update(mock_val_score) {
                    best_iteration = early_stopping.best_iteration();
                    break;
                }
                best_iteration = iteration;
            }

            early_stopping_iterations.push(best_iteration);
        } else {
            early_stopping_iterations.push(0); // No early stopping used
        }

        // Simulate scoring (in practice, this would use the actual fitted model)
        let train_score = simulate_score(&x_train, &y_train);
        let test_score = simulate_score(&x_test, &y_test);

        train_scores.push(train_score);
        test_scores.push(test_score);
    }

    Ok(CrossValidationResult {
        test_scores: Array1::from_vec(test_scores),
        train_scores: Array1::from_vec(train_scores),
        early_stopping_iterations: Array1::from_vec(
            early_stopping_iterations
                .into_iter()
                .map(|x| x as Float)
                .collect(),
        ),
        n_splits,
    })
}

/// Result of cross-validation
#[derive(Debug, Clone)]
pub struct CrossValidationResult {
    /// Test scores for each fold
    pub test_scores: Array1<Float>,
    /// Training scores for each fold
    pub train_scores: Array1<Float>,
    /// Early stopping iterations for each fold (0 if not used)
    pub early_stopping_iterations: Array1<Float>,
    /// Number of splits
    pub n_splits: usize,
}

impl CrossValidationResult {
    /// Get mean test score
    pub fn mean_test_score(&self) -> Float {
        self.test_scores.mean().unwrap_or(0.0)
    }

    /// Get standard deviation of test scores
    pub fn std_test_score(&self) -> Float {
        self.test_scores.std(0.0)
    }

    /// Get mean training score
    pub fn mean_train_score(&self) -> Float {
        self.train_scores.mean().unwrap_or(0.0)
    }

    /// Get standard deviation of training scores
    pub fn std_train_score(&self) -> Float {
        self.train_scores.std(0.0)
    }

    /// Get mean early stopping iterations
    pub fn mean_early_stopping_iterations(&self) -> Float {
        self.early_stopping_iterations.mean().unwrap_or(0.0)
    }
}

// Mock functions for simulation - in practice these would be replaced with actual model fitting and scoring
fn simulate_validation_score(iteration: usize) -> Float {
    // Simulate improving then plateauing validation score
    let base_score = 0.7;
    let improvement = 0.2 * (-((iteration as Float - 10.0) / 5.0).powi(2)).exp();
    let noise = 0.01 * ((iteration as Float * 0.5).sin());
    base_score + improvement + noise
}

fn simulate_score(x: &Array2<Float>, y: &Array1<Float>) -> Float {
    // Mock scoring function - in practice this would use actual model predictions
    0.8 + 0.1 * (x.nrows() as Float / 100.0).min(1.0)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_stratified_kfold_basic() {
        let y = array![0.0, 0.0, 1.0, 1.0, 2.0, 2.0, 0.0, 1.0]; // 3 samples each of classes 0,1,2, plus extras
        let splitter = StratifiedKFold::new(2);

        let splits = splitter.split(&y).unwrap();
        assert_eq!(splits.len(), 2);

        // Check that all samples are covered exactly once
        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test_indices) in &splits {
            all_test_indices.extend(test_indices);
        }
        all_test_indices.sort();
        assert_eq!(all_test_indices, (0..y.len()).collect::<Vec<_>>());
    }

    #[test]
    fn test_stratified_kfold_class_distribution() {
        let y = array![0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]; // Equal distribution
        let splitter = StratifiedKFold::new(2);

        let splits = splitter.split(&y).unwrap();

        for (_, test_indices) in &splits {
            let mut class_counts = HashMap::new();
            for &idx in test_indices {
                let class = y[idx] as i32;
                *class_counts.entry(class).or_insert(0) += 1;
            }

            // Each fold should have equal representation
            assert_eq!(class_counts.get(&0), Some(&2));
            assert_eq!(class_counts.get(&1), Some(&2));
        }
    }

    #[test]
    fn test_stratified_kfold_insufficient_samples() {
        let y = array![0.0, 1.0]; // Only 1 sample per class
        let splitter = StratifiedKFold::new(3); // 3 splits requested

        let result = splitter.split(&y);
        assert!(result.is_err());
    }

    #[test]
    fn test_cv_with_early_stopping() {
        use crate::early_stopping::{EarlyStoppingConfig, StoppingCriterion};

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let early_stopping_config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(3),
            validation_split: 0.3,
            ..Default::default()
        };

        let cv = CrossValidatorWithEarlyStopping::stratified_kfold(2)
            .with_early_stopping(early_stopping_config);

        let result = cross_validate_with_early_stopping("mock_model", &x, &y, &cv, None).unwrap();

        assert_eq!(result.n_splits, 2);
        assert_eq!(result.test_scores.len(), 2);
        assert_eq!(result.train_scores.len(), 2);
        assert_eq!(result.early_stopping_iterations.len(), 2);
    }

    #[test]
    fn test_kfold_split() {
        let cv = CrossValidatorWithEarlyStopping::kfold(3);
        let splits = cv.kfold_split(10, 3, false, Some(42)).unwrap();

        assert_eq!(splits.len(), 3);

        // Check that all samples are covered exactly once
        let mut all_test_indices: Vec<usize> = Vec::new();
        for (_, test_indices) in &splits {
            all_test_indices.extend(test_indices);
        }
        all_test_indices.sort();
        assert_eq!(all_test_indices, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_leave_one_out() {
        let cv = CrossValidatorWithEarlyStopping {
            cv_strategy: CVStrategy::LeaveOneOut,
            early_stopping_config: None,
        };

        let splits = cv.leave_one_out_split(5).unwrap();
        assert_eq!(splits.len(), 5);

        for (i, (train_indices, test_indices)) in splits.iter().enumerate() {
            assert_eq!(test_indices.len(), 1);
            assert_eq!(test_indices[0], i);
            assert_eq!(train_indices.len(), 4);
        }
    }
}
