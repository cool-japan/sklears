//! Early stopping utilities for linear models
//!
//! This module provides early stopping functionality to prevent overfitting during
//! cross-validation and iterative optimization processes.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{rngs::StdRng, SeedableRng, SliceRandomExt};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};
use std::collections::VecDeque;

/// Early stopping criteria
#[derive(Debug, Clone)]
pub enum StoppingCriterion {
    /// Stop when validation score doesn't improve for n consecutive iterations
    Patience(usize),
    /// Stop when validation score improvement is less than threshold for n iterations
    TolerancePatience { tolerance: Float, patience: usize },
    /// Stop when validation score reaches a target value
    TargetScore(Float),
    /// Stop when relative improvement is less than threshold
    RelativeImprovement(Float),
}

/// Configuration for early stopping
#[derive(Debug, Clone)]
pub struct EarlyStoppingConfig {
    /// Stopping criterion to use
    pub criterion: StoppingCriterion,
    /// Validation split ratio (0.0 to 1.0)
    pub validation_split: Float,
    /// Whether to shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for shuffling
    pub random_state: Option<u64>,
    /// Whether higher scores are better (default: true)
    pub higher_is_better: bool,
    /// Minimum number of iterations before early stopping can trigger
    pub min_iterations: usize,
    /// Whether to restore best weights when stopping
    pub restore_best_weights: bool,
}

impl Default for EarlyStoppingConfig {
    fn default() -> Self {
        Self {
            criterion: StoppingCriterion::Patience(5),
            validation_split: 0.2,
            shuffle: true,
            random_state: None,
            higher_is_better: true,
            min_iterations: 10,
            restore_best_weights: true,
        }
    }
}

/// Early stopping monitor that tracks validation performance
#[derive(Debug, Clone)]
pub struct EarlyStopping {
    config: EarlyStoppingConfig,
    best_score: Option<Float>,
    best_iteration: usize,
    patience_counter: usize,
    score_history: VecDeque<Float>,
    should_stop: bool,
    current_iteration: usize,
}

impl EarlyStopping {
    /// Create a new early stopping monitor
    pub fn new(config: EarlyStoppingConfig) -> Self {
        Self {
            config,
            best_score: None,
            best_iteration: 0,
            patience_counter: 0,
            score_history: VecDeque::new(),
            should_stop: false,
            current_iteration: 0,
        }
    }

    /// Update the monitor with a new validation score
    ///
    /// Returns true if training should continue, false if it should stop
    pub fn update(&mut self, validation_score: Float) -> bool {
        self.current_iteration += 1;
        self.score_history.push_back(validation_score);

        // Keep only relevant history for some criteria
        if self.score_history.len() > 100 {
            self.score_history.pop_front();
        }

        // Check if we should stop based on the criterion
        self.should_stop = self.check_stopping_criterion(validation_score);

        // Update best score and iteration
        if self.is_improvement(validation_score) {
            let previous_best = self.best_score;
            self.best_score = Some(validation_score);
            self.best_iteration = self.current_iteration;

            // For TolerancePatience, check if improvement is significant enough
            if let StoppingCriterion::TolerancePatience { tolerance, .. } = &self.config.criterion {
                if let Some(best) = previous_best {
                    let improvement = if self.config.higher_is_better {
                        validation_score - best
                    } else {
                        best - validation_score
                    };
                    if improvement > *tolerance {
                        self.patience_counter = 0;
                    } else {
                        self.patience_counter += 1;
                    }
                } else {
                    self.patience_counter = 0;
                }
            } else {
                self.patience_counter = 0;
            }
        } else {
            self.patience_counter += 1;
        }

        // Don't stop if we haven't reached minimum iterations
        if self.current_iteration < self.config.min_iterations {
            self.should_stop = false;
        }

        !self.should_stop
    }

    /// Check if the current score represents an improvement
    fn is_improvement(&self, score: Float) -> bool {
        match self.best_score {
            None => true,
            Some(best) => {
                if self.config.higher_is_better {
                    score > best
                } else {
                    score < best
                }
            }
        }
    }

    /// Check if stopping criterion is met
    fn check_stopping_criterion(&self, validation_score: Float) -> bool {
        match &self.config.criterion {
            StoppingCriterion::Patience(patience) => self.patience_counter >= *patience,
            StoppingCriterion::TolerancePatience {
                tolerance,
                patience,
            } => {
                if let Some(best) = self.best_score {
                    let improvement = if self.config.higher_is_better {
                        validation_score - best
                    } else {
                        best - validation_score
                    };
                    improvement <= *tolerance && self.patience_counter >= *patience
                } else {
                    false
                }
            }
            StoppingCriterion::TargetScore(target) => {
                if self.config.higher_is_better {
                    validation_score >= *target
                } else {
                    validation_score <= *target
                }
            }
            StoppingCriterion::RelativeImprovement(min_improvement) => {
                if let Some(best) = self.best_score {
                    let relative_improvement = if self.config.higher_is_better {
                        (validation_score - best) / best.abs()
                    } else {
                        (best - validation_score) / best.abs()
                    };
                    relative_improvement <= *min_improvement
                } else {
                    false
                }
            }
        }
    }

    /// Get the best validation score achieved so far
    pub fn best_score(&self) -> Option<Float> {
        self.best_score
    }

    /// Get the iteration where the best score was achieved
    pub fn best_iteration(&self) -> usize {
        self.best_iteration
    }

    /// Check if training should stop
    pub fn should_stop(&self) -> bool {
        self.should_stop
    }

    /// Get the current iteration count
    pub fn current_iteration(&self) -> usize {
        self.current_iteration
    }

    /// Get the validation score history
    pub fn score_history(&self) -> &VecDeque<Float> {
        &self.score_history
    }

    /// Reset the early stopping monitor
    pub fn reset(&mut self) {
        self.best_score = None;
        self.best_iteration = 0;
        self.patience_counter = 0;
        self.score_history.clear();
        self.should_stop = false;
        self.current_iteration = 0;
    }
}

/// Split data into training and validation sets
pub fn train_validation_split(
    x: &Array2<Float>,
    y: &Array1<Float>,
    validation_split: Float,
    shuffle: bool,
    random_state: Option<u64>,
) -> Result<(Array2<Float>, Array1<Float>, Array2<Float>, Array1<Float>)> {
    let n_samples = x.nrows();
    let _n_features = x.ncols();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y have inconsistent numbers of samples".to_string(),
        ));
    }

    if validation_split <= 0.0 || validation_split >= 1.0 {
        return Err(SklearsError::InvalidParameter {
            name: "validation_split".to_string(),
            reason: "must be between 0.0 and 1.0".to_string(),
        });
    }

    let val_size = (n_samples as Float * validation_split).round() as usize;
    let train_size = n_samples - val_size;

    // Create indices
    let mut indices: Vec<usize> = (0..n_samples).collect();

    // Shuffle if requested
    if shuffle {
        let mut rng = match random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
        };
        indices.shuffle(&mut rng);
    }

    // Split indices
    let train_indices = &indices[..train_size];
    let val_indices = &indices[train_size..];

    // Create training and validation sets
    let x_train = x.select(Axis(0), train_indices);
    let y_train = y.select(Axis(0), train_indices);
    let x_val = x.select(Axis(0), val_indices);
    let y_val = y.select(Axis(0), val_indices);

    Ok((x_train, y_train, x_val, y_val))
}

/// Early stopping callback trait for integration with optimizers
pub trait EarlyStoppingCallback {
    /// Called after each iteration with validation score
    /// Returns true if training should continue
    fn on_iteration(&mut self, iteration: usize, validation_score: Float) -> bool;

    /// Called when training stops early
    fn on_early_stop(&mut self) {}

    /// Get the best score achieved
    fn best_score(&self) -> Option<Float>;
}

impl EarlyStoppingCallback for EarlyStopping {
    fn on_iteration(&mut self, _iteration: usize, validation_score: Float) -> bool {
        self.update(validation_score)
    }

    fn on_early_stop(&mut self) {
        // Could log or perform cleanup here
    }

    fn best_score(&self) -> Option<Float> {
        self.best_score
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_early_stopping_patience() {
        let config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(3),
            min_iterations: 1,
            ..Default::default()
        };

        let mut early_stopping = EarlyStopping::new(config);

        // Improving scores
        assert!(early_stopping.update(0.8)); // Continue
        assert!(early_stopping.update(0.85)); // Continue (improvement)
        assert!(early_stopping.update(0.9)); // Continue (improvement)

        // No improvement for 3 iterations
        assert!(early_stopping.update(0.88)); // Continue (patience=1)
        assert!(early_stopping.update(0.87)); // Continue (patience=2)
        assert!(early_stopping.update(0.86)); // Continue (patience=3)
        assert!(!early_stopping.update(0.85)); // Stop (patience exceeded)

        assert_eq!(early_stopping.best_score(), Some(0.9));
        assert_eq!(early_stopping.best_iteration(), 3);
    }

    #[test]
    fn test_early_stopping_tolerance_patience() {
        let config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TolerancePatience {
                tolerance: 0.01,
                patience: 2,
            },
            min_iterations: 1,
            ..Default::default()
        };

        let mut early_stopping = EarlyStopping::new(config);

        assert!(early_stopping.update(0.8)); // Continue
        assert!(early_stopping.update(0.9)); // Continue (improvement > tolerance, reset patience)
        assert!(early_stopping.update(0.905)); // Continue (improvement <= tolerance, patience=1)
        assert!(early_stopping.update(0.907)); // Continue (improvement <= tolerance, patience=2)
        assert!(!early_stopping.update(0.908)); // Stop (improvement <= tolerance, patience exceeded)
    }

    #[test]
    fn test_early_stopping_target_score() {
        let config = EarlyStoppingConfig {
            criterion: StoppingCriterion::TargetScore(0.95),
            min_iterations: 1,
            ..Default::default()
        };

        let mut early_stopping = EarlyStopping::new(config);

        assert!(early_stopping.update(0.8)); // Continue
        assert!(early_stopping.update(0.9)); // Continue
        assert!(!early_stopping.update(0.96)); // Stop (target reached)
    }

    #[test]
    fn test_min_iterations() {
        let config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(1),
            min_iterations: 5,
            ..Default::default()
        };

        let mut early_stopping = EarlyStopping::new(config);

        // Even with patience=1, should continue until min_iterations
        assert!(early_stopping.update(0.9));
        assert!(early_stopping.update(0.8)); // No improvement, but min_iterations not reached
        assert!(early_stopping.update(0.7)); // No improvement, but min_iterations not reached
        assert!(early_stopping.update(0.6)); // No improvement, but min_iterations not reached
        assert!(!early_stopping.update(0.5)); // min_iterations reached (5), patience exceeded, can stop now
    }

    #[test]
    fn test_train_validation_split() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0];

        let (x_train, y_train, x_val, y_val) =
            train_validation_split(&x, &y, 0.4, false, Some(42)).unwrap();

        assert_eq!(x_train.nrows(), 3); // 60% of 5 = 3
        assert_eq!(x_val.nrows(), 2); // 40% of 5 = 2
        assert_eq!(y_train.len(), 3);
        assert_eq!(y_val.len(), 2);

        // Check total samples preserved
        assert_eq!(x_train.nrows() + x_val.nrows(), x.nrows());
        assert_eq!(y_train.len() + y_val.len(), y.len());
    }

    #[test]
    fn test_train_validation_split_invalid_ratio() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![1.0, 2.0];

        // Test invalid validation split
        let result = train_validation_split(&x, &y, 1.5, false, None);
        assert!(result.is_err());

        let result = train_validation_split(&x, &y, 0.0, false, None);
        assert!(result.is_err());
    }

    #[test]
    fn test_lower_is_better() {
        let config = EarlyStoppingConfig {
            criterion: StoppingCriterion::Patience(2),
            higher_is_better: false, // Lower scores are better (e.g., loss)
            min_iterations: 1,
            ..Default::default()
        };

        let mut early_stopping = EarlyStopping::new(config);

        assert!(early_stopping.update(1.0)); // Continue
        assert!(early_stopping.update(0.8)); // Continue (improvement: lower is better)
        assert!(early_stopping.update(0.6)); // Continue (improvement)
        assert!(early_stopping.update(0.7)); // Continue (no improvement, patience=1)
        assert!(early_stopping.update(0.8)); // Continue (no improvement, patience=2)
        assert!(!early_stopping.update(0.9)); // Stop (patience exceeded)

        assert_eq!(early_stopping.best_score(), Some(0.6));
        assert_eq!(early_stopping.best_iteration(), 3);
    }
}
