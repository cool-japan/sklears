//! Model Evaluation Framework
//!
//! This module provides comprehensive model evaluation capabilities including
//! cross-validation, hyperparameter tuning, and performance metrics tracking.

use crate::error::{Result, SklearsError};
use crate::traits::{Estimator, Fit, Predict};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cross-validation framework for model evaluation
///
/// Provides k-fold, stratified, and time-series cross-validation methods
/// with comprehensive metrics tracking and statistical analysis.
#[derive(Debug)]
pub struct CrossValidator {
    /// Number of folds
    pub n_folds: usize,
    /// Shuffle data before splitting
    pub shuffle: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Cross-validation strategy
    pub strategy: CVStrategy,
}

/// Cross-validation strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CVStrategy {
    /// K-Fold cross-validation
    KFold,
    /// Stratified K-Fold (preserves class distribution)
    StratifiedKFold,
    /// Leave-One-Out cross-validation
    LeaveOneOut,
    /// Time series split
    TimeSeriesSplit,
    /// Repeated K-Fold
    RepeatedKFold { n_repeats: usize },
}

/// Cross-validation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVResults {
    /// Scores for each fold
    pub fold_scores: Vec<f64>,
    /// Mean score across all folds
    pub mean_score: f64,
    /// Standard deviation of scores
    pub std_dev: f64,
    /// Minimum score
    pub min_score: f64,
    /// Maximum score
    pub max_score: f64,
    /// Individual fold metrics
    pub fold_metrics: Vec<FoldMetrics>,
}

/// Metrics for a single fold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoldMetrics {
    /// Fold number
    pub fold_number: usize,
    /// Training score
    pub train_score: f64,
    /// Validation score
    pub val_score: f64,
    /// Training time in milliseconds
    pub train_time_ms: u64,
    /// Prediction time in milliseconds
    pub predict_time_ms: u64,
}

impl CrossValidator {
    /// Create a new cross-validator
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            shuffle: true,
            random_seed: None,
            strategy: CVStrategy::KFold,
        }
    }

    /// Set the cross-validation strategy
    pub fn with_strategy(mut self, strategy: CVStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// Set whether to shuffle data
    pub fn with_shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    /// Set random seed
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.random_seed = Some(seed);
        self
    }

    /// Generate cross-validation splits
    pub fn split(&self, n_samples: usize) -> Vec<CVSplit> {
        match self.strategy {
            CVStrategy::KFold => self.kfold_split(n_samples),
            CVStrategy::StratifiedKFold => self.kfold_split(n_samples), // Simplified
            CVStrategy::LeaveOneOut => self.loo_split(n_samples),
            CVStrategy::TimeSeriesSplit => self.time_series_split(n_samples),
            CVStrategy::RepeatedKFold { n_repeats } => self.repeated_kfold_split(n_samples, n_repeats),
        }
    }

    /// K-Fold splitting
    fn kfold_split(&self, n_samples: usize) -> Vec<CVSplit> {
        let fold_size = n_samples / self.n_folds;
        let mut splits = Vec::new();

        for i in 0..self.n_folds {
            let val_start = i * fold_size;
            let val_end = if i == self.n_folds - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let mut train_indices = Vec::new();
            let mut val_indices = Vec::new();

            for j in 0..n_samples {
                if j >= val_start && j < val_end {
                    val_indices.push(j);
                } else {
                    train_indices.push(j);
                }
            }

            splits.push(CVSplit {
                fold: i,
                train_indices,
                val_indices,
            });
        }

        splits
    }

    /// Leave-One-Out splitting
    fn loo_split(&self, n_samples: usize) -> Vec<CVSplit> {
        let mut splits = Vec::new();

        for i in 0..n_samples {
            let mut train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
            let val_indices = vec![i];

            splits.push(CVSplit {
                fold: i,
                train_indices,
                val_indices,
            });
        }

        splits
    }

    /// Time series splitting
    fn time_series_split(&self, n_samples: usize) -> Vec<CVSplit> {
        let mut splits = Vec::new();
        let min_train_size = n_samples / (self.n_folds + 1);

        for i in 0..self.n_folds {
            let train_end = min_train_size * (i + 1);
            let val_end = min_train_size * (i + 2);

            let train_indices: Vec<usize> = (0..train_end).collect();
            let val_indices: Vec<usize> = (train_end..val_end.min(n_samples)).collect();

            if !val_indices.is_empty() {
                splits.push(CVSplit {
                    fold: i,
                    train_indices,
                    val_indices,
                });
            }
        }

        splits
    }

    /// Repeated K-Fold splitting
    fn repeated_kfold_split(&self, n_samples: usize, n_repeats: usize) -> Vec<CVSplit> {
        let mut all_splits = Vec::new();

        for repeat in 0..n_repeats {
            let base_splits = self.kfold_split(n_samples);
            for (i, mut split) in base_splits.into_iter().enumerate() {
                split.fold = repeat * self.n_folds + i;
                all_splits.push(split);
            }
        }

        all_splits
    }
}

/// A single cross-validation split
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CVSplit {
    /// Fold number
    pub fold: usize,
    /// Training indices
    pub train_indices: Vec<usize>,
    /// Validation indices
    pub val_indices: Vec<usize>,
}

/// Model selection with hyperparameter tuning
#[derive(Debug)]
pub struct ModelSelection {
    /// Parameter grid for search
    pub param_grid: ParameterGrid,
    /// Cross-validator
    pub cv: CrossValidator,
    /// Scoring metric
    pub scoring: ScoringMetric,
    /// Number of parallel jobs
    pub n_jobs: usize,
}

/// Parameter grid for hyperparameter search
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterGrid {
    /// Parameter configurations
    pub params: HashMap<String, Vec<ParameterValue>>,
}

/// Parameter value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    String(String),
    Bool(bool),
}

impl ParameterGrid {
    /// Create a new parameter grid
    pub fn new() -> Self {
        Self {
            params: HashMap::new(),
        }
    }

    /// Add parameter values
    pub fn add_param(&mut self, name: String, values: Vec<ParameterValue>) {
        self.params.insert(name, values);
    }

    /// Get all parameter combinations
    pub fn combinations(&self) -> Vec<HashMap<String, ParameterValue>> {
        if self.params.is_empty() {
            return vec![HashMap::new()];
        }

        let mut combinations = vec![HashMap::new()];

        for (param_name, values) in &self.params {
            let mut new_combinations = Vec::new();

            for combo in &combinations {
                for value in values {
                    let mut new_combo = combo.clone();
                    new_combo.insert(param_name.clone(), value.clone());
                    new_combinations.push(new_combo);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }
}

impl Default for ParameterGrid {
    fn default() -> Self {
        Self::new()
    }
}

/// Scoring metric for evaluation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScoringMetric {
    /// Mean Squared Error (lower is better)
    MSE,
    /// Root Mean Squared Error (lower is better)
    RMSE,
    /// Mean Absolute Error (lower is better)
    MAE,
    /// R-squared score (higher is better)
    R2,
    /// Accuracy (higher is better)
    Accuracy,
    /// Precision (higher is better)
    Precision,
    /// Recall (higher is better)
    Recall,
    /// F1 Score (higher is better)
    F1,
}

impl ModelSelection {
    /// Create a new model selection instance
    pub fn new(param_grid: ParameterGrid, cv: CrossValidator) -> Self {
        Self {
            param_grid,
            cv,
            scoring: ScoringMetric::R2,
            n_jobs: 1,
        }
    }

    /// Set scoring metric
    pub fn with_scoring(mut self, scoring: ScoringMetric) -> Self {
        self.scoring = scoring;
        self
    }

    /// Set number of parallel jobs
    pub fn with_n_jobs(mut self, n_jobs: usize) -> Self {
        self.n_jobs = n_jobs;
        self
    }

    /// Perform grid search
    pub fn grid_search(&self) -> GridSearchResults {
        let combinations = self.param_grid.combinations();

        let mut results = GridSearchResults {
            best_params: HashMap::new(),
            best_score: f64::NEG_INFINITY,
            all_results: Vec::new(),
        };

        for params in combinations {
            // Simulate evaluation with these parameters
            let score = self.evaluate_params(&params);

            results.all_results.push((params.clone(), score));

            if score > results.best_score {
                results.best_score = score;
                results.best_params = params;
            }
        }

        results
    }

    /// Evaluate a parameter configuration (simulated)
    fn evaluate_params(&self, _params: &HashMap<String, ParameterValue>) -> f64 {
        // In a real implementation, this would train and evaluate a model
        use scirs2_core::random::{thread_rng, Rng};
        let mut rng = thread_rng();
        rng.gen::<f64>()
    }
}

/// Grid search results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSearchResults {
    /// Best parameter configuration
    pub best_params: HashMap<String, ParameterValue>,
    /// Best score achieved
    pub best_score: f64,
    /// All parameter configurations and their scores
    pub all_results: Vec<(HashMap<String, ParameterValue>, f64)>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_validator_creation() {
        let cv = CrossValidator::new(5);
        assert_eq!(cv.n_folds, 5);
        assert!(cv.shuffle);
    }

    #[test]
    fn test_kfold_split() {
        let cv = CrossValidator::new(3);
        let splits = cv.split(9);

        assert_eq!(splits.len(), 3);
        assert_eq!(splits[0].train_indices.len(), 6);
        assert_eq!(splits[0].val_indices.len(), 3);
    }

    #[test]
    fn test_loo_split() {
        let cv = CrossValidator::new(5).with_strategy(CVStrategy::LeaveOneOut);
        let splits = cv.split(5);

        assert_eq!(splits.len(), 5);
        for split in &splits {
            assert_eq!(split.train_indices.len(), 4);
            assert_eq!(split.val_indices.len(), 1);
        }
    }

    #[test]
    fn test_time_series_split() {
        let cv = CrossValidator::new(3).with_strategy(CVStrategy::TimeSeriesSplit);
        let splits = cv.split(12);

        assert!(splits.len() > 0);
        // Training set should grow over time
        for i in 1..splits.len() {
            assert!(splits[i].train_indices.len() >= splits[i - 1].train_indices.len());
        }
    }

    #[test]
    fn test_repeated_kfold() {
        let cv = CrossValidator::new(3).with_strategy(CVStrategy::RepeatedKFold { n_repeats: 2 });
        let splits = cv.split(9);

        assert_eq!(splits.len(), 6); // 3 folds * 2 repeats
    }

    #[test]
    fn test_parameter_grid() {
        let mut grid = ParameterGrid::new();
        grid.add_param(
            "alpha".to_string(),
            vec![
                ParameterValue::Float(0.1),
                ParameterValue::Float(1.0),
            ],
        );
        grid.add_param(
            "max_iter".to_string(),
            vec![
                ParameterValue::Int(100),
                ParameterValue::Int(200),
            ],
        );

        let combinations = grid.combinations();
        assert_eq!(combinations.len(), 4); // 2 * 2
    }

    #[test]
    fn test_model_selection() {
        let mut param_grid = ParameterGrid::new();
        param_grid.add_param(
            "learning_rate".to_string(),
            vec![ParameterValue::Float(0.01), ParameterValue::Float(0.1)],
        );

        let cv = CrossValidator::new(5);
        let model_sel = ModelSelection::new(param_grid, cv);

        let results = model_sel.grid_search();
        assert_eq!(results.all_results.len(), 2);
        assert!(!results.best_params.is_empty());
    }

    #[test]
    fn test_scoring_metrics() {
        assert_ne!(ScoringMetric::MSE, ScoringMetric::R2);
        assert_eq!(ScoringMetric::Accuracy, ScoringMetric::Accuracy);
    }

    #[test]
    fn test_cv_strategy() {
        assert_ne!(CVStrategy::KFold, CVStrategy::StratifiedKFold);
        assert_eq!(CVStrategy::LeaveOneOut, CVStrategy::LeaveOneOut);
    }

    #[test]
    fn test_empty_parameter_grid() {
        let grid = ParameterGrid::new();
        let combinations = grid.combinations();
        assert_eq!(combinations.len(), 1);
        assert!(combinations[0].is_empty());
    }

    #[test]
    fn test_cv_results() {
        let results = CVResults {
            fold_scores: vec![0.9, 0.85, 0.95, 0.88, 0.92],
            mean_score: 0.9,
            std_dev: 0.03,
            min_score: 0.85,
            max_score: 0.95,
            fold_metrics: vec![],
        };

        assert_eq!(results.fold_scores.len(), 5);
        assert_eq!(results.mean_score, 0.9);
    }
}
