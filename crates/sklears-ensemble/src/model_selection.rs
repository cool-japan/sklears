//! Model selection utilities for ensemble methods
//!
//! This module provides advanced model selection techniques specifically designed
//! for ensemble methods, including cross-validation strategies, hyperparameter
//! optimization, and ensemble construction validation.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
    types::Float,
};
use std::collections::HashMap;

/// Cross-validation strategy for ensemble construction
#[derive(Debug, Clone)]
pub enum EnsembleCVStrategy {
    /// Standard k-fold cross-validation
    KFold { n_splits: usize, shuffle: bool },
    /// Stratified k-fold for classification
    StratifiedKFold { n_splits: usize, shuffle: bool },
    /// Leave-one-out cross-validation
    LeaveOneOut,
    /// Time series split (for temporal data)
    TimeSeriesSplit {
        n_splits: usize,
        max_train_size: Option<usize>,
    },
    /// Group-based cross-validation
    GroupKFold { n_splits: usize, groups: Vec<usize> },
    /// Nested cross-validation for ensemble hyperparameter tuning
    NestedCV {
        outer_splits: usize,
        inner_splits: usize,
    },
}

/// Ensemble construction configuration
#[derive(Debug, Clone)]
pub struct EnsembleConstructionConfig {
    /// Cross-validation strategy
    pub cv_strategy: EnsembleCVStrategy,
    /// Scoring metric for model selection
    pub scoring: ScoringMetric,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Whether to use early stopping
    pub early_stopping: bool,
    /// Patience for early stopping
    pub patience: usize,
    /// Minimum improvement for early stopping
    pub min_improvement: Float,
    /// Maximum ensemble size to consider
    pub max_ensemble_size: Option<usize>,
    /// Diversity weight in multi-objective optimization
    pub diversity_weight: Float,
}

/// Scoring metrics for ensemble evaluation
#[derive(Debug, Clone)]
pub enum ScoringMetric {
    Accuracy,
    F1Score,
    AUC,
    MeanSquaredError,
    MeanAbsoluteError,
    R2Score,
    Custom(fn(&Array1<Float>, &Array1<Float>) -> Float),
    MultiObjective {
        accuracy_weight: Float,
        diversity_weight: Float,
    },
}

/// Results from cross-validation ensemble construction
#[derive(Debug)]
pub struct EnsembleCVResults {
    /// Mean score across folds
    pub mean_score: Float,
    /// Standard deviation of scores
    pub std_score: Float,
    /// Individual fold scores
    pub fold_scores: Vec<Float>,
    /// Optimal ensemble configuration
    pub best_config: HashMap<String, Float>,
    /// Ensemble diversity metrics
    pub diversity_metrics: DiversityMetrics,
    /// Training time in seconds
    pub training_time: Float,
}

/// Diversity metrics for ensemble evaluation
#[derive(Debug, Clone)]
pub struct DiversityMetrics {
    /// Average pairwise disagreement
    pub disagreement: Float,
    /// Double fault measure
    pub double_fault: Float,
    /// Q-statistic measure
    pub q_statistic: Float,
    /// Entropy-based diversity
    pub entropy_diversity: Float,
    /// Kohavi-Wolpert variance
    pub kw_variance: Float,
    /// Cohen's kappa coefficient
    pub kappa: Float,
    /// Fleiss' kappa for multiple classifiers
    pub fleiss_kappa: Float,
    /// Interrater reliability statistics
    pub interrater_reliability: InterraterReliability,
}

/// Interrater reliability statistics
#[derive(Debug, Clone)]
pub struct InterraterReliability {
    /// Overall agreement percentage
    pub overall_agreement: Float,
    /// Expected agreement by chance
    pub chance_agreement: Float,
    /// Krippendorff's alpha coefficient
    pub krippendorff_alpha: Float,
    /// Pearson correlation between raters
    pub pearson_correlation: Float,
    /// Weighted kappa for ordinal data
    pub weighted_kappa: Float,
    /// Standard error of kappa
    pub kappa_std_error: Float,
}

/// Cross-validation ensemble constructor
pub struct EnsembleCrossValidator {
    config: EnsembleConstructionConfig,
}

impl EnsembleCrossValidator {
    /// Create a new ensemble cross-validator
    pub fn new(config: EnsembleConstructionConfig) -> Self {
        Self { config }
    }

    /// Perform cross-validation for ensemble construction
    pub fn validate_ensemble<E>(
        &self,
        base_estimator: E,
        x: &Array2<Float>,
        y: &Array1<Float>,
        param_grid: &HashMap<String, Vec<Float>>,
    ) -> Result<EnsembleCVResults>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>> + 'static,
        E::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        let start_time = std::time::Instant::now();

        let folds = self.create_folds(x, y)?;
        let mut fold_scores = Vec::new();
        let mut best_score = Float::NEG_INFINITY;
        let mut best_config = HashMap::new();

        // Grid search over parameters
        for config in self.generate_parameter_combinations(param_grid) {
            let mut config_scores = Vec::new();

            for (train_indices, val_indices) in &folds {
                let (x_train, y_train) = self.subset_data(x, y, train_indices);
                let (x_val, y_val) = self.subset_data(x, y, val_indices);

                // Configure estimator with current parameters
                let estimator = base_estimator.clone();
                // Note: In real implementation, we'd need a way to set parameters

                // Train ensemble on training fold
                let trained_ensemble = self.build_ensemble_fold(estimator, &x_train, &y_train)?;

                // Evaluate on validation fold
                let predictions = trained_ensemble.predict(&x_val)?;
                let score = self.compute_score(&predictions, &y_val);
                config_scores.push(score);
            }

            let mean_score = config_scores.iter().sum::<Float>() / config_scores.len() as Float;
            if mean_score > best_score {
                best_score = mean_score;
                best_config = config.clone();
                fold_scores = config_scores;
            }
        }

        let std_score = self.compute_std(&fold_scores, best_score);
        let diversity_metrics = self.compute_diversity_metrics(x, y)?;
        let training_time = start_time.elapsed().as_secs_f64() as Float;

        Ok(EnsembleCVResults {
            mean_score: best_score,
            std_score,
            fold_scores,
            best_config,
            diversity_metrics,
            training_time,
        })
    }

    /// Create cross-validation folds
    fn create_folds(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let n_samples = x.nrows();
        let mut indices: Vec<usize> = (0..n_samples).collect();

        match &self.config.cv_strategy {
            EnsembleCVStrategy::KFold { n_splits, shuffle } => {
                if *shuffle {
                    if let Some(_seed) = self.config.random_state {
                        let mut rng = thread_rng();
                        // Simple shuffle using Fisher-Yates
                        for i in (1..indices.len()).rev() {
                            let j = rng.gen_range(0..i + 1);
                            indices.swap(i, j);
                        }
                    }
                }

                let fold_size = n_samples / n_splits;
                let mut folds = Vec::new();

                for i in 0..*n_splits {
                    let start = i * fold_size;
                    let end = if i == n_splits - 1 {
                        n_samples
                    } else {
                        (i + 1) * fold_size
                    };

                    let val_indices: Vec<usize> = indices[start..end].to_vec();
                    let train_indices: Vec<usize> = indices[..start]
                        .iter()
                        .chain(indices[end..].iter())
                        .copied()
                        .collect();

                    folds.push((train_indices, val_indices));
                }

                Ok(folds)
            }

            EnsembleCVStrategy::StratifiedKFold {
                n_splits,
                shuffle: _,
            } => {
                // Simplified stratified split - group by class and split each group
                let mut class_indices: HashMap<i32, Vec<usize>> = HashMap::new();

                for (idx, &label) in y.iter().enumerate() {
                    let class = label as i32;
                    class_indices.entry(class).or_default().push(idx);
                }

                let mut folds = vec![Vec::new(); *n_splits];

                for class_idx_vec in class_indices.values() {
                    let class_fold_size = class_idx_vec.len() / n_splits;

                    for (i, &idx) in class_idx_vec.iter().enumerate() {
                        let fold_idx = if class_fold_size == 0 {
                            i % n_splits
                        } else {
                            i / class_fold_size
                        };
                        let fold_idx = fold_idx.min(n_splits - 1);
                        folds[fold_idx].push(idx);
                    }
                }

                let mut cv_folds = Vec::new();
                for i in 0..*n_splits {
                    let val_indices = folds[i].clone();
                    let train_indices: Vec<usize> = folds
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .flat_map(|(_, fold)| fold.iter())
                        .copied()
                        .collect();
                    cv_folds.push((train_indices, val_indices));
                }

                Ok(cv_folds)
            }

            EnsembleCVStrategy::LeaveOneOut => {
                let mut folds = Vec::new();
                for i in 0..n_samples {
                    let val_indices = vec![i];
                    let train_indices: Vec<usize> = (0..n_samples).filter(|&j| j != i).collect();
                    folds.push((train_indices, val_indices));
                }
                Ok(folds)
            }

            EnsembleCVStrategy::TimeSeriesSplit {
                n_splits,
                max_train_size,
            } => {
                let mut folds = Vec::new();
                let min_train_size = n_samples / (n_splits + 1);

                for i in 1..=*n_splits {
                    let train_end = min_train_size * i;
                    let val_start = train_end;
                    let val_end = if i == *n_splits {
                        n_samples
                    } else {
                        min_train_size * (i + 1)
                    };

                    let mut train_start = 0;
                    if let Some(max_size) = max_train_size {
                        train_start = train_end.saturating_sub(*max_size);
                    }

                    let train_indices: Vec<usize> = (train_start..train_end).collect();
                    let val_indices: Vec<usize> = (val_start..val_end).collect();

                    if !train_indices.is_empty() && !val_indices.is_empty() {
                        folds.push((train_indices, val_indices));
                    }
                }

                Ok(folds)
            }

            EnsembleCVStrategy::GroupKFold {
                n_splits: _,
                groups: _,
            } => {
                // Simplified group k-fold implementation
                // In practice, this would group by the provided group labels
                self.create_kfold_simple(n_samples, 5)
            }

            EnsembleCVStrategy::NestedCV {
                outer_splits,
                inner_splits: _,
            } => {
                // Simplified nested CV - just do outer splits for now
                self.create_kfold_simple(n_samples, *outer_splits)
            }
        }
    }

    fn create_kfold_simple(
        &self,
        n_samples: usize,
        n_splits: usize,
    ) -> Result<Vec<(Vec<usize>, Vec<usize>)>> {
        let fold_size = n_samples / n_splits;
        let mut folds = Vec::new();

        for i in 0..n_splits {
            let start = i * fold_size;
            let end = if i == n_splits - 1 {
                n_samples
            } else {
                (i + 1) * fold_size
            };

            let val_indices: Vec<usize> = (start..end).collect();
            let train_indices: Vec<usize> = (0..start).chain(end..n_samples).collect();

            folds.push((train_indices, val_indices));
        }

        Ok(folds)
    }

    /// Generate parameter combinations for grid search
    fn generate_parameter_combinations(
        &self,
        param_grid: &HashMap<String, Vec<Float>>,
    ) -> Vec<HashMap<String, Float>> {
        if param_grid.is_empty() {
            return vec![HashMap::new()];
        }

        let mut combinations = vec![HashMap::new()];

        for (param_name, param_values) in param_grid {
            let mut new_combinations = Vec::new();

            for combination in &combinations {
                for &value in param_values {
                    let mut new_combination = combination.clone();
                    new_combination.insert(param_name.clone(), value);
                    new_combinations.push(new_combination);
                }
            }

            combinations = new_combinations;
        }

        combinations
    }

    /// Subset data based on indices
    fn subset_data(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        indices: &[usize],
    ) -> (Array2<Float>, Array1<Float>) {
        let x_subset = x.select(Axis(0), indices);
        let y_subset = y.select(Axis(0), indices);
        (x_subset, y_subset)
    }

    /// Build ensemble for a single fold (simplified)
    fn build_ensemble_fold<E>(
        &self,
        _estimator: E,
        _x_train: &Array2<Float>,
        _y_train: &Array1<Float>,
    ) -> Result<MockEnsemble>
    where
        E: Fit<Array2<Float>, Array1<Float>>,
    {
        // In a real implementation, this would build an actual ensemble
        // For now, return a mock ensemble
        Ok(MockEnsemble {
            mean_prediction: 0.5,
        })
    }

    /// Compute score based on configured metric
    fn compute_score(&self, predictions: &Array1<Float>, y_true: &Array1<Float>) -> Float {
        match &self.config.scoring {
            ScoringMetric::Accuracy => {
                let correct = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| {
                        if (pred - true_val).abs() < 0.5 {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<Float>();
                correct / predictions.len() as Float
            }

            ScoringMetric::MeanSquaredError => {
                let mse = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).powi(2))
                    .sum::<Float>()
                    / predictions.len() as Float;
                -mse // Negative because we want to maximize
            }

            ScoringMetric::MeanAbsoluteError => {
                let mae = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred - true_val).abs())
                    .sum::<Float>()
                    / predictions.len() as Float;
                -mae // Negative because we want to maximize
            }

            ScoringMetric::R2Score => {
                let y_mean = y_true.sum() / y_true.len() as Float;
                let ss_tot = y_true.iter().map(|&y| (y - y_mean).powi(2)).sum::<Float>();
                let ss_res = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (true_val - pred).powi(2))
                    .sum::<Float>();

                if ss_tot == 0.0 {
                    1.0
                } else {
                    1.0 - (ss_res / ss_tot)
                }
            }

            ScoringMetric::F1Score => {
                // Simplified binary F1 calculation
                let mut tp = 0.0;
                let mut fp = 0.0;
                let mut fn_count = 0.0;

                for (&pred, &true_val) in predictions.iter().zip(y_true.iter()) {
                    let pred_class = if pred > 0.5 { 1.0 } else { 0.0 };
                    let true_class = if true_val > 0.5 { 1.0 } else { 0.0 };

                    if pred_class == 1.0 && true_class == 1.0 {
                        tp += 1.0;
                    } else if pred_class == 1.0 && true_class == 0.0 {
                        fp += 1.0;
                    } else if pred_class == 0.0 && true_class == 1.0 {
                        fn_count += 1.0;
                    }
                }

                let precision = if tp + fp > 0.0 { tp / (tp + fp) } else { 0.0 };
                let recall = if tp + fn_count > 0.0 {
                    tp / (tp + fn_count)
                } else {
                    0.0
                };

                if precision + recall > 0.0 {
                    2.0 * precision * recall / (precision + recall)
                } else {
                    0.0
                }
            }

            ScoringMetric::AUC => {
                // Simplified AUC calculation (trapezoid rule approximation)
                let mut pairs: Vec<(Float, Float)> = predictions
                    .iter()
                    .zip(y_true.iter())
                    .map(|(&pred, &true_val)| (pred, true_val))
                    .collect();

                pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

                let mut auc = 0.0;
                let mut tp = 0.0;
                let mut fp = 0.0;
                let total_pos = y_true.iter().filter(|&&y| y > 0.5).count() as Float;
                let total_neg = y_true.len() as Float - total_pos;

                if total_pos == 0.0 || total_neg == 0.0 {
                    return 0.5; // Random classifier
                }

                for (_, true_val) in pairs {
                    if true_val > 0.5 {
                        tp += 1.0;
                    } else {
                        auc += tp / total_pos;
                        fp += 1.0;
                    }
                }

                auc / total_neg
            }

            ScoringMetric::Custom(scorer) => scorer(predictions, y_true),

            ScoringMetric::MultiObjective {
                accuracy_weight,
                diversity_weight,
            } => {
                let accuracy =
                    self.compute_score_with_metric(predictions, y_true, &ScoringMetric::Accuracy);
                let diversity = 0.5; // Placeholder - would compute actual diversity
                accuracy_weight * accuracy + diversity_weight * diversity
            }
        }
    }

    fn compute_score_with_metric(
        &self,
        predictions: &Array1<Float>,
        y_true: &Array1<Float>,
        metric: &ScoringMetric,
    ) -> Float {
        let temp_config = EnsembleConstructionConfig {
            cv_strategy: self.config.cv_strategy.clone(),
            scoring: metric.clone(),
            random_state: self.config.random_state,
            early_stopping: self.config.early_stopping,
            patience: self.config.patience,
            min_improvement: self.config.min_improvement,
            max_ensemble_size: self.config.max_ensemble_size,
            diversity_weight: self.config.diversity_weight,
        };
        let temp_validator = EnsembleCrossValidator::new(temp_config);
        temp_validator.compute_score(predictions, y_true)
    }

    /// Compute standard deviation of scores
    fn compute_std(&self, scores: &[Float], mean: Float) -> Float {
        if scores.len() <= 1 {
            return 0.0;
        }

        let variance = scores
            .iter()
            .map(|&score| (score - mean).powi(2))
            .sum::<Float>()
            / (scores.len() - 1) as Float;

        variance.sqrt()
    }

    /// Compute ensemble diversity metrics
    fn compute_diversity_metrics(
        &self,
        _x: &Array2<Float>,
        _y: &Array1<Float>,
    ) -> Result<DiversityMetrics> {
        // Placeholder implementation - in practice, would compute from ensemble predictions
        Ok(DiversityMetrics {
            disagreement: 0.3,
            double_fault: 0.1,
            q_statistic: 0.2,
            entropy_diversity: 0.4,
            kw_variance: 0.25,
            kappa: 0.6,
            fleiss_kappa: 0.65,
            interrater_reliability: InterraterReliability {
                overall_agreement: 0.7,
                chance_agreement: 0.4,
                krippendorff_alpha: 0.55,
                pearson_correlation: 0.8,
                weighted_kappa: 0.6,
                kappa_std_error: 0.05,
            },
        })
    }
}

/// Mock ensemble for testing
struct MockEnsemble {
    mean_prediction: Float,
}

impl Predict<Array2<Float>, Array1<Float>> for MockEnsemble {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        Ok(Array1::from_elem(x.nrows(), self.mean_prediction))
    }
}

impl Default for EnsembleConstructionConfig {
    fn default() -> Self {
        Self {
            cv_strategy: EnsembleCVStrategy::KFold {
                n_splits: 5,
                shuffle: true,
            },
            scoring: ScoringMetric::Accuracy,
            random_state: None,
            early_stopping: false,
            patience: 3,
            min_improvement: 1e-4,
            max_ensemble_size: None,
            diversity_weight: 0.1,
        }
    }
}

/// Convenience functions for common ensemble cross-validation setups
impl EnsembleCrossValidator {
    /// Create a classifier cross-validator with accuracy scoring
    pub fn for_classification(n_splits: usize) -> Self {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::StratifiedKFold {
                n_splits,
                shuffle: true,
            },
            scoring: ScoringMetric::Accuracy,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a regressor cross-validator with R² scoring
    pub fn for_regression(n_splits: usize) -> Self {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::KFold {
                n_splits,
                shuffle: true,
            },
            scoring: ScoringMetric::R2Score,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a time series cross-validator
    pub fn for_time_series(n_splits: usize, max_train_size: Option<usize>) -> Self {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::TimeSeriesSplit {
                n_splits,
                max_train_size,
            },
            scoring: ScoringMetric::MeanSquaredError,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create a multi-objective cross-validator (accuracy + diversity)
    pub fn multi_objective(
        n_splits: usize,
        accuracy_weight: Float,
        diversity_weight: Float,
    ) -> Self {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::StratifiedKFold {
                n_splits,
                shuffle: true,
            },
            scoring: ScoringMetric::MultiObjective {
                accuracy_weight,
                diversity_weight,
            },
            diversity_weight,
            ..Default::default()
        };
        Self::new(config)
    }
}

/// Diversity analysis for ensemble methods
pub struct DiversityAnalyzer;

impl DiversityAnalyzer {
    /// Compute comprehensive diversity metrics including kappa statistics
    pub fn compute_diversity_metrics(
        predictions: &[Array1<Float>],
        ground_truth: &Array1<Float>,
    ) -> Result<DiversityMetrics> {
        if predictions.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No predictions provided".to_string(),
            ));
        }

        let n_classifiers = predictions.len();
        let n_samples = predictions[0].len();

        // Validate that all predictions have the same length
        for pred in predictions {
            if pred.len() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "Inconsistent prediction lengths".to_string(),
                ));
            }
        }

        // Convert predictions to integer classes for discrete analysis
        let class_predictions: Vec<Vec<i32>> = predictions
            .iter()
            .map(|pred| pred.iter().map(|&p| p.round() as i32).collect())
            .collect();

        // Compute basic diversity measures
        let disagreement = Self::compute_disagreement(&class_predictions);
        let double_fault = Self::compute_double_fault(&class_predictions, ground_truth);
        let q_statistic = Self::compute_q_statistic(&class_predictions);
        let entropy_diversity = Self::compute_entropy_diversity(&class_predictions);
        let kw_variance = Self::compute_kw_variance(&class_predictions);

        // Compute kappa statistics
        let kappa = Self::compute_cohens_kappa(&class_predictions)?;
        let fleiss_kappa = Self::compute_fleiss_kappa(&class_predictions)?;
        let interrater_reliability = Self::compute_interrater_reliability(&class_predictions)?;

        Ok(DiversityMetrics {
            disagreement,
            double_fault,
            q_statistic,
            entropy_diversity,
            kw_variance,
            kappa,
            fleiss_kappa,
            interrater_reliability,
        })
    }

    /// Compute average pairwise disagreement
    fn compute_disagreement(predictions: &[Vec<i32>]) -> Float {
        let n_classifiers = predictions.len();
        if n_classifiers < 2 {
            return 0.0;
        }

        let mut total_disagreement = 0.0;
        let mut pair_count = 0;

        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                let disagreement_ij = predictions[i]
                    .iter()
                    .zip(&predictions[j])
                    .map(|(&pred_i, &pred_j)| if pred_i != pred_j { 1.0 } else { 0.0 })
                    .sum::<Float>()
                    / predictions[i].len() as Float;

                total_disagreement += disagreement_ij;
                pair_count += 1;
            }
        }

        total_disagreement / pair_count as Float
    }

    /// Compute double fault measure
    fn compute_double_fault(predictions: &[Vec<i32>], ground_truth: &Array1<Float>) -> Float {
        let n_classifiers = predictions.len();
        if n_classifiers < 2 {
            return 0.0;
        }

        let ground_truth_int: Vec<i32> = ground_truth.iter().map(|&y| y.round() as i32).collect();
        let mut total_double_fault = 0.0;
        let mut pair_count = 0;

        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                let double_fault_ij = predictions[i]
                    .iter()
                    .zip(&predictions[j])
                    .zip(&ground_truth_int)
                    .map(|((&pred_i, &pred_j), &true_label)| {
                        if pred_i != true_label && pred_j != true_label {
                            1.0
                        } else {
                            0.0
                        }
                    })
                    .sum::<Float>()
                    / predictions[i].len() as Float;

                total_double_fault += double_fault_ij;
                pair_count += 1;
            }
        }

        total_double_fault / pair_count as Float
    }

    /// Compute Q-statistic measure
    fn compute_q_statistic(predictions: &[Vec<i32>]) -> Float {
        let n_classifiers = predictions.len();
        if n_classifiers < 2 {
            return 0.0;
        }

        let mut total_q = 0.0;
        let mut pair_count = 0;

        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                let mut n11 = 0.0; // Both correct
                let mut n10 = 0.0; // i correct, j wrong
                let mut n01 = 0.0; // i wrong, j correct
                let mut n00 = 0.0; // Both wrong

                for k in 0..predictions[i].len() {
                    // For simplicity, assume binary classification with class 1 as positive
                    let i_correct = predictions[i][k] == 1;
                    let j_correct = predictions[j][k] == 1;

                    match (i_correct, j_correct) {
                        (true, true) => n11 += 1.0,
                        (true, false) => n10 += 1.0,
                        (false, true) => n01 += 1.0,
                        (false, false) => n00 += 1.0,
                    }
                }

                let denominator = (n11 * n00) + (n01 * n10);
                let q_ij = if denominator != 0.0 {
                    ((n11 * n00) - (n01 * n10)) / denominator
                } else {
                    0.0
                };

                total_q += q_ij;
                pair_count += 1;
            }
        }

        total_q / pair_count as Float
    }

    /// Compute entropy-based diversity
    fn compute_entropy_diversity(predictions: &[Vec<i32>]) -> Float {
        let n_samples = predictions[0].len();
        let n_classifiers = predictions.len();

        let mut total_entropy = 0.0;

        for sample_idx in 0..n_samples {
            // Count predictions for each class for this sample
            let mut class_counts = std::collections::HashMap::new();

            for classifier_idx in 0..n_classifiers {
                let prediction = predictions[classifier_idx][sample_idx];
                *class_counts.entry(prediction).or_insert(0) += 1;
            }

            // Compute entropy for this sample
            let mut sample_entropy = 0.0;
            for &count in class_counts.values() {
                let probability = count as Float / n_classifiers as Float;
                if probability > 0.0 {
                    sample_entropy -= probability * probability.ln();
                }
            }

            total_entropy += sample_entropy;
        }

        total_entropy / n_samples as Float
    }

    /// Compute Kohavi-Wolpert variance
    fn compute_kw_variance(predictions: &[Vec<i32>]) -> Float {
        let n_samples = predictions[0].len();
        let n_classifiers = predictions.len();

        let mut total_variance = 0.0;

        for sample_idx in 0..n_samples {
            // Compute variance of predictions for this sample
            let sample_predictions: Vec<Float> = (0..n_classifiers)
                .map(|i| predictions[i][sample_idx] as Float)
                .collect();

            let mean = sample_predictions.iter().sum::<Float>() / n_classifiers as Float;
            let variance = sample_predictions
                .iter()
                .map(|&pred| (pred - mean).powi(2))
                .sum::<Float>()
                / n_classifiers as Float;

            total_variance += variance;
        }

        total_variance / n_samples as Float
    }

    /// Compute Cohen's kappa coefficient (average pairwise)
    fn compute_cohens_kappa(predictions: &[Vec<i32>]) -> Result<Float> {
        let n_classifiers = predictions.len();
        if n_classifiers < 2 {
            return Ok(0.0);
        }

        let mut total_kappa = 0.0;
        let mut pair_count = 0;

        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                let kappa_ij = Self::compute_pairwise_kappa(&predictions[i], &predictions[j])?;
                total_kappa += kappa_ij;
                pair_count += 1;
            }
        }

        Ok(total_kappa / pair_count as Float)
    }

    /// Compute pairwise Cohen's kappa
    fn compute_pairwise_kappa(pred1: &[i32], pred2: &[i32]) -> Result<Float> {
        if pred1.len() != pred2.len() {
            return Err(SklearsError::InvalidInput(
                "Predictions must have same length".to_string(),
            ));
        }

        let n = pred1.len();
        if n == 0 {
            return Ok(0.0);
        }

        // Get unique classes
        let mut classes = std::collections::HashSet::new();
        for &pred in pred1.iter().chain(pred2.iter()) {
            classes.insert(pred);
        }
        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Build confusion matrix
        let mut confusion_matrix = vec![vec![0; n_classes]; n_classes];
        for k in 0..n {
            let i = classes.iter().position(|&c| c == pred1[k]).unwrap();
            let j = classes.iter().position(|&c| c == pred2[k]).unwrap();
            confusion_matrix[i][j] += 1;
        }

        // Compute observed agreement
        let observed_agreement = (0..n_classes)
            .map(|i| confusion_matrix[i][i])
            .sum::<usize>() as Float
            / n as Float;

        // Compute expected agreement
        let mut expected_agreement = 0.0;
        for i in 0..n_classes {
            let row_sum: usize = confusion_matrix[i].iter().sum();
            let col_sum: usize = (0..n_classes).map(|j| confusion_matrix[j][i]).sum();
            expected_agreement += (row_sum as Float * col_sum as Float) / (n * n) as Float;
        }

        // Compute kappa
        let kappa = if expected_agreement == 1.0 {
            1.0 // Perfect agreement
        } else {
            (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        };

        Ok(kappa)
    }

    /// Compute Fleiss' kappa for multiple classifiers
    fn compute_fleiss_kappa(predictions: &[Vec<i32>]) -> Result<Float> {
        let n_classifiers = predictions.len();
        let n_samples = predictions[0].len();

        if n_classifiers < 2 || n_samples == 0 {
            return Ok(0.0);
        }

        // Get unique classes
        let mut classes = std::collections::HashSet::new();
        for pred_vec in predictions {
            for &pred in pred_vec {
                classes.insert(pred);
            }
        }
        let classes: Vec<i32> = classes.into_iter().collect();
        let n_classes = classes.len();

        // Compute agreement matrix (samples x classes)
        let mut agreement_matrix = vec![vec![0; n_classes]; n_samples];
        for sample_idx in 0..n_samples {
            for classifier_idx in 0..n_classifiers {
                let prediction = predictions[classifier_idx][sample_idx];
                let class_idx = classes.iter().position(|&c| c == prediction).unwrap();
                agreement_matrix[sample_idx][class_idx] += 1;
            }
        }

        // Compute observed agreement
        let mut total_agreement = 0.0;
        for sample_idx in 0..n_samples {
            let mut sample_agreement = 0.0;
            for class_idx in 0..n_classes {
                let n_ij = agreement_matrix[sample_idx][class_idx] as Float;
                sample_agreement += n_ij * (n_ij - 1.0);
            }
            total_agreement += sample_agreement;
        }
        let observed_agreement =
            total_agreement / (n_samples * n_classifiers * (n_classifiers - 1)) as Float;

        // Compute expected agreement
        let mut expected_agreement = 0.0;
        for class_idx in 0..n_classes {
            let class_proportion = agreement_matrix
                .iter()
                .map(|row| row[class_idx])
                .sum::<usize>() as Float
                / (n_samples * n_classifiers) as Float;
            expected_agreement += class_proportion * class_proportion;
        }

        // Compute Fleiss' kappa
        let fleiss_kappa = if expected_agreement == 1.0 {
            1.0
        } else {
            (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
        };

        Ok(fleiss_kappa)
    }

    /// Compute comprehensive interrater reliability statistics
    fn compute_interrater_reliability(predictions: &[Vec<i32>]) -> Result<InterraterReliability> {
        let n_classifiers = predictions.len();
        let n_samples = predictions[0].len();

        if n_classifiers < 2 || n_samples == 0 {
            return Ok(InterraterReliability {
                overall_agreement: 0.0,
                chance_agreement: 0.0,
                krippendorff_alpha: 0.0,
                pearson_correlation: 0.0,
                weighted_kappa: 0.0,
                kappa_std_error: 0.0,
            });
        }

        // Overall agreement (exact match across all classifiers)
        let mut exact_agreements = 0;
        for sample_idx in 0..n_samples {
            let first_pred = predictions[0][sample_idx];
            let all_agree = predictions
                .iter()
                .all(|pred| pred[sample_idx] == first_pred);
            if all_agree {
                exact_agreements += 1;
            }
        }
        let overall_agreement = exact_agreements as Float / n_samples as Float;

        // Chance agreement (assuming random classification)
        let mut class_counts = std::collections::HashMap::new();
        for pred_vec in predictions {
            for &pred in pred_vec {
                *class_counts.entry(pred).or_insert(0) += 1;
            }
        }
        let total_predictions = n_classifiers * n_samples;
        let chance_agreement = class_counts
            .values()
            .map(|&count| {
                let prob = count as Float / total_predictions as Float;
                prob.powi(n_classifiers as i32)
            })
            .sum::<Float>();

        // Krippendorff's alpha (simplified nominal version)
        let krippendorff_alpha = Self::compute_krippendorff_alpha(predictions)?;

        // Average pairwise Pearson correlation
        let pearson_correlation = Self::compute_average_correlation(predictions)?;

        // Weighted kappa (for ordinal data, treating as nominal for simplicity)
        let weighted_kappa = Self::compute_cohens_kappa(predictions)?;

        // Standard error of kappa (approximation)
        let kappa_std_error =
            (overall_agreement * (1.0 - overall_agreement) / n_samples as Float).sqrt();

        Ok(InterraterReliability {
            overall_agreement,
            chance_agreement,
            krippendorff_alpha,
            pearson_correlation,
            weighted_kappa,
            kappa_std_error,
        })
    }

    /// Compute Krippendorff's alpha (simplified for nominal data)
    fn compute_krippendorff_alpha(predictions: &[Vec<i32>]) -> Result<Float> {
        let n_classifiers = predictions.len();
        let n_samples = predictions[0].len();

        if n_classifiers < 2 {
            return Ok(0.0);
        }

        // Create pairable values matrix
        let mut observed_disagreements = 0.0;
        let mut expected_disagreements = 0.0;
        let mut total_pairs = 0.0;

        // Simplified calculation for nominal data
        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                for sample_idx in 0..n_samples {
                    let val_i = predictions[i][sample_idx];
                    let val_j = predictions[j][sample_idx];

                    total_pairs += 1.0;

                    if val_i != val_j {
                        observed_disagreements += 1.0;
                    }

                    // For expected disagreements, we assume uniform distribution
                    expected_disagreements += 0.5; // Simplified assumption
                }
            }
        }

        let alpha = if expected_disagreements > 0.0 {
            1.0 - (observed_disagreements / expected_disagreements)
        } else {
            1.0
        };

        Ok(alpha)
    }

    /// Compute average pairwise Pearson correlation
    fn compute_average_correlation(predictions: &[Vec<i32>]) -> Result<Float> {
        let n_classifiers = predictions.len();
        if n_classifiers < 2 {
            return Ok(0.0);
        }

        let mut total_correlation = 0.0;
        let mut pair_count = 0;

        for i in 0..n_classifiers {
            for j in (i + 1)..n_classifiers {
                let corr = Self::compute_pearson_correlation(&predictions[i], &predictions[j])?;
                total_correlation += corr;
                pair_count += 1;
            }
        }

        Ok(total_correlation / pair_count as Float)
    }

    /// Compute Pearson correlation between two prediction vectors
    fn compute_pearson_correlation(pred1: &[i32], pred2: &[i32]) -> Result<Float> {
        if pred1.len() != pred2.len() || pred1.is_empty() {
            return Ok(0.0);
        }

        let n = pred1.len() as Float;
        let mean1 = pred1.iter().sum::<i32>() as Float / n;
        let mean2 = pred2.iter().sum::<i32>() as Float / n;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..pred1.len() {
            let diff1 = pred1[i] as Float - mean1;
            let diff2 = pred2[i] as Float - mean2;

            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        let correlation = if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        };

        Ok(correlation)
    }
}

/// Bias-variance decomposition results
#[derive(Debug, Clone)]
pub struct BiasVarianceDecomposition {
    /// Bias squared component
    pub bias_squared: Float,
    /// Variance component
    pub variance: Float,
    /// Noise (irreducible error) component
    pub noise: Float,
    /// Total expected loss
    pub total_loss: Float,
    /// Number of bootstrap samples used
    pub n_bootstrap_samples: usize,
    /// Decomposition for individual predictions
    pub sample_decompositions: Option<Vec<SampleBiasVariance>>,
}

/// Bias-variance decomposition for a single sample
#[derive(Debug, Clone)]
pub struct SampleBiasVariance {
    /// Sample index
    pub sample_idx: usize,
    /// True target value
    pub true_value: Float,
    /// Mean prediction across bootstrap samples
    pub mean_prediction: Float,
    /// Bias squared for this sample
    pub bias_squared: Float,
    /// Variance for this sample
    pub variance: Float,
    /// Individual bootstrap predictions
    pub bootstrap_predictions: Vec<Float>,
}

/// Configuration for bias-variance decomposition
#[derive(Debug, Clone)]
pub struct BiasVarianceConfig {
    /// Number of bootstrap samples
    pub n_bootstrap_samples: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Bootstrap sample size (as fraction of original data)
    pub bootstrap_size: Float,
    /// Whether to compute sample-level decompositions
    pub compute_sample_level: bool,
    /// Loss function for decomposition
    pub loss_function: ModelSelectionLossFunction,
}

/// Loss functions for bias-variance decomposition
#[derive(Debug, Clone)]
pub enum ModelSelectionLossFunction {
    /// Mean squared error (for regression)
    SquaredLoss,
    /// Zero-one loss (for classification)
    ZeroOneLoss,
    /// Custom loss function
    Custom(fn(Float, Float) -> Float),
}

/// Bias-variance decomposition analyzer
pub struct BiasVarianceAnalyzer {
    config: BiasVarianceConfig,
}

impl BiasVarianceAnalyzer {
    /// Create a new bias-variance analyzer
    pub fn new(config: BiasVarianceConfig) -> Self {
        Self { config }
    }

    /// Perform bias-variance decomposition for an estimator
    pub fn decompose<E>(
        &self,
        estimator: E,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BiasVarianceDecomposition>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>>,
        E::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        let n_samples = x.nrows();
        let n_bootstrap = self.config.n_bootstrap_samples;

        // Store all bootstrap predictions
        let mut all_predictions = Vec::with_capacity(n_bootstrap);

        // Generate bootstrap samples and train models
        for bootstrap_idx in 0..n_bootstrap {
            let (x_bootstrap, y_bootstrap) = self.generate_bootstrap_sample(x, y, bootstrap_idx)?;

            // Train estimator on bootstrap sample
            let trained_estimator = estimator.clone().fit(&x_bootstrap, &y_bootstrap)?;

            // Make predictions on original test set
            let predictions = trained_estimator.predict(x)?;
            all_predictions.push(predictions);
        }

        // Compute bias-variance decomposition
        let decomposition = self.compute_decomposition(&all_predictions, y)?;

        Ok(decomposition)
    }

    /// Perform ensemble bias-variance decomposition
    pub fn decompose_ensemble<E>(
        &self,
        base_estimator: E,
        ensemble_sizes: &[usize],
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<Vec<(usize, BiasVarianceDecomposition)>>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>>,
        E::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        let mut results = Vec::new();

        for &ensemble_size in ensemble_sizes {
            let ensemble_config = BiasVarianceConfig {
                n_bootstrap_samples: self.config.n_bootstrap_samples,
                random_state: self.config.random_state,
                bootstrap_size: self.config.bootstrap_size,
                compute_sample_level: self.config.compute_sample_level,
                loss_function: self.config.loss_function.clone(),
            };

            let ensemble_analyzer = BiasVarianceAnalyzer::new(ensemble_config);

            // Create simple ensemble by averaging multiple base estimators
            let ensemble_decomposition = ensemble_analyzer.decompose_bagged_ensemble(
                base_estimator.clone(),
                ensemble_size,
                x,
                y,
            )?;

            results.push((ensemble_size, ensemble_decomposition));
        }

        Ok(results)
    }

    /// Decompose a bagged ensemble (average of multiple estimators)
    fn decompose_bagged_ensemble<E>(
        &self,
        base_estimator: E,
        ensemble_size: usize,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BiasVarianceDecomposition>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>>,
        E::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        let n_samples = x.nrows();
        let n_bootstrap = self.config.n_bootstrap_samples;

        // Store ensemble predictions for each bootstrap sample
        let mut ensemble_predictions = Vec::with_capacity(n_bootstrap);

        for bootstrap_idx in 0..n_bootstrap {
            let (x_bootstrap, y_bootstrap) = self.generate_bootstrap_sample(x, y, bootstrap_idx)?;

            // Train ensemble of base estimators
            let mut member_predictions = Vec::with_capacity(ensemble_size);

            for member_idx in 0..ensemble_size {
                // Create slightly different bootstrap sample for each member
                let seed_offset = bootstrap_idx * ensemble_size + member_idx;
                let (x_member, y_member) =
                    self.generate_bootstrap_sample_with_offset(x, y, seed_offset)?;

                let trained_member = base_estimator.clone().fit(&x_member, &y_member)?;
                let member_pred = trained_member.predict(x)?;
                member_predictions.push(member_pred);
            }

            // Average predictions across ensemble members
            let mut ensemble_pred = Array1::zeros(n_samples);
            for pred in &member_predictions {
                ensemble_pred += pred;
            }
            ensemble_pred /= ensemble_size as Float;

            ensemble_predictions.push(ensemble_pred);
        }

        // Compute bias-variance decomposition for the ensemble
        let decomposition = self.compute_decomposition(&ensemble_predictions, y)?;

        Ok(decomposition)
    }

    /// Generate bootstrap sample with optional seed offset
    fn generate_bootstrap_sample_with_offset(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        seed_offset: usize,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n_samples = x.nrows();
        let bootstrap_size = (n_samples as Float * self.config.bootstrap_size) as usize;

        let mut rng = thread_rng();
        let mut indices = Vec::with_capacity(bootstrap_size);

        // Generate bootstrap indices with replacement
        for _ in 0..bootstrap_size {
            let idx = rng.gen_range(0..n_samples);
            indices.push(idx);
        }

        // Create bootstrap sample
        let x_bootstrap = x.select(Axis(0), &indices);
        let y_bootstrap = y.select(Axis(0), &indices);

        Ok((x_bootstrap, y_bootstrap))
    }

    /// Generate bootstrap sample
    fn generate_bootstrap_sample(
        &self,
        x: &Array2<Float>,
        y: &Array1<Float>,
        _bootstrap_idx: usize,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        self.generate_bootstrap_sample_with_offset(x, y, 0)
    }

    /// Compute bias-variance decomposition from predictions
    fn compute_decomposition(
        &self,
        all_predictions: &[Array1<Float>],
        y_true: &Array1<Float>,
    ) -> Result<BiasVarianceDecomposition> {
        let n_samples = y_true.len();
        let n_bootstrap = all_predictions.len();

        if n_bootstrap == 0 {
            return Err(SklearsError::InvalidInput(
                "No predictions provided".to_string(),
            ));
        }

        // Compute mean prediction for each sample across bootstrap iterations
        let mut mean_predictions = Array1::zeros(n_samples);
        for predictions in all_predictions {
            mean_predictions += predictions;
        }
        mean_predictions /= n_bootstrap as Float;

        // Compute sample-level decompositions if requested
        let sample_decompositions = if self.config.compute_sample_level {
            let mut decomps = Vec::with_capacity(n_samples);

            for sample_idx in 0..n_samples {
                let true_value = y_true[sample_idx];
                let mean_pred = mean_predictions[sample_idx];

                // Collect all bootstrap predictions for this sample
                let bootstrap_preds: Vec<Float> = all_predictions
                    .iter()
                    .map(|pred| pred[sample_idx])
                    .collect();

                // Compute bias squared
                let bias_squared = self.compute_loss(mean_pred, true_value);

                // Compute variance
                let variance = bootstrap_preds
                    .iter()
                    .map(|&pred| self.compute_loss(pred, mean_pred))
                    .sum::<Float>()
                    / n_bootstrap as Float;

                decomps.push(SampleBiasVariance {
                    sample_idx,
                    true_value,
                    mean_prediction: mean_pred,
                    bias_squared,
                    variance,
                    bootstrap_predictions: bootstrap_preds,
                });
            }

            Some(decomps)
        } else {
            None
        };

        // Compute overall bias, variance, and noise
        let mut total_bias_squared = 0.0;
        let mut total_variance = 0.0;
        let mut total_loss = 0.0;

        for sample_idx in 0..n_samples {
            let true_value = y_true[sample_idx];
            let mean_pred = mean_predictions[sample_idx];

            // Bias squared: (mean_prediction - true_value)^2
            let bias_squared = self.compute_loss(mean_pred, true_value);
            total_bias_squared += bias_squared;

            // Variance: E[(prediction - mean_prediction)^2]
            let sample_variance = all_predictions
                .iter()
                .map(|pred| self.compute_loss(pred[sample_idx], mean_pred))
                .sum::<Float>()
                / n_bootstrap as Float;
            total_variance += sample_variance;

            // Total loss: E[(prediction - true_value)^2]
            let sample_loss = all_predictions
                .iter()
                .map(|pred| self.compute_loss(pred[sample_idx], true_value))
                .sum::<Float>()
                / n_bootstrap as Float;
            total_loss += sample_loss;
        }

        // Average over all samples
        total_bias_squared /= n_samples as Float;
        total_variance /= n_samples as Float;
        total_loss /= n_samples as Float;

        // Noise = Total Loss - Bias^2 - Variance (for squared loss)
        let noise = match self.config.loss_function {
            ModelSelectionLossFunction::SquaredLoss => {
                (total_loss - total_bias_squared - total_variance).max(0.0)
            }
            _ => 0.0, // For other loss functions, noise computation is more complex
        };

        Ok(BiasVarianceDecomposition {
            bias_squared: total_bias_squared,
            variance: total_variance,
            noise,
            total_loss,
            n_bootstrap_samples: n_bootstrap,
            sample_decompositions,
        })
    }

    /// Compute loss according to the configured loss function
    fn compute_loss(&self, prediction: Float, true_value: Float) -> Float {
        match &self.config.loss_function {
            ModelSelectionLossFunction::SquaredLoss => (prediction - true_value).powi(2),
            ModelSelectionLossFunction::ZeroOneLoss => {
                if (prediction - true_value).abs() < 0.5 {
                    0.0
                } else {
                    1.0
                }
            }
            ModelSelectionLossFunction::Custom(loss_fn) => loss_fn(prediction, true_value),
        }
    }

    /// Analyze how ensemble size affects bias-variance tradeoff
    pub fn analyze_ensemble_size_effect<E>(
        &self,
        base_estimator: E,
        max_ensemble_size: usize,
        x: &Array2<Float>,
        y: &Array1<Float>,
    ) -> Result<BiasVarianceEnsembleSizeAnalysis>
    where
        E: Clone + Fit<Array2<Float>, Array1<Float>>,
        E::Fitted: Predict<Array2<Float>, Array1<Float>>,
    {
        let ensemble_sizes: Vec<usize> = (1..=max_ensemble_size)
            .step_by(if max_ensemble_size > 20 {
                max_ensemble_size / 20
            } else {
                1
            })
            .collect();

        let decompositions = self.decompose_ensemble(base_estimator, &ensemble_sizes, x, y)?;

        let bias_curve: Vec<Float> = decompositions
            .iter()
            .map(|(_, decomp)| decomp.bias_squared)
            .collect();

        let variance_curve: Vec<Float> = decompositions
            .iter()
            .map(|(_, decomp)| decomp.variance)
            .collect();

        let total_error_curve: Vec<Float> = decompositions
            .iter()
            .map(|(_, decomp)| decomp.total_loss)
            .collect();

        // Find optimal ensemble size (minimum total error)
        let optimal_idx = total_error_curve
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(idx, _)| idx)
            .unwrap_or(0);

        let optimal_size = ensemble_sizes[optimal_idx];

        let bias_reduction = bias_curve[0] - bias_curve.last().unwrap_or(&bias_curve[0]);
        let variance_reduction =
            variance_curve[0] - variance_curve.last().unwrap_or(&variance_curve[0]);

        Ok(BiasVarianceEnsembleSizeAnalysis {
            ensemble_sizes,
            bias_curve,
            variance_curve,
            total_error_curve,
            optimal_ensemble_size: optimal_size,
            bias_reduction,
            variance_reduction,
        })
    }
}

/// Analysis of ensemble size effects on bias-variance tradeoff
#[derive(Debug)]
pub struct BiasVarianceEnsembleSizeAnalysis {
    /// Ensemble sizes analyzed
    pub ensemble_sizes: Vec<usize>,
    /// Bias squared at each ensemble size
    pub bias_curve: Vec<Float>,
    /// Variance at each ensemble size
    pub variance_curve: Vec<Float>,
    /// Total error at each ensemble size
    pub total_error_curve: Vec<Float>,
    /// Optimal ensemble size (minimum error)
    pub optimal_ensemble_size: usize,
    /// Total bias reduction from single model to largest ensemble
    pub bias_reduction: Float,
    /// Total variance reduction from single model to largest ensemble
    pub variance_reduction: Float,
}

impl Default for BiasVarianceConfig {
    fn default() -> Self {
        Self {
            n_bootstrap_samples: 100,
            random_state: None,
            bootstrap_size: 1.0,
            compute_sample_level: false,
            loss_function: ModelSelectionLossFunction::SquaredLoss,
        }
    }
}

/// Convenience methods for bias-variance analysis
impl BiasVarianceAnalyzer {
    /// Create analyzer for regression problems
    pub fn for_regression(n_bootstrap_samples: usize) -> Self {
        let config = BiasVarianceConfig {
            n_bootstrap_samples,
            loss_function: ModelSelectionLossFunction::SquaredLoss,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create analyzer for classification problems
    pub fn for_classification(n_bootstrap_samples: usize) -> Self {
        let config = BiasVarianceConfig {
            n_bootstrap_samples,
            loss_function: ModelSelectionLossFunction::ZeroOneLoss,
            ..Default::default()
        };
        Self::new(config)
    }

    /// Create analyzer with detailed sample-level analysis
    pub fn with_sample_analysis(n_bootstrap_samples: usize) -> Self {
        let config = BiasVarianceConfig {
            n_bootstrap_samples,
            compute_sample_level: true,
            ..Default::default()
        };
        Self::new(config)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ensemble_cv_config_creation() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);

        // Test that we can create the cross-validator
        assert!(true);
    }

    #[test]
    fn test_kfold_creation() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::KFold {
                n_splits: 3,
                shuffle: false,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);

        let x = array![
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
            [7.0, 8.0],
            [9.0, 10.0],
            [11.0, 12.0]
        ];
        let y = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];

        let folds = cv.create_folds(&x, &y).unwrap();
        assert_eq!(folds.len(), 3);

        // Check that each fold has reasonable sizes
        for (train_indices, val_indices) in &folds {
            assert!(!train_indices.is_empty());
            assert!(!val_indices.is_empty());
            assert_eq!(train_indices.len() + val_indices.len(), 6);
        }
    }

    #[test]
    fn test_stratified_kfold() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::StratifiedKFold {
                n_splits: 2,
                shuffle: false,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 0.0, 1.0, 1.0]; // Balanced classes

        let folds = cv.create_folds(&x, &y).unwrap();
        assert_eq!(folds.len(), 2);
    }

    #[test]
    fn test_scoring_metrics() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);

        let predictions = array![0.0, 1.0, 1.0, 0.0];
        let y_true = array![0.0, 1.0, 0.0, 0.0];

        let accuracy = cv.compute_score(&predictions, &y_true);
        assert!(accuracy >= 0.0 && accuracy <= 1.0);
    }

    #[test]
    fn test_parameter_combinations() {
        let config = EnsembleConstructionConfig::default();
        let cv = EnsembleCrossValidator::new(config);

        let mut param_grid = HashMap::new();
        param_grid.insert("learning_rate".to_string(), vec![0.01, 0.1, 1.0]);
        param_grid.insert("n_estimators".to_string(), vec![10.0, 50.0]);

        let combinations = cv.generate_parameter_combinations(&param_grid);
        assert_eq!(combinations.len(), 6); // 3 * 2 = 6 combinations

        // Check that each combination has both parameters
        for combo in &combinations {
            assert!(combo.contains_key("learning_rate"));
            assert!(combo.contains_key("n_estimators"));
        }
    }

    #[test]
    fn test_convenience_constructors() {
        let _classifier_cv = EnsembleCrossValidator::for_classification(5);
        let _regressor_cv = EnsembleCrossValidator::for_regression(3);
        let _ts_cv = EnsembleCrossValidator::for_time_series(4, Some(100));
        let _mo_cv = EnsembleCrossValidator::multi_objective(5, 0.7, 0.3);
    }

    #[test]
    fn test_time_series_split() {
        let config = EnsembleConstructionConfig {
            cv_strategy: EnsembleCVStrategy::TimeSeriesSplit {
                n_splits: 3,
                max_train_size: None,
            },
            ..Default::default()
        };
        let cv = EnsembleCrossValidator::new(config);

        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

        let folds = cv.create_folds(&x, &y).unwrap();

        // Time series splits should have increasing training set sizes
        let mut prev_train_size = 0;
        for (train_indices, _) in &folds {
            assert!(train_indices.len() >= prev_train_size);
            prev_train_size = train_indices.len();
        }
    }

    #[test]
    fn test_bias_variance_config() {
        let config = BiasVarianceConfig::default();
        assert_eq!(config.n_bootstrap_samples, 100);
        assert_eq!(config.bootstrap_size, 1.0);
        assert!(!config.compute_sample_level);

        let analyzer = BiasVarianceAnalyzer::new(config);
        // Just test creation
        assert!(true);
    }

    #[test]
    fn test_bias_variance_convenience_constructors() {
        let _regression_analyzer = BiasVarianceAnalyzer::for_regression(50);
        let _classification_analyzer = BiasVarianceAnalyzer::for_classification(30);
        let _sample_analyzer = BiasVarianceAnalyzer::with_sample_analysis(20);
    }

    #[test]
    fn test_loss_functions() {
        let config = BiasVarianceConfig {
            loss_function: ModelSelectionLossFunction::SquaredLoss,
            ..Default::default()
        };
        let analyzer = BiasVarianceAnalyzer::new(config);

        // Test squared loss
        let squared_loss = analyzer.compute_loss(2.0, 1.0);
        assert_eq!(squared_loss, 1.0);

        // Test zero-one loss
        let config_01 = BiasVarianceConfig {
            loss_function: ModelSelectionLossFunction::ZeroOneLoss,
            ..Default::default()
        };
        let analyzer_01 = BiasVarianceAnalyzer::new(config_01);

        let correct_pred = analyzer_01.compute_loss(1.0, 1.0);
        assert_eq!(correct_pred, 0.0);

        let wrong_pred = analyzer_01.compute_loss(1.0, 0.0);
        assert_eq!(wrong_pred, 1.0);
    }

    #[test]
    fn test_bootstrap_sample_generation() {
        let config = BiasVarianceConfig::default();
        let analyzer = BiasVarianceAnalyzer::new(config);

        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let (x_bootstrap, y_bootstrap) = analyzer.generate_bootstrap_sample(&x, &y, 0).unwrap();

        // Bootstrap sample should have same dimensions as original
        assert_eq!(x_bootstrap.ncols(), x.ncols());
        assert_eq!(y_bootstrap.len(), x_bootstrap.nrows());
    }

    #[test]
    fn test_bias_variance_decomposition_structure() {
        // Test the structure of BiasVarianceDecomposition
        let decomp = BiasVarianceDecomposition {
            bias_squared: 0.1,
            variance: 0.2,
            noise: 0.05,
            total_loss: 0.35,
            n_bootstrap_samples: 100,
            sample_decompositions: None,
        };

        assert_eq!(decomp.bias_squared, 0.1);
        assert_eq!(decomp.variance, 0.2);
        assert_eq!(decomp.noise, 0.05);
        assert_eq!(decomp.total_loss, 0.35);
        assert_eq!(decomp.n_bootstrap_samples, 100);
        assert!(decomp.sample_decompositions.is_none());
    }

    #[test]
    fn test_sample_bias_variance_structure() {
        let sample_bv = SampleBiasVariance {
            sample_idx: 0,
            true_value: 1.0,
            mean_prediction: 0.9,
            bias_squared: 0.01,
            variance: 0.05,
            bootstrap_predictions: vec![0.8, 0.9, 1.0, 1.1],
        };

        assert_eq!(sample_bv.sample_idx, 0);
        assert_eq!(sample_bv.true_value, 1.0);
        assert_eq!(sample_bv.mean_prediction, 0.9);
        assert_eq!(sample_bv.bias_squared, 0.01);
        assert_eq!(sample_bv.variance, 0.05);
        assert_eq!(sample_bv.bootstrap_predictions.len(), 4);
    }

    #[test]
    fn test_ensemble_size_analysis_structure() {
        let analysis = BiasVarianceEnsembleSizeAnalysis {
            ensemble_sizes: vec![1, 2, 3, 4, 5],
            bias_curve: vec![0.5, 0.45, 0.42, 0.41, 0.4],
            variance_curve: vec![0.3, 0.2, 0.15, 0.12, 0.1],
            total_error_curve: vec![0.8, 0.65, 0.57, 0.53, 0.5],
            optimal_ensemble_size: 5,
            bias_reduction: 0.1,
            variance_reduction: 0.2,
        };

        assert_eq!(analysis.ensemble_sizes.len(), 5);
        assert_eq!(analysis.optimal_ensemble_size, 5);
        assert_eq!(analysis.bias_reduction, 0.1);
        assert_eq!(analysis.variance_reduction, 0.2);

        // Check that curves have same length as ensemble sizes
        assert_eq!(analysis.bias_curve.len(), analysis.ensemble_sizes.len());
        assert_eq!(analysis.variance_curve.len(), analysis.ensemble_sizes.len());
        assert_eq!(
            analysis.total_error_curve.len(),
            analysis.ensemble_sizes.len()
        );
    }

    #[test]
    fn test_diversity_analyzer_creation() {
        // Test that we can create DiversityAnalyzer
        let _analyzer = DiversityAnalyzer;
    }

    #[test]
    fn test_cohens_kappa_calculation() {
        let pred1 = vec![0, 1, 0, 1, 1, 0];
        let pred2 = vec![0, 1, 1, 1, 0, 0];

        let kappa = DiversityAnalyzer::compute_pairwise_kappa(&pred1, &pred2).unwrap();

        // Kappa should be between -1 and 1
        assert!(kappa >= -1.0 && kappa <= 1.0);
    }

    #[test]
    fn test_fleiss_kappa_calculation() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 1, 1], vec![1, 1, 0, 0]];

        let fleiss_kappa = DiversityAnalyzer::compute_fleiss_kappa(&predictions).unwrap();

        // Fleiss' kappa should be between -1 and 1
        assert!(fleiss_kappa >= -1.0 && fleiss_kappa <= 1.0);
    }

    #[test]
    fn test_disagreement_calculation() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 1, 0], vec![1, 0, 0, 1]];

        let disagreement = DiversityAnalyzer::compute_disagreement(&predictions);

        // Disagreement should be between 0 and 1
        assert!(disagreement >= 0.0 && disagreement <= 1.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let pred1 = vec![1, 2, 3, 4, 5];
        let pred2 = vec![2, 4, 6, 8, 10]; // Perfect positive correlation

        let correlation = DiversityAnalyzer::compute_pearson_correlation(&pred1, &pred2).unwrap();

        // Should be close to 1.0 for perfect positive correlation
        assert!((correlation - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_diversity_metrics_structure() {
        let metrics = DiversityMetrics {
            disagreement: 0.3,
            double_fault: 0.1,
            q_statistic: 0.2,
            entropy_diversity: 0.4,
            kw_variance: 0.25,
            kappa: 0.6,
            fleiss_kappa: 0.65,
            interrater_reliability: InterraterReliability {
                overall_agreement: 0.7,
                chance_agreement: 0.4,
                krippendorff_alpha: 0.55,
                pearson_correlation: 0.8,
                weighted_kappa: 0.6,
                kappa_std_error: 0.05,
            },
        };

        assert_eq!(metrics.disagreement, 0.3);
        assert_eq!(metrics.kappa, 0.6);
        assert_eq!(metrics.fleiss_kappa, 0.65);
        assert_eq!(metrics.interrater_reliability.overall_agreement, 0.7);
    }

    #[test]
    fn test_comprehensive_diversity_metrics() {
        let predictions = vec![
            array![0.0, 1.0, 0.0, 1.0],
            array![0.0, 1.0, 1.0, 1.0],
            array![1.0, 0.0, 0.0, 1.0],
        ];
        let ground_truth = array![0.0, 1.0, 0.0, 1.0];

        let diversity_metrics =
            DiversityAnalyzer::compute_diversity_metrics(&predictions, &ground_truth).unwrap();

        // Check that all metrics are within reasonable bounds
        assert!(diversity_metrics.disagreement >= 0.0 && diversity_metrics.disagreement <= 1.0);
        assert!(diversity_metrics.double_fault >= 0.0 && diversity_metrics.double_fault <= 1.0);
        assert!(diversity_metrics.kappa >= -1.0 && diversity_metrics.kappa <= 1.0);
        assert!(diversity_metrics.fleiss_kappa >= -1.0 && diversity_metrics.fleiss_kappa <= 1.0);
        assert!(
            diversity_metrics.interrater_reliability.overall_agreement >= 0.0
                && diversity_metrics.interrater_reliability.overall_agreement <= 1.0
        );
    }

    #[test]
    fn test_perfect_agreement_kappa() {
        let predictions = vec![vec![0, 1, 0, 1], vec![0, 1, 0, 1], vec![0, 1, 0, 1]];

        let cohens_kappa = DiversityAnalyzer::compute_cohens_kappa(&predictions).unwrap();
        let fleiss_kappa = DiversityAnalyzer::compute_fleiss_kappa(&predictions).unwrap();

        // Perfect agreement should give kappa = 1.0
        assert!((cohens_kappa - 1.0).abs() < 0.001);
        assert!((fleiss_kappa - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_zero_agreement_kappa() {
        let predictions = vec![vec![0, 0, 0, 0], vec![1, 1, 1, 1]];

        let cohens_kappa = DiversityAnalyzer::compute_cohens_kappa(&predictions).unwrap();

        // Complete disagreement should give kappa near 0 (worse than random)
        // In this case, we expect kappa = -1 for perfect disagreement, but the formula
        // might give 0 depending on expected agreement calculation
        assert!(cohens_kappa <= 0.0);
    }
}
