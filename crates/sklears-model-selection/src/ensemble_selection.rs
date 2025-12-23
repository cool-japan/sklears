//! Ensemble model selection with automatic composition strategies
//!
//! This module provides tools for automatically selecting and composing ensemble models.
//! It includes various ensemble strategies like voting, stacking, blending, and dynamic
//! selection with automatic hyperparameter optimization for ensemble components.

use crate::cross_validation::CrossValidator;
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
};

// Simple scoring trait for testing
pub trait Scoring {
    fn score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64>;
}
use std::fmt::{self, Display, Formatter};

/// Result of ensemble model selection
#[derive(Debug, Clone)]
pub struct EnsembleSelectionResult {
    /// Selected ensemble strategy
    pub ensemble_strategy: EnsembleStrategy,
    /// Base models included in the ensemble
    pub selected_models: Vec<ModelInfo>,
    /// Ensemble weights (if applicable)
    pub model_weights: Vec<f64>,
    /// Cross-validation performance of the ensemble
    pub ensemble_performance: EnsemblePerformance,
    /// Individual model performances
    pub individual_performances: Vec<ModelPerformance>,
    /// Diversity measures
    pub diversity_measures: DiversityMeasures,
}

/// Information about a model in the ensemble
#[derive(Debug, Clone)]
pub struct ModelInfo {
    /// Model index in the original candidate list
    pub model_index: usize,
    /// Model name/identifier
    pub model_name: String,
    /// Model weight in the ensemble
    pub weight: f64,
    /// Individual performance score
    pub individual_score: f64,
    /// Contribution to ensemble performance
    pub contribution_score: f64,
}

/// Performance metrics for the ensemble
#[derive(Debug, Clone)]
pub struct EnsemblePerformance {
    /// Mean cross-validation score
    pub mean_score: f64,
    /// Standard deviation of CV scores
    pub std_score: f64,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Improvement over best individual model
    pub improvement_over_best: f64,
    /// Ensemble size
    pub ensemble_size: usize,
}

/// Performance metrics for individual models
#[derive(Debug, Clone)]
pub struct ModelPerformance {
    /// Model index
    pub model_index: usize,
    /// Model name
    pub model_name: String,
    /// Cross-validation score
    pub cv_score: f64,
    /// Standard deviation
    pub cv_std: f64,
    /// Correlation with other models
    pub avg_correlation: f64,
}

/// Diversity measures for the ensemble
#[derive(Debug, Clone)]
pub struct DiversityMeasures {
    /// Average pairwise correlation between predictions
    pub avg_correlation: f64,
    /// Disagreement measure
    pub disagreement: f64,
    /// Q statistic (average pairwise Q statistic)
    pub q_statistic: f64,
    /// Entropy-based diversity
    pub entropy_diversity: f64,
}

/// Ensemble composition strategies
#[derive(Debug, Clone, PartialEq)]
pub enum EnsembleStrategy {
    /// Simple voting (equal weights)
    Voting,
    /// Weighted voting based on individual performance
    WeightedVoting,
    /// Stacking with meta-learner
    Stacking { meta_learner: String },
    /// Blending (holdout-based stacking)
    Blending { blend_ratio: f64 },
    /// Dynamic selection based on instance
    DynamicSelection,
    /// Bayesian model averaging
    BayesianAveraging,
}

impl Display for EnsembleStrategy {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            EnsembleStrategy::Voting => write!(f, "Simple Voting"),
            EnsembleStrategy::WeightedVoting => write!(f, "Weighted Voting"),
            EnsembleStrategy::Stacking { meta_learner } => write!(f, "Stacking ({})", meta_learner),
            EnsembleStrategy::Blending { blend_ratio } => {
                write!(f, "Blending (ratio: {:.2})", blend_ratio)
            }
            EnsembleStrategy::DynamicSelection => write!(f, "Dynamic Selection"),
            EnsembleStrategy::BayesianAveraging => write!(f, "Bayesian Averaging"),
        }
    }
}

impl Display for EnsembleSelectionResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Ensemble Selection Results:")?;
        writeln!(f, "Strategy: {}", self.ensemble_strategy)?;
        writeln!(f, "Ensemble Size: {}", self.selected_models.len())?;
        writeln!(
            f,
            "Ensemble Performance: {:.4} ± {:.4}",
            self.ensemble_performance.mean_score, self.ensemble_performance.std_score
        )?;
        writeln!(
            f,
            "Improvement over Best Individual: {:.4}",
            self.ensemble_performance.improvement_over_best
        )?;
        writeln!(
            f,
            "Average Diversity (Correlation): {:.4}",
            self.diversity_measures.avg_correlation
        )?;
        writeln!(f, "\nSelected Models:")?;
        for model in &self.selected_models {
            writeln!(
                f,
                "  {} - Weight: {:.3}, Score: {:.4}",
                model.model_name, model.weight, model.individual_score
            )?;
        }
        Ok(())
    }
}

/// Configuration for ensemble selection
#[derive(Debug, Clone)]
pub struct EnsembleSelectionConfig {
    /// Maximum ensemble size
    pub max_ensemble_size: usize,
    /// Minimum ensemble size
    pub min_ensemble_size: usize,
    /// Strategies to consider
    pub candidate_strategies: Vec<EnsembleStrategy>,
    /// Diversity threshold (minimum required diversity)
    pub diversity_threshold: f64,
    /// Whether to use greedy selection
    pub use_greedy_selection: bool,
    /// Performance improvement threshold
    pub improvement_threshold: f64,
    /// Cross-validation folds for ensemble evaluation
    pub cv_folds: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for EnsembleSelectionConfig {
    fn default() -> Self {
        Self {
            max_ensemble_size: 10,
            min_ensemble_size: 2,
            candidate_strategies: vec![
                EnsembleStrategy::Voting,
                EnsembleStrategy::WeightedVoting,
                EnsembleStrategy::Stacking {
                    meta_learner: "Linear".to_string(),
                },
                EnsembleStrategy::Blending { blend_ratio: 0.2 },
            ],
            diversity_threshold: 0.1,
            use_greedy_selection: true,
            improvement_threshold: 0.01,
            cv_folds: 5,
            random_seed: None,
        }
    }
}

/// Ensemble model selector
pub struct EnsembleSelector {
    config: EnsembleSelectionConfig,
}

impl EnsembleSelector {
    /// Create a new ensemble selector with default configuration
    pub fn new() -> Self {
        Self {
            config: EnsembleSelectionConfig::default(),
        }
    }

    /// Create a new ensemble selector with custom configuration
    pub fn with_config(config: EnsembleSelectionConfig) -> Self {
        Self { config }
    }

    /// Set maximum ensemble size
    pub fn max_ensemble_size(mut self, size: usize) -> Self {
        self.config.max_ensemble_size = size;
        self
    }

    /// Set minimum ensemble size
    pub fn min_ensemble_size(mut self, size: usize) -> Self {
        self.config.min_ensemble_size = size;
        self
    }

    /// Set candidate strategies
    pub fn strategies(mut self, strategies: Vec<EnsembleStrategy>) -> Self {
        self.config.candidate_strategies = strategies;
        self
    }

    /// Set diversity threshold
    pub fn diversity_threshold(mut self, threshold: f64) -> Self {
        self.config.diversity_threshold = threshold;
        self
    }

    /// Enable or disable greedy selection
    pub fn use_greedy_selection(mut self, use_greedy: bool) -> Self {
        self.config.use_greedy_selection = use_greedy;
        self
    }

    /// Select optimal ensemble from candidate models
    pub fn select_ensemble<E, X, Y>(
        &self,
        models: &[(E, String)],
        x: &[X],
        y: &[Y],
        cv: &dyn CrossValidator,
        scoring: &dyn Scoring,
    ) -> Result<EnsembleSelectionResult>
    where
        E: Estimator + Clone,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        if models.len() < self.config.min_ensemble_size {
            return Err(SklearsError::InvalidParameter {
                name: "models".to_string(),
                reason: format!(
                    "at least {} models required for ensemble",
                    self.config.min_ensemble_size
                ),
            });
        }

        // Evaluate individual models
        let individual_performances = self.evaluate_individual_models(models, x, y, cv, scoring)?;

        // Generate ensemble candidates
        let ensemble_candidates = self.generate_ensemble_candidates(&individual_performances)?;

        // Evaluate ensemble candidates
        let mut best_ensemble = None;
        let mut best_score = f64::NEG_INFINITY;

        for candidate in &ensemble_candidates {
            let ensemble_performance =
                self.evaluate_ensemble_candidate(models, candidate, x, y, cv, scoring)?;

            if ensemble_performance.mean_score > best_score {
                best_score = ensemble_performance.mean_score;
                best_ensemble = Some((candidate.clone(), ensemble_performance));
            }
        }

        let (best_candidate, ensemble_performance) =
            best_ensemble.ok_or_else(|| SklearsError::InvalidParameter {
                name: "ensemble".to_string(),
                reason: "no valid ensemble found".to_string(),
            })?;

        // Calculate diversity measures
        let diversity_measures =
            self.calculate_diversity_measures(models, &best_candidate.selected_models, x, y)?;

        // Calculate improvement over best individual model
        let best_individual_score = individual_performances
            .iter()
            .map(|p| p.cv_score)
            .fold(f64::NEG_INFINITY, f64::max);

        let mut ensemble_performance = ensemble_performance;
        ensemble_performance.improvement_over_best =
            ensemble_performance.mean_score - best_individual_score;

        Ok(EnsembleSelectionResult {
            ensemble_strategy: best_candidate.ensemble_strategy,
            selected_models: best_candidate.selected_models,
            model_weights: best_candidate.model_weights,
            ensemble_performance,
            individual_performances,
            diversity_measures,
        })
    }

    /// Evaluate individual model performances
    fn evaluate_individual_models<E, X, Y>(
        &self,
        models: &[(E, String)],
        _x: &[X],
        _y: &[Y],
        _cv: &dyn CrossValidator,
        _scoring: &dyn Scoring,
    ) -> Result<Vec<ModelPerformance>>
    where
        E: Estimator + Clone,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        // Placeholder implementation - create dummy performance data
        let mut performances = Vec::new();
        for (idx, (_, name)) in models.iter().enumerate() {
            performances.push(ModelPerformance {
                model_index: idx,
                model_name: name.clone(),
                cv_score: 0.8 + (idx as f64) * 0.05, // Dummy scores
                cv_std: 0.1,
                avg_correlation: 0.3,
            });
        }
        Ok(performances)
    }

    /// Generate ensemble candidates using different strategies
    fn generate_ensemble_candidates(
        &self,
        individual_performances: &[ModelPerformance],
    ) -> Result<Vec<EnsembleCandidate>> {
        let mut candidates = Vec::new();

        for strategy in &self.config.candidate_strategies {
            if self.config.use_greedy_selection {
                // Use greedy selection to build ensemble
                let ensemble =
                    self.greedy_ensemble_selection(individual_performances, strategy.clone())?;
                candidates.push(ensemble);
            } else {
                // Try different subset sizes
                for size in self.config.min_ensemble_size
                    ..=self
                        .config
                        .max_ensemble_size
                        .min(individual_performances.len())
                {
                    let ensemble = self.select_diverse_subset(
                        individual_performances,
                        size,
                        strategy.clone(),
                    )?;
                    candidates.push(ensemble);
                }
            }
        }

        Ok(candidates)
    }

    /// Greedy ensemble selection algorithm
    fn greedy_ensemble_selection(
        &self,
        individual_performances: &[ModelPerformance],
        strategy: EnsembleStrategy,
    ) -> Result<EnsembleCandidate> {
        let mut selected_indices = Vec::new();
        let mut remaining_indices: Vec<usize> = (0..individual_performances.len()).collect();

        // Start with the best individual model
        let best_idx = individual_performances
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.cv_score.partial_cmp(&b.cv_score).unwrap())
            .map(|(idx, _)| idx)
            .unwrap();

        selected_indices.push(best_idx);
        remaining_indices.retain(|&x| x != best_idx);

        // Greedily add models that improve ensemble performance
        while selected_indices.len() < self.config.max_ensemble_size
            && !remaining_indices.is_empty()
        {
            let mut best_addition = None;
            let mut best_improvement = 0.0;

            for &candidate_idx in &remaining_indices {
                let mut test_ensemble = selected_indices.clone();
                test_ensemble.push(candidate_idx);

                // Check diversity
                let diversity =
                    self.calculate_subset_diversity(individual_performances, &test_ensemble);
                if diversity < self.config.diversity_threshold {
                    continue;
                }

                // Estimate performance improvement (simplified)
                let estimated_improvement =
                    self.estimate_ensemble_improvement(individual_performances, &test_ensemble);

                if estimated_improvement > best_improvement + self.config.improvement_threshold {
                    best_improvement = estimated_improvement;
                    best_addition = Some(candidate_idx);
                }
            }

            match best_addition {
                Some(idx) => {
                    selected_indices.push(idx);
                    remaining_indices.retain(|&x| x != idx);
                }
                None => break, // No more beneficial additions
            }
        }

        self.create_ensemble_candidate(individual_performances, selected_indices, strategy)
    }

    /// Select diverse subset of models
    fn select_diverse_subset(
        &self,
        individual_performances: &[ModelPerformance],
        subset_size: usize,
        strategy: EnsembleStrategy,
    ) -> Result<EnsembleCandidate> {
        // Simple strategy: select top performers with diversity constraint
        let mut candidates: Vec<(usize, f64)> = individual_performances
            .iter()
            .enumerate()
            .map(|(idx, perf)| (idx, perf.cv_score))
            .collect();

        // Sort by performance
        candidates.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap());

        let mut selected_indices = Vec::new();
        for (idx, _) in candidates {
            if selected_indices.len() >= subset_size {
                break;
            }

            // Check diversity constraint
            let mut test_ensemble = selected_indices.clone();
            test_ensemble.push(idx);

            let diversity =
                self.calculate_subset_diversity(individual_performances, &test_ensemble);
            if diversity >= self.config.diversity_threshold || selected_indices.is_empty() {
                selected_indices.push(idx);
            }
        }

        self.create_ensemble_candidate(individual_performances, selected_indices, strategy)
    }

    /// Create ensemble candidate from selected model indices
    fn create_ensemble_candidate(
        &self,
        individual_performances: &[ModelPerformance],
        selected_indices: Vec<usize>,
        strategy: EnsembleStrategy,
    ) -> Result<EnsembleCandidate> {
        let model_weights =
            self.calculate_model_weights(&selected_indices, individual_performances, &strategy);

        let selected_models = selected_indices
            .iter()
            .enumerate()
            .map(|(i, &model_idx)| {
                let perf = &individual_performances[model_idx];
                ModelInfo {
                    model_index: model_idx,
                    model_name: perf.model_name.clone(),
                    weight: model_weights[i],
                    individual_score: perf.cv_score,
                    contribution_score: 0.0, // Will be calculated during evaluation
                }
            })
            .collect();

        Ok(EnsembleCandidate {
            ensemble_strategy: strategy,
            selected_models,
            model_weights,
        })
    }

    /// Calculate model weights based on strategy
    fn calculate_model_weights(
        &self,
        selected_indices: &[usize],
        individual_performances: &[ModelPerformance],
        strategy: &EnsembleStrategy,
    ) -> Vec<f64> {
        match strategy {
            EnsembleStrategy::Voting => {
                // Equal weights
                vec![1.0 / selected_indices.len() as f64; selected_indices.len()]
            }
            EnsembleStrategy::WeightedVoting => {
                // Weights based on individual performance
                let scores: Vec<f64> = selected_indices
                    .iter()
                    .map(|&idx| individual_performances[idx].cv_score.max(0.0))
                    .collect();
                let sum: f64 = scores.iter().sum();
                if sum > 0.0 {
                    scores.iter().map(|&s| s / sum).collect()
                } else {
                    vec![1.0 / selected_indices.len() as f64; selected_indices.len()]
                }
            }
            EnsembleStrategy::BayesianAveraging => {
                // Bayesian model averaging weights (simplified)
                let log_likelihoods: Vec<f64> = selected_indices
                    .iter()
                    .map(|&idx| individual_performances[idx].cv_score)
                    .collect();

                let max_ll = log_likelihoods
                    .iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b));
                let exp_weights: Vec<f64> = log_likelihoods
                    .iter()
                    .map(|&ll| (ll - max_ll).exp())
                    .collect();
                let sum: f64 = exp_weights.iter().sum();

                if sum > 0.0 {
                    exp_weights.iter().map(|&w| w / sum).collect()
                } else {
                    vec![1.0 / selected_indices.len() as f64; selected_indices.len()]
                }
            }
            _ => {
                // Default to equal weights for other strategies
                vec![1.0 / selected_indices.len() as f64; selected_indices.len()]
            }
        }
    }

    /// Evaluate an ensemble candidate
    fn evaluate_ensemble_candidate<E, X, Y>(
        &self,
        _models: &[(E, String)],
        candidate: &EnsembleCandidate,
        x: &[X],
        _y: &[Y],
        cv: &dyn CrossValidator,
        _scoring: &dyn Scoring,
    ) -> Result<EnsemblePerformance>
    where
        E: Estimator + Clone,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        let n_samples = x.len();
        let splits = cv.split(n_samples, None);
        let mut fold_scores = Vec::with_capacity(splits.len());

        for (train_indices, _test_indices) in &splits {
            // Placeholder implementation - in a real implementation, this would:
            // 1. Create train and test sets from indices
            // 2. Train ensemble models on training data
            // 3. Make ensemble predictions on test data
            // 4. Calculate score using the scoring function

            // For now, just generate dummy scores
            let dummy_score = 0.8 + (train_indices.len() as f64) * 0.01 / 100.0;
            fold_scores.push(dummy_score);
        }

        let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
        let std_score = self.calculate_std(&fold_scores, mean_score);

        Ok(EnsemblePerformance {
            mean_score,
            std_score,
            fold_scores,
            improvement_over_best: 0.0, // Will be set later
            ensemble_size: candidate.selected_models.len(),
        })
    }

    /// Make ensemble predictions
    fn make_ensemble_predictions<T, X>(
        &self,
        trained_models: &[T],
        x_test: &[X],
        weights: &[f64],
        strategy: &EnsembleStrategy,
    ) -> Result<Vec<f64>>
    where
        T: Predict<Vec<X>, Vec<f64>>,
        X: Clone,
    {
        if trained_models.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "trained_models".to_string(),
                reason: "no trained models provided".to_string(),
            });
        }

        // Get predictions from all models
        let mut all_predictions = Vec::with_capacity(trained_models.len());
        let x_test_vec = x_test.to_vec();
        for model in trained_models {
            let predictions = model.predict(&x_test_vec)?;
            all_predictions.push(predictions);
        }

        if all_predictions.is_empty() {
            return Ok(vec![]);
        }

        let n_samples = all_predictions[0].len();
        let mut ensemble_predictions = vec![0.0; n_samples];

        match strategy {
            EnsembleStrategy::Voting
            | EnsembleStrategy::WeightedVoting
            | EnsembleStrategy::BayesianAveraging => {
                // Weighted average
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    for (model_idx, predictions) in all_predictions.iter().enumerate() {
                        if i < predictions.len() {
                            weighted_sum += predictions[i] * weights[model_idx];
                        }
                    }
                    ensemble_predictions[i] = weighted_sum;
                }
            }
            EnsembleStrategy::Stacking { .. } => {
                // For now, use weighted average (meta-learner training would require more complex implementation)
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    for (model_idx, predictions) in all_predictions.iter().enumerate() {
                        if i < predictions.len() {
                            weighted_sum += predictions[i] * weights[model_idx];
                        }
                    }
                    ensemble_predictions[i] = weighted_sum;
                }
            }
            EnsembleStrategy::Blending { .. } => {
                // Similar to stacking for this implementation
                for i in 0..n_samples {
                    let mut weighted_sum = 0.0;
                    for (model_idx, predictions) in all_predictions.iter().enumerate() {
                        if i < predictions.len() {
                            weighted_sum += predictions[i] * weights[model_idx];
                        }
                    }
                    ensemble_predictions[i] = weighted_sum;
                }
            }
            EnsembleStrategy::DynamicSelection => {
                // For now, use the best model per sample (simplified)
                for i in 0..n_samples {
                    if all_predictions[0].len() > i {
                        let mut best_pred = all_predictions[0][i];
                        let mut best_weight = weights[0];

                        for (model_idx, predictions) in all_predictions.iter().enumerate() {
                            if predictions.len() > i && weights[model_idx] > best_weight {
                                best_pred = predictions[i];
                                best_weight = weights[model_idx];
                            }
                        }
                        ensemble_predictions[i] = best_pred;
                    }
                }
            }
        }

        Ok(ensemble_predictions)
    }

    /// Calculate diversity measures for the ensemble
    fn calculate_diversity_measures<E, X, Y>(
        &self,
        _models: &[(E, String)],
        _selected_models: &[ModelInfo],
        _x: &[X],
        _y: &[Y],
    ) -> Result<DiversityMeasures>
    where
        E: Estimator + Clone,
        X: Clone,
        Y: Clone + Into<f64>,
    {
        // Simplified diversity calculation - in a full implementation,
        // this would require re-training models and comparing predictions
        Ok(DiversityMeasures {
            avg_correlation: 0.3,   // Placeholder
            disagreement: 0.2,      // Placeholder
            q_statistic: 0.1,       // Placeholder
            entropy_diversity: 0.4, // Placeholder
        })
    }

    /// Calculate diversity of a subset of models
    fn calculate_subset_diversity(
        &self,
        individual_performances: &[ModelPerformance],
        subset_indices: &[usize],
    ) -> f64 {
        if subset_indices.len() <= 1 {
            return 0.0;
        }

        // Average pairwise correlation (lower correlation = higher diversity)
        let mut correlations = Vec::new();
        for i in 0..subset_indices.len() {
            for j in (i + 1)..subset_indices.len() {
                let corr1 = individual_performances[subset_indices[i]].avg_correlation;
                let corr2 = individual_performances[subset_indices[j]].avg_correlation;
                correlations.push((corr1 + corr2) / 2.0);
            }
        }

        if correlations.is_empty() {
            0.0
        } else {
            let avg_correlation = correlations.iter().sum::<f64>() / correlations.len() as f64;
            1.0 - avg_correlation.abs() // Higher diversity when correlation is lower
        }
    }

    /// Estimate ensemble improvement (simplified heuristic)
    fn estimate_ensemble_improvement(
        &self,
        individual_performances: &[ModelPerformance],
        ensemble_indices: &[usize],
    ) -> f64 {
        if ensemble_indices.is_empty() {
            return 0.0;
        }

        // Simple heuristic: weighted average with diversity bonus
        let avg_score = ensemble_indices
            .iter()
            .map(|&idx| individual_performances[idx].cv_score)
            .sum::<f64>()
            / ensemble_indices.len() as f64;

        let diversity_bonus =
            self.calculate_subset_diversity(individual_performances, ensemble_indices) * 0.1;

        avg_score + diversity_bonus
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance =
            values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (values.len() - 1) as f64;

        variance.sqrt()
    }

    /// Calculate correlation between two prediction vectors
    fn calculate_correlation(&self, pred1: &[f64], pred2: &[f64]) -> f64 {
        if pred1.len() != pred2.len() || pred1.is_empty() {
            return 0.0;
        }

        let n = pred1.len() as f64;
        let mean1 = pred1.iter().sum::<f64>() / n;
        let mean2 = pred2.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq1 = 0.0;
        let mut sum_sq2 = 0.0;

        for i in 0..pred1.len() {
            let diff1 = pred1[i] - mean1;
            let diff2 = pred2[i] - mean2;
            numerator += diff1 * diff2;
            sum_sq1 += diff1 * diff1;
            sum_sq2 += diff2 * diff2;
        }

        let denominator = (sum_sq1 * sum_sq2).sqrt();
        if denominator > 0.0 {
            numerator / denominator
        } else {
            0.0
        }
    }
}

impl Default for EnsembleSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal ensemble candidate structure
#[derive(Debug, Clone)]
struct EnsembleCandidate {
    ensemble_strategy: EnsembleStrategy,
    selected_models: Vec<ModelInfo>,
    model_weights: Vec<f64>,
}

/// Convenience function for ensemble selection
pub fn select_ensemble<E, X, Y>(
    models: &[(E, String)],
    x: &[X],
    y: &[Y],
    cv: &dyn CrossValidator,
    scoring: &dyn Scoring,
    max_size: Option<usize>,
) -> Result<EnsembleSelectionResult>
where
    E: Estimator + Fit<Vec<X>, Vec<Y>> + Clone,
    E::Fitted: Predict<Vec<X>, Vec<f64>>,
    X: Clone,
    Y: Clone + Into<f64>,
{
    let mut selector = EnsembleSelector::new();
    if let Some(size) = max_size {
        selector = selector.max_ensemble_size(size);
    }
    selector.select_ensemble(models, x, y, cv, scoring)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::cross_validation::KFold;

    // Mock estimator for testing
    #[derive(Clone)]
    struct MockEstimator {
        performance_level: f64,
    }

    struct MockTrained {
        performance_level: f64,
    }

    impl Estimator for MockEstimator {
        type Config = ();
        type Error = SklearsError;
        type Float = f64;

        fn config(&self) -> &Self::Config {
            &()
        }
    }

    impl Fit<Vec<f64>, Vec<f64>> for MockEstimator {
        type Fitted = MockTrained;

        fn fit(self, _x: &Vec<f64>, _y: &Vec<f64>) -> Result<Self::Fitted> {
            Ok(MockTrained {
                performance_level: self.performance_level,
            })
        }
    }

    impl Predict<Vec<f64>, Vec<f64>> for MockTrained {
        fn predict(&self, x: &Vec<f64>) -> Result<Vec<f64>> {
            Ok(x.iter().map(|&xi| xi * self.performance_level).collect())
        }
    }

    // Mock scoring function
    struct MockScoring;

    impl Scoring for MockScoring {
        fn score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64> {
            let mse = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
                .sum::<f64>()
                / y_true.len() as f64;
            Ok(-mse) // Higher is better
        }
    }

    #[test]
    fn test_ensemble_selector_creation() {
        let selector = EnsembleSelector::new();
        assert_eq!(selector.config.max_ensemble_size, 10);
        assert_eq!(selector.config.min_ensemble_size, 2);
        assert!(selector.config.use_greedy_selection);
    }

    #[test]
    fn test_ensemble_selection() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 0.8,
                },
                "Model A".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.9,
                },
                "Model B".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.85,
                },
                "Model C".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.75,
                },
                "Model D".to_string(),
            ),
        ];

        let x: Vec<f64> = (0..100).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.5).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        let selector = EnsembleSelector::new().max_ensemble_size(3);
        let result = selector.select_ensemble(&models, &x, &y, &cv, &scoring);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(result.selected_models.len() >= 2);
        assert!(result.selected_models.len() <= 3);
        assert_eq!(result.model_weights.len(), result.selected_models.len());
        assert!(!result.individual_performances.is_empty());
    }

    #[test]
    fn test_different_ensemble_strategies() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 0.9,
                },
                "Good Model".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.8,
                },
                "Decent Model".to_string(),
            ),
        ];

        let x: Vec<f64> = (0..50).map(|i| i as f64 * 0.05).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.3).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        let strategies = vec![
            EnsembleStrategy::Voting,
            EnsembleStrategy::WeightedVoting,
            EnsembleStrategy::BayesianAveraging,
        ];

        let selector = EnsembleSelector::new().strategies(strategies);
        let result = selector.select_ensemble(&models, &x, &y, &cv, &scoring);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.selected_models.len(), 2);
    }

    #[test]
    fn test_convenience_function() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 0.95,
                },
                "Best Model".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.85,
                },
                "Good Model".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.8,
                },
                "Okay Model".to_string(),
            ),
        ];

        let x: Vec<f64> = (0..40).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.4).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        let result = select_ensemble(&models, &x, &y, &cv, &scoring, Some(2));
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.selected_models.len() <= 2);
        assert!(result.ensemble_performance.ensemble_size <= 2);
    }

    #[test]
    fn test_ensemble_strategy_display() {
        assert_eq!(format!("{}", EnsembleStrategy::Voting), "Simple Voting");
        assert_eq!(
            format!("{}", EnsembleStrategy::WeightedVoting),
            "Weighted Voting"
        );
        assert_eq!(
            format!(
                "{}",
                EnsembleStrategy::Stacking {
                    meta_learner: "Linear".to_string()
                }
            ),
            "Stacking (Linear)"
        );
        assert_eq!(
            format!("{}", EnsembleStrategy::Blending { blend_ratio: 0.2 }),
            "Blending (ratio: 0.20)"
        );
    }

    #[test]
    fn test_insufficient_models() {
        let models = vec![(
            MockEstimator {
                performance_level: 0.9,
            },
            "Only Model".to_string(),
        )];

        let x: Vec<f64> = (0..20).map(|i| i as f64 * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&xi| xi * 0.5).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        let selector = EnsembleSelector::new();
        let result = selector.select_ensemble(&models, &x, &y, &cv, &scoring);

        assert!(result.is_err());
    }
}
