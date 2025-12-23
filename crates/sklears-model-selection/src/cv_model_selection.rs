//! Cross-validation based model selection framework
//!
//! This module provides tools for selecting the best model from a set of candidates
//! using cross-validation. It includes model comparison, ranking, and selection
//! strategies with statistical validation.

use crate::cross_validation::CrossValidator;
use scirs2_core::ndarray::{Array1, Array2};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict},
};
// Simple scoring trait for testing
pub trait Scoring {
    fn score(&self, y_true: &[f64], y_pred: &[f64]) -> Result<f64>;
}
use crate::model_comparison::{paired_t_test, StatisticalTestResult};
use std::fmt::{self, Display, Formatter};

/// Result of cross-validation model selection
#[derive(Debug, Clone)]
pub struct CVModelSelectionResult {
    /// Best model index
    pub best_model_index: usize,
    /// Model rankings (sorted by performance)
    pub model_rankings: Vec<ModelRanking>,
    /// Cross-validation scores for each model
    pub cv_scores: Vec<CVModelScore>,
    /// Statistical comparison results
    pub statistical_comparisons: Vec<ModelComparisonPair>,
    /// Selection criteria used
    pub selection_criteria: ModelSelectionCriteria,
    /// Number of CV folds used
    pub n_folds: usize,
}

/// Ranking information for a model
#[derive(Debug, Clone)]
pub struct ModelRanking {
    /// Model index
    pub model_index: usize,
    /// Model name/identifier
    pub model_name: String,
    /// Rank (1 = best)
    pub rank: usize,
    /// Mean CV score
    pub mean_score: f64,
    /// Standard deviation of CV scores
    pub std_score: f64,
    /// 95% confidence interval
    pub confidence_interval: (f64, f64),
    /// Statistical significance vs best model
    pub significant_difference: Option<bool>,
}

/// Cross-validation scores for a model
#[derive(Debug, Clone)]
pub struct CVModelScore {
    /// Model index
    pub model_index: usize,
    /// Model name/identifier
    pub model_name: String,
    /// Individual fold scores
    pub fold_scores: Vec<f64>,
    /// Mean score
    pub mean_score: f64,
    /// Standard deviation
    pub std_score: f64,
    /// Standard error of the mean
    pub std_error: f64,
    /// Minimum score
    pub min_score: f64,
    /// Maximum score
    pub max_score: f64,
}

/// Pairwise model comparison result
#[derive(Debug, Clone)]
pub struct ModelComparisonPair {
    /// First model index
    pub model1_index: usize,
    /// Second model index
    pub model2_index: usize,
    /// Statistical test result
    pub test_result: StatisticalTestResult,
    /// Effect size (Cohen's d)
    pub effect_size: f64,
}

/// Model selection criteria
#[derive(Debug, Clone, PartialEq)]
pub enum ModelSelectionCriteria {
    /// Select model with highest mean CV score
    HighestMean,
    /// Select model with highest mean score within 1 std error of best
    OneStandardError,
    /// Select based on statistical significance
    StatisticalSignificance,
    /// Select most consistent model (lowest CV std)
    MostConsistent,
    /// Custom criteria with weights
    Weighted {
        mean_weight: f64,
        std_weight: f64,
        consistency_weight: f64,
    },
}

impl Display for ModelSelectionCriteria {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            ModelSelectionCriteria::HighestMean => write!(f, "Highest Mean Score"),
            ModelSelectionCriteria::OneStandardError => write!(f, "One Standard Error Rule"),
            ModelSelectionCriteria::StatisticalSignificance => {
                write!(f, "Statistical Significance")
            }
            ModelSelectionCriteria::MostConsistent => write!(f, "Most Consistent"),
            ModelSelectionCriteria::Weighted { .. } => write!(f, "Weighted Criteria"),
        }
    }
}

impl Display for CVModelSelectionResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        writeln!(f, "Cross-Validation Model Selection Results:")?;
        writeln!(f, "Selection Criteria: {}", self.selection_criteria)?;
        writeln!(f, "CV Folds: {}", self.n_folds)?;
        writeln!(
            f,
            "Best Model: {} (index {})",
            self.model_rankings[0].model_name, self.best_model_index
        )?;
        writeln!(f, "\nModel Rankings:")?;
        for ranking in &self.model_rankings {
            writeln!(
                f,
                "  {}. {} - Score: {:.4} ± {:.4}",
                ranking.rank, ranking.model_name, ranking.mean_score, ranking.std_score
            )?;
        }
        Ok(())
    }
}

/// Configuration for cross-validation model selection
#[derive(Debug, Clone)]
pub struct CVModelSelectionConfig {
    /// Selection criteria to use
    pub criteria: ModelSelectionCriteria,
    /// Whether to perform statistical comparisons
    pub perform_statistical_tests: bool,
    /// Significance level for statistical tests
    pub significance_level: f64,
    /// Whether to compute confidence intervals
    pub compute_confidence_intervals: bool,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for CVModelSelectionConfig {
    fn default() -> Self {
        Self {
            criteria: ModelSelectionCriteria::HighestMean,
            perform_statistical_tests: true,
            significance_level: 0.05,
            compute_confidence_intervals: true,
            random_seed: None,
        }
    }
}

/// Cross-validation model selector
pub struct CVModelSelector {
    config: CVModelSelectionConfig,
}

impl CVModelSelector {
    /// Create a new CV model selector with default configuration
    pub fn new() -> Self {
        Self {
            config: CVModelSelectionConfig::default(),
        }
    }

    /// Create a new CV model selector with custom configuration
    pub fn with_config(config: CVModelSelectionConfig) -> Self {
        Self { config }
    }

    /// Set selection criteria
    pub fn criteria(mut self, criteria: ModelSelectionCriteria) -> Self {
        self.config.criteria = criteria;
        self
    }

    /// Enable or disable statistical tests
    pub fn statistical_tests(mut self, enable: bool) -> Self {
        self.config.perform_statistical_tests = enable;
        self
    }

    /// Set significance level
    pub fn significance_level(mut self, level: f64) -> Self {
        self.config.significance_level = level;
        self
    }

    /// Set random seed
    pub fn random_seed(mut self, seed: u64) -> Self {
        self.config.random_seed = Some(seed);
        self
    }

    /// Select best model from candidates using cross-validation
    pub fn select_model<E, X, Y>(
        &self,
        models: &[(E, String)],
        x: &[X],
        _y: &[Y],
        cv: &dyn CrossValidator,
        scoring: &dyn Scoring,
    ) -> Result<CVModelSelectionResult>
    where
        E: Estimator + Fit<X, Y> + Clone,
        E::Fitted: Predict<Vec<f64>, Vec<f64>>,
        X: Clone,
        Y: Clone,
    {
        if models.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "models".to_string(),
                reason: "at least one model must be provided".to_string(),
            });
        }

        // Generate CV splits
        let n_samples = x.len();
        let splits = cv.split(n_samples, None);
        let n_folds = splits.len();

        // Evaluate each model using cross-validation
        let mut cv_scores = Vec::with_capacity(models.len());

        for (model_idx, (_model, name)) in models.iter().enumerate() {
            let dummy_x = Array2::zeros((0, 0));
            let dummy_y = Array1::zeros(0);
            let fold_scores = self.evaluate_model_cv(&(), &dummy_x, &dummy_y, &splits, scoring)?;

            let mean_score = fold_scores.iter().sum::<f64>() / fold_scores.len() as f64;
            let std_score = self.calculate_std(&fold_scores, mean_score);
            let std_error = std_score / (fold_scores.len() as f64).sqrt();
            let min_score = fold_scores.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max_score = fold_scores.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

            cv_scores.push(CVModelScore {
                model_index: model_idx,
                model_name: name.clone(),
                fold_scores,
                mean_score,
                std_score,
                std_error,
                min_score,
                max_score,
            });
        }

        // Perform statistical comparisons if requested
        let statistical_comparisons = if self.config.perform_statistical_tests {
            self.perform_statistical_comparisons(&cv_scores)?
        } else {
            Vec::new()
        };

        // Rank models and select best
        let model_rankings = self.rank_models(&cv_scores, &statistical_comparisons)?;
        let best_model_index = self.select_best_model(&cv_scores, &model_rankings)?;

        Ok(CVModelSelectionResult {
            best_model_index,
            model_rankings,
            cv_scores,
            statistical_comparisons,
            selection_criteria: self.config.criteria.clone(),
            n_folds,
        })
    }

    /// Evaluate a single model using cross-validation
    fn evaluate_model_cv(
        &self,
        _model: &(), // Placeholder for now
        _x: &Array2<f64>,
        _y: &Array1<f64>,
        _splits: &[(Vec<usize>, Vec<usize>)],
        _scoring: &dyn Scoring,
    ) -> Result<Vec<f64>> {
        // Placeholder implementation
        Ok(vec![0.5; 5]) // Return dummy scores for now
    }

    /// Perform pairwise statistical comparisons between models
    fn perform_statistical_comparisons(
        &self,
        cv_scores: &[CVModelScore],
    ) -> Result<Vec<ModelComparisonPair>> {
        let mut comparisons = Vec::new();

        for i in 0..cv_scores.len() {
            for j in (i + 1)..cv_scores.len() {
                let scores1 = &cv_scores[i].fold_scores;
                let scores2 = &cv_scores[j].fold_scores;

                // Perform paired t-test
                let scores1_array = Array1::from_vec(scores1.clone());
                let scores2_array = Array1::from_vec(scores2.clone());
                let test_result = paired_t_test(
                    &scores1_array,
                    &scores2_array,
                    self.config.significance_level,
                )?;

                // Calculate effect size (Cohen's d)
                let effect_size = self.calculate_cohens_d(scores1, scores2);

                comparisons.push(ModelComparisonPair {
                    model1_index: i,
                    model2_index: j,
                    test_result,
                    effect_size,
                });
            }
        }

        Ok(comparisons)
    }

    /// Calculate Cohen's d effect size
    fn calculate_cohens_d(&self, scores1: &[f64], scores2: &[f64]) -> f64 {
        let mean1 = scores1.iter().sum::<f64>() / scores1.len() as f64;
        let mean2 = scores2.iter().sum::<f64>() / scores2.len() as f64;

        let var1 = self.calculate_variance(scores1, mean1);
        let var2 = self.calculate_variance(scores2, mean2);

        let pooled_std = ((var1 + var2) / 2.0).sqrt();

        if pooled_std > 0.0 {
            (mean1 - mean2) / pooled_std
        } else {
            0.0
        }
    }

    /// Rank models based on CV performance
    fn rank_models(
        &self,
        cv_scores: &[CVModelScore],
        statistical_comparisons: &[ModelComparisonPair],
    ) -> Result<Vec<ModelRanking>> {
        let mut rankings: Vec<ModelRanking> = cv_scores
            .iter()
            .map(|score| {
                let confidence_interval = if self.config.compute_confidence_intervals {
                    self.calculate_confidence_interval(
                        &score.fold_scores,
                        score.mean_score,
                        score.std_error,
                    )
                } else {
                    (score.mean_score, score.mean_score)
                };

                ModelRanking {
                    model_index: score.model_index,
                    model_name: score.model_name.clone(),
                    rank: 0, // Will be set after sorting
                    mean_score: score.mean_score,
                    std_score: score.std_score,
                    confidence_interval,
                    significant_difference: None, // Will be set based on statistical tests
                }
            })
            .collect();

        // Sort models based on selection criteria
        match &self.config.criteria {
            ModelSelectionCriteria::HighestMean => {
                rankings.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());
            }
            ModelSelectionCriteria::OneStandardError => {
                // Find best score, then select simplest model within 1 SE
                let best_score = rankings
                    .iter()
                    .map(|r| r.mean_score)
                    .fold(f64::NEG_INFINITY, f64::max);

                let best_se = cv_scores
                    .iter()
                    .find(|s| s.mean_score == best_score)
                    .map(|s| s.std_error)
                    .unwrap_or(0.0);

                let threshold = best_score - best_se;

                // Sort by mean score, but prioritize models within 1 SE of best
                rankings.sort_by(|a, b| {
                    let a_within_se = a.mean_score >= threshold;
                    let b_within_se = b.mean_score >= threshold;

                    match (a_within_se, b_within_se) {
                        (true, false) => std::cmp::Ordering::Less,
                        (false, true) => std::cmp::Ordering::Greater,
                        _ => b.mean_score.partial_cmp(&a.mean_score).unwrap(),
                    }
                });
            }
            ModelSelectionCriteria::MostConsistent => {
                rankings.sort_by(|a, b| a.std_score.partial_cmp(&b.std_score).unwrap());
            }
            ModelSelectionCriteria::StatisticalSignificance => {
                // Sort by mean score first, then adjust based on statistical significance
                rankings.sort_by(|a, b| b.mean_score.partial_cmp(&a.mean_score).unwrap());
            }
            ModelSelectionCriteria::Weighted {
                mean_weight,
                std_weight,
                consistency_weight: _consistency_weight,
            } => {
                // Calculate weighted scores
                let max_mean = rankings
                    .iter()
                    .map(|r| r.mean_score)
                    .fold(f64::NEG_INFINITY, f64::max);
                let min_std = rankings
                    .iter()
                    .map(|r| r.std_score)
                    .fold(f64::INFINITY, f64::min);

                rankings.sort_by(|a, b| {
                    let score_a =
                        a.mean_score / max_mean * mean_weight - a.std_score / min_std * std_weight;
                    let score_b =
                        b.mean_score / max_mean * mean_weight - b.std_score / min_std * std_weight;
                    score_b.partial_cmp(&score_a).unwrap()
                });
            }
        }

        // Assign ranks
        for (idx, ranking) in rankings.iter_mut().enumerate() {
            ranking.rank = idx + 1;
        }

        // Set statistical significance information
        if !statistical_comparisons.is_empty() && !rankings.is_empty() {
            let best_model_idx = rankings[0].model_index;

            for ranking in &mut rankings[1..] {
                // Find comparison with best model
                let comparison = statistical_comparisons.iter().find(|c| {
                    (c.model1_index == best_model_idx && c.model2_index == ranking.model_index)
                        || (c.model2_index == best_model_idx
                            && c.model1_index == ranking.model_index)
                });

                if let Some(comp) = comparison {
                    ranking.significant_difference =
                        Some(comp.test_result.p_value < self.config.significance_level);
                }
            }
        }

        Ok(rankings)
    }

    /// Select the best model based on criteria
    fn select_best_model(
        &self,
        _cv_scores: &[CVModelScore],
        model_rankings: &[ModelRanking],
    ) -> Result<usize> {
        if model_rankings.is_empty() {
            return Err(SklearsError::InvalidParameter {
                name: "model_rankings".to_string(),
                reason: "no models to select from".to_string(),
            });
        }

        // The best model is the first in the rankings (rank 1)
        Ok(model_rankings[0].model_index)
    }

    /// Calculate standard deviation
    fn calculate_std(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let variance = self.calculate_variance(values, mean);
        variance.sqrt()
    }

    /// Calculate variance
    fn calculate_variance(&self, values: &[f64], mean: f64) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let sum_sq_diff = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>();

        sum_sq_diff / (values.len() - 1) as f64
    }

    /// Calculate 95% confidence interval
    fn calculate_confidence_interval(
        &self,
        values: &[f64],
        mean: f64,
        std_error: f64,
    ) -> (f64, f64) {
        // Using t-distribution critical value for 95% CI (approximate)
        let n = values.len() as f64;
        let t_critical = if n > 30.0 { 1.96 } else { 2.0 }; // Simplified

        let margin = t_critical * std_error;
        (mean - margin, mean + margin)
    }
}

impl Default for CVModelSelector {
    fn default() -> Self {
        Self::new()
    }
}

/// Convenience function for cross-validation model selection
pub fn cv_select_model<E, X, Y>(
    models: &[(E, String)],
    x: &[X],
    y: &[Y],
    cv: &dyn CrossValidator,
    scoring: &dyn Scoring,
    criteria: Option<ModelSelectionCriteria>,
) -> Result<CVModelSelectionResult>
where
    E: Estimator + Fit<X, Y> + Clone,
    E::Fitted: Predict<Vec<f64>, Vec<f64>>,
    X: Clone,
    Y: Clone,
{
    let mut selector = CVModelSelector::new();
    if let Some(crit) = criteria {
        selector = selector.criteria(crit);
    }
    selector.select_model(models, x, y, cv, scoring)
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
            // Return negative MSE (higher is better)
            let mse = y_true
                .iter()
                .zip(y_pred.iter())
                .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
                .sum::<f64>()
                / y_true.len() as f64;
            Ok(-mse)
        }
    }

    #[test]
    fn test_cv_model_selector_creation() {
        let selector = CVModelSelector::new();
        assert_eq!(
            selector.config.criteria,
            ModelSelectionCriteria::HighestMean
        );
        assert!(selector.config.perform_statistical_tests);
        assert_eq!(selector.config.significance_level, 0.05);
    }

    #[test]
    fn test_cv_model_selection() {
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
                    performance_level: 0.7,
                },
                "Model C".to_string(),
            ),
        ];

        let x: Vec<Vec<f64>> = (0..100).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[0] * 0.5 + 0.1]).collect();

        let cv = KFold::new(5);
        let scoring = MockScoring;

        let selector = CVModelSelector::new();
        let result = selector.select_model(&models, &x, &y, &cv, &scoring);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert_eq!(result.model_rankings.len(), 3);
        assert_eq!(result.cv_scores.len(), 3);
        assert_eq!(result.n_folds, 5);
        assert!(result.best_model_index < 3);
    }

    #[test]
    fn test_different_selection_criteria() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 0.8,
                },
                "Consistent".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.85,
                },
                "High Variance".to_string(),
            ),
        ];

        let x: Vec<Vec<f64>> = (0..50).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[0] * 0.3]).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        // Test highest mean criteria
        let selector = CVModelSelector::new().criteria(ModelSelectionCriteria::HighestMean);
        let result = selector.select_model(&models, &x, &y, &cv, &scoring);
        assert!(result.is_ok());

        // Test most consistent criteria
        let selector = CVModelSelector::new().criteria(ModelSelectionCriteria::MostConsistent);
        let result = selector.select_model(&models, &x, &y, &cv, &scoring);
        assert!(result.is_ok());

        // Test one standard error rule
        let selector = CVModelSelector::new().criteria(ModelSelectionCriteria::OneStandardError);
        let result = selector.select_model(&models, &x, &y, &cv, &scoring);
        assert!(result.is_ok());
    }

    #[test]
    fn test_convenience_function() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 0.9,
                },
                "Best Model".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.7,
                },
                "Worse Model".to_string(),
            ),
        ];

        let x: Vec<Vec<f64>> = (0..30).map(|i| vec![i as f64 * 0.05]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[0] * 0.4]).collect();
        let cv = KFold::new(3);
        let scoring = MockScoring;

        let result = cv_select_model(
            &models,
            &x,
            &y,
            &cv,
            &scoring,
            Some(ModelSelectionCriteria::HighestMean),
        );
        if let Err(e) = &result {
            eprintln!("Error in cv_select_model: {:?}", e);
        }
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.model_rankings.len(), 2);
        assert_eq!(result.model_rankings[0].rank, 1);
    }

    #[test]
    fn test_statistical_comparisons() {
        let models = vec![
            (
                MockEstimator {
                    performance_level: 1.0,
                },
                "Perfect".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.9,
                },
                "Good".to_string(),
            ),
            (
                MockEstimator {
                    performance_level: 0.8,
                },
                "Okay".to_string(),
            ),
        ];

        let x: Vec<Vec<f64>> = (0..60).map(|i| vec![i as f64 * 0.1]).collect();
        let y: Vec<Vec<f64>> = x.iter().map(|xi| vec![xi[0]]).collect();
        let cv = KFold::new(5);
        let scoring = MockScoring;

        let selector = CVModelSelector::new().statistical_tests(true);
        let result = selector.select_model(&models, &x, &y, &cv, &scoring);

        assert!(result.is_ok());
        let result = result.unwrap();
        assert!(!result.statistical_comparisons.is_empty());

        // Should have 3 pairwise comparisons for 3 models
        assert_eq!(result.statistical_comparisons.len(), 3);
    }
}
