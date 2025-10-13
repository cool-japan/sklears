//! Enhanced scoring utilities for model selection

use scirs2_core::ndarray::Array1;
use scirs2_core::random::{rngs::StdRng, Rng, SeedableRng};
use sklears_core::{
    error::{Result, SklearsError},
    traits::Score,
    types::Float,
};
use sklears_metrics::{
    classification::{accuracy_score, f1_score, precision_score, recall_score},
    regression::{explained_variance_score, mean_absolute_error, mean_squared_error, r2_score},
};
use std::collections::HashMap;
use std::sync::Arc;

/// Custom scoring function trait
pub trait CustomScorer: Send + Sync + std::fmt::Debug {
    /// Compute score given true and predicted values
    fn score(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<f64>;
    /// Get the name of this custom scorer
    fn name(&self) -> &str;
    /// Whether higher scores are better (true) or lower scores are better (false)
    fn higher_is_better(&self) -> bool;
}

/// Custom scoring function wrapper for closures
pub struct ClosureScorer {
    name: String,
    scorer_fn: Arc<dyn Fn(&Array1<Float>, &Array1<Float>) -> Result<f64> + Send + Sync>,
    higher_is_better: bool,
}

impl ClosureScorer {
    /// Create a new custom scorer from a closure
    pub fn new<F>(name: String, scorer_fn: F, higher_is_better: bool) -> Self
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<f64> + Send + Sync + 'static,
    {
        Self {
            name,
            scorer_fn: Arc::new(scorer_fn),
            higher_is_better,
        }
    }
}

impl std::fmt::Debug for ClosureScorer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ClosureScorer")
            .field("name", &self.name)
            .field("higher_is_better", &self.higher_is_better)
            .finish()
    }
}

impl CustomScorer for ClosureScorer {
    fn score(&self, y_true: &Array1<Float>, y_pred: &Array1<Float>) -> Result<f64> {
        (self.scorer_fn)(y_true, y_pred)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn higher_is_better(&self) -> bool {
        self.higher_is_better
    }
}

/// Scorer registry for built-in and custom scorers
#[derive(Debug, Clone)]
pub struct ScorerRegistry {
    custom_scorers: HashMap<String, Arc<dyn CustomScorer>>,
}

impl Default for ScorerRegistry {
    fn default() -> Self {
        Self::new()
    }
}

impl ScorerRegistry {
    /// Create a new scorer registry
    pub fn new() -> Self {
        Self {
            custom_scorers: HashMap::new(),
        }
    }

    /// Register a custom scorer
    pub fn register_scorer(&mut self, scorer: Arc<dyn CustomScorer>) {
        self.custom_scorers
            .insert(scorer.name().to_string(), scorer);
    }

    /// Register a custom scorer from a closure
    pub fn register_closure_scorer<F>(&mut self, name: String, scorer_fn: F, higher_is_better: bool)
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<f64> + Send + Sync + 'static,
    {
        let scorer = Arc::new(ClosureScorer::new(name, scorer_fn, higher_is_better));
        self.register_scorer(scorer);
    }

    /// Get a custom scorer by name
    pub fn get_scorer(&self, name: &str) -> Option<&Arc<dyn CustomScorer>> {
        self.custom_scorers.get(name)
    }

    /// List all registered custom scorers
    pub fn list_scorers(&self) -> Vec<&str> {
        self.custom_scorers.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a scorer is registered
    pub fn has_scorer(&self, name: &str) -> bool {
        self.custom_scorers.contains_key(name)
    }
}

/// Enhanced scoring configuration
#[derive(Debug, Clone)]
pub struct ScoringConfig {
    /// Primary scoring metric
    pub primary: String,
    /// Additional metrics to compute
    pub additional: Vec<String>,
    /// Whether to compute confidence intervals
    pub confidence_intervals: bool,
    /// Confidence level for intervals (0.95 = 95%)
    pub confidence_level: f64,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Random state for bootstrap sampling
    pub random_state: Option<u64>,
    /// Custom scorer registry
    pub scorer_registry: ScorerRegistry,
}

impl Default for ScoringConfig {
    fn default() -> Self {
        Self {
            primary: "accuracy".to_string(),
            additional: vec![],
            confidence_intervals: false,
            confidence_level: 0.95,
            n_bootstrap: 1000,
            random_state: None,
            scorer_registry: ScorerRegistry::new(),
        }
    }
}

impl ScoringConfig {
    /// Create a new scoring configuration with primary metric
    pub fn new(primary: &str) -> Self {
        Self {
            primary: primary.to_string(),
            ..Default::default()
        }
    }

    /// Add additional metrics
    pub fn with_additional_metrics(mut self, metrics: Vec<String>) -> Self {
        self.additional = metrics;
        self
    }

    /// Enable confidence intervals
    pub fn with_confidence_intervals(mut self, level: f64, n_bootstrap: usize) -> Self {
        self.confidence_intervals = true;
        self.confidence_level = level;
        self.n_bootstrap = n_bootstrap;
        self
    }

    /// Set random state for reproducibility
    pub fn with_random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Register a custom scorer
    pub fn with_custom_scorer(mut self, scorer: Arc<dyn CustomScorer>) -> Self {
        self.scorer_registry.register_scorer(scorer);
        self
    }

    /// Register a custom scorer from a closure
    pub fn with_closure_scorer<F>(
        mut self,
        name: String,
        scorer_fn: F,
        higher_is_better: bool,
    ) -> Self
    where
        F: Fn(&Array1<Float>, &Array1<Float>) -> Result<f64> + Send + Sync + 'static,
    {
        self.scorer_registry
            .register_closure_scorer(name, scorer_fn, higher_is_better);
        self
    }

    /// Get a mutable reference to the scorer registry
    pub fn scorer_registry_mut(&mut self) -> &mut ScorerRegistry {
        &mut self.scorer_registry
    }

    /// Get a reference to the scorer registry
    pub fn scorer_registry(&self) -> &ScorerRegistry {
        &self.scorer_registry
    }
}

/// Scoring result with confidence intervals and multiple metrics
#[derive(Debug, Clone)]
pub struct ScoringResult {
    /// Primary metric scores
    pub primary_scores: Array1<f64>,
    /// Additional metric scores
    pub additional_scores: HashMap<String, Array1<f64>>,
    /// Confidence intervals for primary metric
    pub confidence_interval: Option<(f64, f64)>,
    /// Confidence intervals for additional metrics
    pub additional_confidence_intervals: HashMap<String, (f64, f64)>,
    /// Mean scores
    pub mean_scores: HashMap<String, f64>,
    /// Standard deviations
    pub std_scores: HashMap<String, f64>,
}

impl ScoringResult {
    /// Get the primary metric mean score
    pub fn primary_mean(&self) -> f64 {
        self.mean_scores
            .get(&"primary".to_string())
            .copied()
            .unwrap_or(0.0)
    }

    /// Get mean score for a specific metric
    pub fn mean_score(&self, metric: &str) -> Option<f64> {
        self.mean_scores.get(metric).copied()
    }

    /// Get all mean scores
    pub fn all_mean_scores(&self) -> &HashMap<String, f64> {
        &self.mean_scores
    }
}

/// Enhanced scorer that supports multiple metrics and confidence intervals
pub struct EnhancedScorer {
    config: ScoringConfig,
}

impl EnhancedScorer {
    /// Create a new enhanced scorer
    pub fn new(config: ScoringConfig) -> Self {
        Self { config }
    }

    /// Score predictions with multiple metrics and confidence intervals
    pub fn score_predictions(
        &self,
        y_true_splits: &[Array1<Float>],
        y_pred_splits: &[Array1<Float>],
        task_type: TaskType,
    ) -> Result<ScoringResult> {
        if y_true_splits.len() != y_pred_splits.len() {
            return Err(SklearsError::InvalidInput(
                "Number of true and predicted splits must match".to_string(),
            ));
        }

        let n_splits = y_true_splits.len();
        let mut primary_scores = Vec::with_capacity(n_splits);
        let mut additional_scores: HashMap<String, Vec<f64>> = HashMap::new();

        // Initialize additional scores storage
        for metric in &self.config.additional {
            additional_scores.insert(metric.clone(), Vec::with_capacity(n_splits));
        }

        // Compute scores for each split
        for (y_true, y_pred) in y_true_splits.iter().zip(y_pred_splits.iter()) {
            // Primary metric
            let primary_score =
                self.compute_metric_score(&self.config.primary, y_true, y_pred, task_type)?;
            primary_scores.push(primary_score);

            // Additional metrics
            for metric in &self.config.additional {
                let score = self.compute_metric_score(metric, y_true, y_pred, task_type)?;
                additional_scores.get_mut(metric).unwrap().push(score);
            }
        }

        // Convert to arrays
        let primary_scores_array = Array1::from_vec(primary_scores.clone());
        let mut additional_scores_arrays = HashMap::new();
        for (metric, scores) in additional_scores.iter() {
            additional_scores_arrays.insert(metric.clone(), Array1::from_vec(scores.clone()));
        }

        // Compute confidence intervals if requested
        let confidence_interval = if self.config.confidence_intervals {
            Some(self.bootstrap_confidence_interval(&primary_scores)?)
        } else {
            None
        };

        let mut additional_confidence_intervals = HashMap::new();
        if self.config.confidence_intervals {
            for (metric, scores) in &additional_scores {
                let ci = self.bootstrap_confidence_interval(scores)?;
                additional_confidence_intervals.insert(metric.clone(), ci);
            }
        }

        // Compute mean and std
        let mut mean_scores = HashMap::new();
        let mut std_scores = HashMap::new();

        mean_scores.insert("primary".to_string(), primary_scores_array.mean().unwrap());
        std_scores.insert("primary".to_string(), primary_scores_array.std(1.0));

        for (metric, scores) in &additional_scores_arrays {
            mean_scores.insert(metric.clone(), scores.mean().unwrap());
            std_scores.insert(metric.clone(), scores.std(1.0));
        }

        Ok(ScoringResult {
            primary_scores: primary_scores_array,
            additional_scores: additional_scores_arrays,
            confidence_interval,
            additional_confidence_intervals,
            mean_scores,
            std_scores,
        })
    }

    /// Compute score for a specific metric
    fn compute_metric_score(
        &self,
        metric: &str,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
        task_type: TaskType,
    ) -> Result<f64> {
        // First check if it's a custom scorer
        if let Some(custom_scorer) = self.config.scorer_registry.get_scorer(metric) {
            return custom_scorer.score(y_true, y_pred);
        }

        // Otherwise use built-in scorers
        match task_type {
            TaskType::Classification => self.compute_classification_score(metric, y_true, y_pred),
            TaskType::Regression => self.compute_regression_score(metric, y_true, y_pred),
        }
    }

    fn compute_classification_score(
        &self,
        metric: &str,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<f64> {
        // Convert float arrays to integer arrays for classification metrics
        let y_true_int: Array1<i32> = y_true.mapv(|x| x as i32);
        let y_pred_int: Array1<i32> = y_pred.mapv(|x| x as i32);

        let score = match metric {
            "accuracy" => accuracy_score(&y_true_int, &y_pred_int)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            "precision" => precision_score(&y_true_int, &y_pred_int, None)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            "recall" => recall_score(&y_true_int, &y_pred_int, None)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            "f1" => f1_score(&y_true_int, &y_pred_int, None)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown classification metric: {}",
                    metric
                )))
            }
        };

        Ok(score)
    }

    fn compute_regression_score(
        &self,
        metric: &str,
        y_true: &Array1<Float>,
        y_pred: &Array1<Float>,
    ) -> Result<f64> {
        let score = match metric {
            "r2" | "r2_score" => {
                r2_score(y_true, y_pred).map_err(|e| SklearsError::InvalidInput(e.to_string()))?
            }
            "neg_mean_squared_error" => -mean_squared_error(y_true, y_pred)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            "neg_mean_absolute_error" => -mean_absolute_error(y_true, y_pred)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            "explained_variance" => explained_variance_score(y_true, y_pred)
                .map_err(|e| SklearsError::InvalidInput(e.to_string()))?,
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown regression metric: {}",
                    metric
                )))
            }
        };

        Ok(score)
    }

    /// Compute bootstrap confidence interval
    fn bootstrap_confidence_interval(&self, scores: &[f64]) -> Result<(f64, f64)> {
        let mut rng = match self.config.random_state {
            Some(seed) => StdRng::seed_from_u64(seed),
            None => StdRng::seed_from_u64(42),
        };

        let n_scores = scores.len();
        let mut bootstrap_means = Vec::with_capacity(self.config.n_bootstrap);

        for _ in 0..self.config.n_bootstrap {
            let mut bootstrap_sample = Vec::with_capacity(n_scores);
            for _ in 0..n_scores {
                let idx = rng.gen_range(0..n_scores);
                bootstrap_sample.push(scores[idx]);
            }

            let mean = bootstrap_sample.iter().sum::<f64>() / n_scores as f64;
            bootstrap_means.push(mean);
        }

        bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

        let alpha = 1.0 - self.config.confidence_level;
        let lower_idx = ((alpha / 2.0) * self.config.n_bootstrap as f64) as usize;
        let upper_idx = ((1.0 - alpha / 2.0) * self.config.n_bootstrap as f64) as usize;

        let lower = bootstrap_means[lower_idx.min(self.config.n_bootstrap - 1)];
        let upper = bootstrap_means[upper_idx.min(self.config.n_bootstrap - 1)];

        Ok((lower, upper))
    }
}

/// Task type for scoring
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    /// Classification
    Classification,
    /// Regression
    Regression,
}

/// Statistical significance test result
#[derive(Debug, Clone)]
pub struct SignificanceTestResult {
    /// Test statistic
    pub statistic: f64,
    /// P-value
    pub p_value: f64,
    /// Whether the result is significant at alpha level
    pub is_significant: bool,
    /// Alpha level used
    pub alpha: f64,
    /// Test name
    pub test_name: String,
}

/// Perform paired t-test for comparing two sets of CV scores
pub fn paired_ttest(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
    alpha: f64,
) -> Result<SignificanceTestResult> {
    if scores1.len() != scores2.len() {
        return Err(SklearsError::InvalidInput(
            "Score arrays must have the same length".to_string(),
        ));
    }

    let n = scores1.len() as f64;
    if n < 2.0 {
        return Err(SklearsError::InvalidInput(
            "Need at least 2 samples for t-test".to_string(),
        ));
    }

    // Compute differences
    let differences: Array1<f64> = scores1 - scores2;
    let mean_diff = differences.mean().unwrap();
    let std_diff = differences.std(1.0);

    if std_diff == 0.0 {
        return Err(SklearsError::InvalidInput(
            "Standard deviation of differences is zero".to_string(),
        ));
    }

    // Compute t-statistic
    let t_stat = mean_diff * (n.sqrt()) / std_diff;

    // Compute p-value (two-tailed test)
    // Using approximation for t-distribution
    let df = n - 1.0;
    let p_value = 2.0 * (1.0 - student_t_cdf(t_stat.abs(), df));

    Ok(SignificanceTestResult {
        statistic: t_stat,
        p_value,
        is_significant: p_value < alpha,
        alpha,
        test_name: "Paired t-test".to_string(),
    })
}

/// Approximate CDF of Student's t-distribution
fn student_t_cdf(t: f64, df: f64) -> f64 {
    // Simple approximation using normal distribution for large df
    if df > 30.0 {
        return standard_normal_cdf(t);
    }

    // Basic approximation for small df
    let x = t / (df + t * t).sqrt();
    0.5 + 0.5 * x * (1.0 - x * x / 3.0)
}

/// Standard normal CDF approximation
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Error function approximation
fn erf(x: f64) -> f64 {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

/// Perform Wilcoxon signed-rank test (non-parametric alternative to t-test)
pub fn wilcoxon_signed_rank_test(
    scores1: &Array1<f64>,
    scores2: &Array1<f64>,
    alpha: f64,
) -> Result<SignificanceTestResult> {
    if scores1.len() != scores2.len() {
        return Err(SklearsError::InvalidInput(
            "Score arrays must have the same length".to_string(),
        ));
    }

    let differences: Vec<f64> = scores1
        .iter()
        .zip(scores2.iter())
        .map(|(a, b)| a - b)
        .filter(|&d| d != 0.0) // Remove zero differences
        .collect();

    let n = differences.len();
    if n < 5 {
        return Err(SklearsError::InvalidInput(
            "Need at least 5 non-zero differences for Wilcoxon test".to_string(),
        ));
    }

    // Rank absolute differences
    let mut abs_diffs_with_indices: Vec<(f64, usize, f64)> = differences
        .iter()
        .enumerate()
        .map(|(i, &d)| (d.abs(), i, d))
        .collect();

    abs_diffs_with_indices.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j < n && abs_diffs_with_indices[j].0 == abs_diffs_with_indices[i].0 {
            j += 1;
        }

        let rank = (i + j + 1) as f64 / 2.0; // Average rank for ties
        for k in i..j {
            ranks[abs_diffs_with_indices[k].1] = rank;
        }
        i = j;
    }

    // Sum of positive ranks
    let w_plus: f64 = differences
        .iter()
        .zip(&ranks)
        .filter(|(&d, _)| d > 0.0)
        .map(|(_, &rank)| rank)
        .sum();

    // Expected value and variance under null hypothesis
    let expected = n as f64 * (n + 1) as f64 / 4.0;
    let variance = n as f64 * (n + 1) as f64 * (2 * n + 1) as f64 / 24.0;

    // Z-statistic with continuity correction
    let z = if w_plus > expected {
        (w_plus - 0.5 - expected) / variance.sqrt()
    } else {
        (w_plus + 0.5 - expected) / variance.sqrt()
    };

    let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));

    Ok(SignificanceTestResult {
        statistic: w_plus,
        p_value,
        is_significant: p_value < alpha,
        alpha,
        test_name: "Wilcoxon signed-rank test".to_string(),
    })
}
