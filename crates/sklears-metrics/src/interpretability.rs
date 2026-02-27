//! Interpretability Metrics for Machine Learning Model Explanations
//!
//! This module provides comprehensive metrics for evaluating the quality, reliability,
//! and trustworthiness of machine learning model explanations. These metrics are essential
//! for assessing explanation methods like SHAP, LIME, integrated gradients, and other
//! interpretability techniques.
//!
//! # Features
//!
//! - Faithfulness metrics for measuring how well explanations reflect model behavior
//! - Stability metrics for assessing explanation consistency across similar inputs
//! - Comprehensibility measures for human understanding of explanations
//! - Trustworthiness metrics combining multiple quality indicators
//! - Explanation quality assessment with statistical validation
//! - Comparative analysis between different explanation methods
//! - Feature importance ranking validation and consistency checks
//!
//! # Examples
//!
//! ```rust
//! use sklears_metrics::interpretability::*;
//! use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
//!
//! // Model predictions and feature importance explanations
//! let predictions = Array1::from_vec(vec![0.8, 0.6, 0.9, 0.3]);
//! let explanations = Array2::from_shape_vec((4, 3), vec![
//!     0.5, 0.3, 0.2,  // Explanation for sample 1
//!     0.4, 0.4, 0.2,  // Explanation for sample 2
//!     0.6, 0.2, 0.2,  // Explanation for sample 3
//!     0.2, 0.5, 0.3,  // Explanation for sample 4
//! ]).unwrap();
//!
//! // Calculate faithfulness using feature removal
//! let faithfulness = calculate_faithfulness_removal(
//!     &predictions,
//!     &explanations,
//!     |features| Array1::from_vec(vec![0.5; features.nrows()]), // Simple mock model function
//!     0.1 // Removal threshold
//! ).unwrap();
//!
//! println!("Faithfulness score: {:.3}", faithfulness.score);
//!
//! // Calculate explanation stability
//! let stability = calculate_explanation_stability(
//!     &explanations,
//!     StabilityMetric::Correlation
//! ).unwrap();
//!
//! println!("Stability score: {:.3}", stability.mean_stability);
//! ```

use crate::{MetricsError, MetricsResult};
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::SeedableRng;
use std::collections::{HashMap, HashSet};

/// Configuration for interpretability analysis
#[derive(Debug, Clone)]
pub struct InterpretabilityConfig {
    /// Number of perturbations for stability testing
    pub n_perturbations: usize,
    /// Perturbation magnitude (fraction of feature range)
    pub perturbation_magnitude: f64,
    /// Significance level for statistical tests
    pub significance_level: f64,
    /// Top-k features to consider for ranking metrics
    pub top_k_features: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for InterpretabilityConfig {
    fn default() -> Self {
        Self {
            n_perturbations: 100,
            perturbation_magnitude: 0.1,
            significance_level: 0.05,
            top_k_features: 10,
            seed: Some(42),
        }
    }
}

/// Different metrics for measuring explanation stability
#[derive(Debug, Clone, Copy)]
pub enum StabilityMetric {
    /// Pearson correlation between explanations
    Correlation,
    /// Cosine similarity between explanations
    CosineSimilarity,
    /// Rank correlation (Spearman)
    RankCorrelation,
    /// Jaccard similarity for top-k features
    TopKJaccard,
}

/// Methods for measuring explanation faithfulness
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FaithfulnessMethod {
    /// Feature removal/occlusion based
    FeatureRemoval,
    /// Feature permutation based
    FeaturePermutation,
    /// Gradient correlation
    GradientCorrelation,
    /// Sufficiency measure
    Sufficiency,
    /// Comprehensiveness measure
    Comprehensiveness,
}

/// Result of faithfulness evaluation
#[derive(Debug, Clone)]
pub struct FaithfulnessResult {
    /// Overall faithfulness score (0-1, higher is better)
    pub score: f64,
    /// Method used for evaluation
    pub method: FaithfulnessMethod,
    /// Individual sample faithfulness scores
    pub sample_scores: Array1<f64>,
    /// Statistical significance of the result
    pub p_value: Option<f64>,
    /// Confidence interval for the score
    pub confidence_interval: Option<(f64, f64)>,
    /// Additional metrics
    pub metadata: HashMap<String, f64>,
}

/// Result of stability analysis
#[derive(Debug, Clone)]
pub struct StabilityResult {
    /// Mean stability score across all pairs
    pub mean_stability: f64,
    /// Standard deviation of stability scores
    pub std_stability: f64,
    /// Minimum stability score
    pub min_stability: f64,
    /// Maximum stability score
    pub max_stability: f64,
    /// Pairwise stability matrix
    pub pairwise_scores: Array2<f64>,
    /// Metric used for evaluation
    pub metric: StabilityMetric,
}

/// Result of comprehensibility assessment
#[derive(Debug, Clone)]
pub struct ComprehensibilityResult {
    /// Overall comprehensibility score (0-1, higher is better)
    pub score: f64,
    /// Explanation complexity measure
    pub complexity: f64,
    /// Sparsity of explanations (fraction of zero/negligible features)
    pub sparsity: f64,
    /// Consistency across similar samples
    pub consistency: f64,
    /// Feature importance distribution entropy
    pub entropy: f64,
}

/// Comprehensive trustworthiness assessment
#[derive(Debug, Clone)]
pub struct TrustworthinessResult {
    /// Overall trustworthiness score (0-1, higher is better)
    pub overall_score: f64,
    /// Individual component scores
    pub faithfulness_score: f64,
    pub stability_score: f64,
    pub comprehensibility_score: f64,
    pub consistency_score: f64,
    /// Weighted combination weights used
    pub weights: HashMap<String, f64>,
    /// Confidence in the assessment
    pub confidence: f64,
}

/// Quality assessment for feature importance rankings
#[derive(Debug, Clone)]
pub struct RankingQualityResult {
    /// Ranking consistency score
    pub consistency: f64,
    /// Top-k stability (how stable are the most important features)
    pub top_k_stability: f64,
    /// Rank correlation with ground truth (if available)
    pub ground_truth_correlation: Option<f64>,
    /// Discriminative power of top features
    pub discriminative_power: f64,
}

/// Calculate faithfulness using feature removal method
pub fn calculate_faithfulness_removal<F>(
    original_predictions: &Array1<f64>,
    explanations: &Array2<f64>,
    model_fn: F,
    removal_threshold: f64,
) -> MetricsResult<FaithfulnessResult>
where
    F: Fn(&Array2<f64>) -> Array1<f64>,
{
    if original_predictions.len() != explanations.nrows() {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![explanations.nrows()],
            actual: vec![original_predictions.len()],
        });
    }

    let n_samples = explanations.nrows();
    let n_features = explanations.ncols();
    let mut sample_scores = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let explanation = explanations.row(i);
        let original_pred = original_predictions[i];

        // Create perturbed input by removing features based on importance
        let mut perturbed_features = Array2::ones((1, n_features));

        // Remove features with importance above threshold
        for j in 0..n_features {
            if explanation[j].abs() > removal_threshold {
                perturbed_features[[0, j]] = 0.0; // Remove feature
            }
        }

        // Get prediction on perturbed input
        let perturbed_pred = model_fn(&perturbed_features)[0];

        // Calculate faithfulness as correlation between importance and prediction change
        let prediction_change = (original_pred - perturbed_pred).abs();
        let importance_sum: f64 = explanation
            .iter()
            .filter(|&&imp| imp.abs() > removal_threshold)
            .map(|imp| imp.abs())
            .sum();

        // Normalize faithfulness score
        let faithfulness = if importance_sum > 0.0 {
            prediction_change / (importance_sum + 1e-8)
        } else {
            0.0
        };

        sample_scores[i] = faithfulness.min(1.0);
    }

    let overall_score = sample_scores.mean().unwrap_or(0.0);

    // Calculate confidence interval using bootstrap
    let confidence_interval = bootstrap_confidence_interval(&sample_scores, 0.95, 1000);

    let mut metadata = HashMap::new();
    metadata.insert("removal_threshold".to_string(), removal_threshold);
    metadata.insert("n_samples".to_string(), n_samples as f64);
    metadata.insert("n_features".to_string(), n_features as f64);

    Ok(FaithfulnessResult {
        score: overall_score,
        method: FaithfulnessMethod::FeatureRemoval,
        sample_scores,
        p_value: None,
        confidence_interval: Some(confidence_interval),
        metadata,
    })
}

/// Calculate faithfulness using feature permutation method
pub fn calculate_faithfulness_permutation<F>(
    original_predictions: &Array1<f64>,
    explanations: &Array2<f64>,
    original_features: &Array2<f64>,
    model_fn: F,
    n_permutations: usize,
) -> MetricsResult<FaithfulnessResult>
where
    F: Fn(&Array2<f64>) -> Array1<f64>,
{
    if original_predictions.len() != explanations.nrows()
        || original_predictions.len() != original_features.nrows()
    {
        return Err(MetricsError::ShapeMismatch {
            expected: vec![explanations.nrows()],
            actual: vec![original_predictions.len(), original_features.nrows()],
        });
    }

    let n_samples = explanations.nrows();
    let n_features = explanations.ncols();
    let mut sample_scores = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let explanation = explanations.row(i);
        let original_pred = original_predictions[i];
        let features = original_features.row(i);

        let _correlation_sum = 0.0;

        // Permute features based on importance ranking
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            explanation[b]
                .abs()
                .partial_cmp(&explanation[a].abs())
                .unwrap()
        });

        // Incrementally permute features and measure prediction change
        let mut permuted_features = features.to_owned();
        let mut prediction_changes = Vec::new();
        let mut importance_weights = Vec::new();

        for &feature_idx in &feature_indices {
            // Permute this feature
            permuted_features[feature_idx] = 0.0; // Or use random value

            let mut perturbed_input = Array2::zeros((1, n_features));
            perturbed_input.row_mut(0).assign(&permuted_features);

            let perturbed_pred = model_fn(&perturbed_input)[0];
            let change = (original_pred - perturbed_pred).abs();

            prediction_changes.push(change);
            importance_weights.push(explanation[feature_idx].abs());
        }

        // Calculate correlation between importance and prediction changes
        if prediction_changes.len() >= 2 {
            let correlation = pearson_correlation(&importance_weights, &prediction_changes);
            sample_scores[i] = correlation.max(0.0); // Clamp to non-negative
        }
    }

    let overall_score = sample_scores.mean().unwrap_or(0.0);
    let confidence_interval = bootstrap_confidence_interval(&sample_scores, 0.95, 1000);

    let mut metadata = HashMap::new();
    metadata.insert("n_permutations".to_string(), n_permutations as f64);

    Ok(FaithfulnessResult {
        score: overall_score,
        method: FaithfulnessMethod::FeaturePermutation,
        sample_scores,
        p_value: None,
        confidence_interval: Some(confidence_interval),
        metadata,
    })
}

/// Calculate explanation stability across similar inputs
pub fn calculate_explanation_stability(
    explanations: &Array2<f64>,
    metric: StabilityMetric,
) -> MetricsResult<StabilityResult> {
    if explanations.nrows() < 2 {
        return Err(MetricsError::InvalidParameter(
            "Need at least 2 explanations for stability analysis".to_string(),
        ));
    }

    let n_samples = explanations.nrows();
    let mut pairwise_scores = Array2::zeros((n_samples, n_samples));

    // Calculate pairwise stability scores
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let explanation1 = explanations.row(i);
            let explanation2 = explanations.row(j);

            let similarity = match metric {
                StabilityMetric::Correlation => {
                    pearson_correlation(&explanation1.to_vec(), &explanation2.to_vec())
                }
                StabilityMetric::CosineSimilarity => {
                    cosine_similarity(&explanation1, &explanation2)
                }
                StabilityMetric::RankCorrelation => {
                    spearman_correlation(&explanation1.to_vec(), &explanation2.to_vec())
                }
                StabilityMetric::TopKJaccard => {
                    top_k_jaccard_similarity(&explanation1, &explanation2, 5)
                }
            };

            pairwise_scores[[i, j]] = similarity;
            pairwise_scores[[j, i]] = similarity; // Symmetric
        }
        pairwise_scores[[i, i]] = 1.0; // Self-similarity
    }

    // Extract upper triangle for statistics (excluding diagonal)
    let mut similarities = Vec::new();
    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            similarities.push(pairwise_scores[[i, j]]);
        }
    }

    let mean_stability = similarities.iter().sum::<f64>() / similarities.len() as f64;
    let variance = similarities
        .iter()
        .map(|x| (x - mean_stability).powi(2))
        .sum::<f64>()
        / similarities.len() as f64;
    let std_stability = variance.sqrt();
    let min_stability = similarities.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_stability = similarities
        .iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    Ok(StabilityResult {
        mean_stability,
        std_stability,
        min_stability,
        max_stability,
        pairwise_scores,
        metric,
    })
}

/// Calculate comprehensibility of explanations
pub fn calculate_comprehensibility(
    explanations: &Array2<f64>,
    sparsity_threshold: f64,
) -> MetricsResult<ComprehensibilityResult> {
    let n_samples = explanations.nrows();
    let n_features = explanations.ncols();

    // Calculate sparsity (fraction of features below threshold)
    let total_elements = (n_samples * n_features) as f64;
    let sparse_elements = explanations
        .iter()
        .filter(|&&x| x.abs() < sparsity_threshold)
        .count() as f64;
    let sparsity = sparse_elements / total_elements;

    // Calculate complexity as the number of "significant" features per explanation
    let mut complexity_scores = Vec::new();
    for i in 0..n_samples {
        let significant_features = explanations
            .row(i)
            .iter()
            .filter(|&&x| x.abs() >= sparsity_threshold)
            .count();
        complexity_scores.push(significant_features as f64);
    }
    let complexity = complexity_scores.iter().sum::<f64>() / n_samples as f64;
    let normalized_complexity = complexity / n_features as f64;

    // Calculate consistency across samples using variance of feature importance
    let mut feature_variances = Vec::new();
    for j in 0..n_features {
        let feature_column: Vec<f64> = (0..n_samples).map(|i| explanations[[i, j]]).collect();
        let variance = calculate_variance(&feature_column);
        feature_variances.push(variance);
    }
    let mean_variance = feature_variances.iter().sum::<f64>() / n_features as f64;
    let consistency = 1.0 / (1.0 + mean_variance); // Higher variance = lower consistency

    // Calculate entropy of feature importance distribution
    let mut all_importances: Vec<f64> = explanations.iter().map(|&x| x.abs()).collect();
    all_importances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let entropy = calculate_entropy(&all_importances, 10); // 10 bins

    // Overall comprehensibility score (weighted combination)
    let score = 0.3 * sparsity
        + 0.2 * (1.0 - normalized_complexity)
        + 0.3 * consistency
        + 0.2 * (1.0 - entropy);

    Ok(ComprehensibilityResult {
        score,
        complexity: normalized_complexity,
        sparsity,
        consistency,
        entropy,
    })
}

/// Calculate comprehensive trustworthiness assessment
pub fn calculate_trustworthiness<F>(
    explanations: &Array2<f64>,
    predictions: &Array1<f64>,
    _features: &Array2<f64>,
    model_fn: F,
    _config: &InterpretabilityConfig,
) -> MetricsResult<TrustworthinessResult>
where
    F: Fn(&Array2<f64>) -> Array1<f64> + Copy,
{
    // Calculate individual components
    let faithfulness_result = calculate_faithfulness_removal(
        predictions,
        explanations,
        model_fn,
        0.1, // threshold
    )?;

    let stability_result =
        calculate_explanation_stability(explanations, StabilityMetric::Correlation)?;

    let comprehensibility_result = calculate_comprehensibility(explanations, 0.01)?;

    // Calculate consistency score (similar to stability but different metric)
    let consistency_score = calculate_consistency_score(explanations)?;

    // Define weights for combining scores
    let mut weights = HashMap::new();
    weights.insert("faithfulness".to_string(), 0.4);
    weights.insert("stability".to_string(), 0.3);
    weights.insert("comprehensibility".to_string(), 0.2);
    weights.insert("consistency".to_string(), 0.1);

    // Calculate weighted overall score
    let overall_score = weights["faithfulness"] * faithfulness_result.score
        + weights["stability"] * stability_result.mean_stability
        + weights["comprehensibility"] * comprehensibility_result.score
        + weights["consistency"] * consistency_score;

    // Calculate confidence based on variance of individual scores
    let score_variance = calculate_variance(&[
        faithfulness_result.score,
        stability_result.mean_stability,
        comprehensibility_result.score,
        consistency_score,
    ]);
    let confidence = 1.0 / (1.0 + score_variance);

    Ok(TrustworthinessResult {
        overall_score,
        faithfulness_score: faithfulness_result.score,
        stability_score: stability_result.mean_stability,
        comprehensibility_score: comprehensibility_result.score,
        consistency_score,
        weights,
        confidence,
    })
}

/// Evaluate quality of feature importance rankings
pub fn evaluate_ranking_quality(
    explanations: &Array2<f64>,
    ground_truth_rankings: Option<&Array2<f64>>,
    top_k: usize,
) -> MetricsResult<RankingQualityResult> {
    let _n_samples = explanations.nrows();

    // Calculate consistency across samples
    let consistency = calculate_ranking_consistency(explanations, top_k)?;

    // Calculate top-k stability
    let top_k_stability = calculate_top_k_stability(explanations, top_k)?;

    // Calculate ground truth correlation if available
    let ground_truth_correlation = if let Some(gt_rankings) = ground_truth_rankings {
        Some(calculate_ranking_correlation(explanations, gt_rankings)?)
    } else {
        None
    };

    // Calculate discriminative power of top features
    let discriminative_power = calculate_discriminative_power(explanations, top_k)?;

    Ok(RankingQualityResult {
        consistency,
        top_k_stability,
        ground_truth_correlation,
        discriminative_power,
    })
}

// Helper functions

/// Calculate Pearson correlation coefficient
fn pearson_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let n = x.len() as f64;
    let mean_x = x.iter().sum::<f64>() / n;
    let mean_y = y.iter().sum::<f64>() / n;

    let mut numerator = 0.0;
    let mut sum_sq_x = 0.0;
    let mut sum_sq_y = 0.0;

    for (&xi, &yi) in x.iter().zip(y.iter()) {
        let dx = xi - mean_x;
        let dy = yi - mean_y;
        numerator += dx * dy;
        sum_sq_x += dx * dx;
        sum_sq_y += dy * dy;
    }

    let denominator = (sum_sq_x * sum_sq_y).sqrt();
    if denominator > f64::EPSILON {
        numerator / denominator
    } else {
        0.0
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &ArrayView1<f64>, b: &ArrayView1<f64>) -> f64 {
    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a > f64::EPSILON && norm_b > f64::EPSILON {
        dot_product / (norm_a * norm_b)
    } else {
        0.0
    }
}

/// Calculate Spearman rank correlation
fn spearman_correlation(x: &[f64], y: &[f64]) -> f64 {
    if x.len() != y.len() || x.len() < 2 {
        return 0.0;
    }

    let ranks_x = calculate_ranks(x);
    let ranks_y = calculate_ranks(y);

    pearson_correlation(&ranks_x, &ranks_y)
}

/// Calculate ranks for Spearman correlation
fn calculate_ranks(values: &[f64]) -> Vec<f64> {
    let mut indexed_values: Vec<(usize, f64)> =
        values.iter().enumerate().map(|(i, &v)| (i, v)).collect();
    indexed_values.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    let mut ranks = vec![0.0; values.len()];
    for (rank, &(original_index, _)) in indexed_values.iter().enumerate() {
        ranks[original_index] = (rank + 1) as f64;
    }

    ranks
}

/// Calculate top-k Jaccard similarity
fn top_k_jaccard_similarity(a: &ArrayView1<f64>, b: &ArrayView1<f64>, k: usize) -> f64 {
    let top_k_a = get_top_k_indices(a, k);
    let top_k_b = get_top_k_indices(b, k);

    let intersection: HashSet<_> = top_k_a.intersection(&top_k_b).collect();
    let union: HashSet<_> = top_k_a.union(&top_k_b).collect();

    intersection.len() as f64 / union.len() as f64
}

/// Get indices of top-k largest values
fn get_top_k_indices(values: &ArrayView1<f64>, k: usize) -> HashSet<usize> {
    let mut indexed_values: Vec<(usize, f64)> = values
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v.abs()))
        .collect();
    indexed_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    indexed_values.iter().take(k).map(|(i, _)| *i).collect()
}

/// Calculate variance of a slice
fn calculate_variance(values: &[f64]) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }

    let mean = values.iter().sum::<f64>() / values.len() as f64;
    let variance = values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;

    variance
}

/// Calculate entropy of values using binning
fn calculate_entropy(values: &[f64], n_bins: usize) -> f64 {
    if values.is_empty() {
        return 0.0;
    }

    let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));

    if (max_val - min_val).abs() < f64::EPSILON {
        return 0.0;
    }

    let bin_width = (max_val - min_val) / n_bins as f64;
    let mut bin_counts = vec![0; n_bins];

    for &value in values {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(n_bins - 1);
        bin_counts[bin_idx] += 1;
    }

    let total_count = values.len() as f64;
    let mut entropy = 0.0;

    for count in bin_counts {
        if count > 0 {
            let p = count as f64 / total_count;
            entropy -= p * p.ln();
        }
    }

    entropy
}

/// Bootstrap confidence interval
fn bootstrap_confidence_interval(
    values: &Array1<f64>,
    confidence_level: f64,
    n_bootstrap: usize,
) -> (f64, f64) {
    use scirs2_core::random::Rng;
    let mut rng = StdRng::seed_from_u64(42);

    let mut bootstrap_means = Vec::with_capacity(n_bootstrap);
    let n = values.len();

    for _ in 0..n_bootstrap {
        let mut bootstrap_sample = Vec::with_capacity(n);
        for _ in 0..n {
            let idx = rng.gen_range(0..n);
            bootstrap_sample.push(values[idx]);
        }
        let mean = bootstrap_sample.iter().sum::<f64>() / n as f64;
        bootstrap_means.push(mean);
    }

    bootstrap_means.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let alpha = 1.0 - confidence_level;
    let lower_idx = (alpha / 2.0 * n_bootstrap as f64) as usize;
    let upper_idx = ((1.0 - alpha / 2.0) * n_bootstrap as f64) as usize;

    (bootstrap_means[lower_idx], bootstrap_means[upper_idx])
}

/// Calculate consistency score across explanations
fn calculate_consistency_score(explanations: &Array2<f64>) -> MetricsResult<f64> {
    let n_features = explanations.ncols();
    let mut feature_consistencies = Vec::new();

    for j in 0..n_features {
        let feature_column: Vec<f64> = (0..explanations.nrows())
            .map(|i| explanations[[i, j]])
            .collect();

        // Consistency as inverse of coefficient of variation
        let mean = feature_column.iter().sum::<f64>() / feature_column.len() as f64;
        let std = calculate_variance(&feature_column).sqrt();
        let cv = if mean.abs() > f64::EPSILON {
            std / mean.abs()
        } else {
            0.0
        };
        let consistency = 1.0 / (1.0 + cv);

        feature_consistencies.push(consistency);
    }

    Ok(feature_consistencies.iter().sum::<f64>() / n_features as f64)
}

/// Calculate ranking consistency across samples
fn calculate_ranking_consistency(explanations: &Array2<f64>, _top_k: usize) -> MetricsResult<f64> {
    let n_samples = explanations.nrows();
    let mut pairwise_correlations = Vec::new();

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let rank_corr =
                spearman_correlation(&explanations.row(i).to_vec(), &explanations.row(j).to_vec());
            pairwise_correlations.push(rank_corr);
        }
    }

    let mean_correlation =
        pairwise_correlations.iter().sum::<f64>() / pairwise_correlations.len() as f64;
    Ok(mean_correlation.max(0.0))
}

/// Calculate top-k stability
fn calculate_top_k_stability(explanations: &Array2<f64>, top_k: usize) -> MetricsResult<f64> {
    let n_samples = explanations.nrows();
    let mut jaccard_similarities = Vec::new();

    for i in 0..n_samples {
        for j in (i + 1)..n_samples {
            let jaccard =
                top_k_jaccard_similarity(&explanations.row(i), &explanations.row(j), top_k);
            jaccard_similarities.push(jaccard);
        }
    }

    let mean_jaccard = jaccard_similarities.iter().sum::<f64>() / jaccard_similarities.len() as f64;
    Ok(mean_jaccard)
}

/// Calculate correlation with ground truth rankings
fn calculate_ranking_correlation(
    explanations: &Array2<f64>,
    ground_truth: &Array2<f64>,
) -> MetricsResult<f64> {
    if explanations.shape() != ground_truth.shape() {
        return Err(MetricsError::ShapeMismatch {
            expected: explanations.shape().to_vec(),
            actual: ground_truth.shape().to_vec(),
        });
    }

    let n_samples = explanations.nrows();
    let mut correlations = Vec::new();

    for i in 0..n_samples {
        let correlation =
            spearman_correlation(&explanations.row(i).to_vec(), &ground_truth.row(i).to_vec());
        correlations.push(correlation);
    }

    let mean_correlation = correlations.iter().sum::<f64>() / correlations.len() as f64;
    Ok(mean_correlation)
}

/// Calculate discriminative power of top features
fn calculate_discriminative_power(explanations: &Array2<f64>, top_k: usize) -> MetricsResult<f64> {
    let n_samples = explanations.nrows();
    let n_features = explanations.ncols();

    // Calculate average importance for each feature across all samples
    let mut feature_importance_means = Vec::new();
    for j in 0..n_features {
        let mean_importance: f64 = (0..n_samples)
            .map(|i| explanations[[i, j]].abs())
            .sum::<f64>()
            / n_samples as f64;
        feature_importance_means.push(mean_importance);
    }

    // Get top-k features by average importance
    let mut indexed_importance: Vec<(usize, f64)> = feature_importance_means
        .iter()
        .enumerate()
        .map(|(i, &imp)| (i, imp))
        .collect();
    indexed_importance.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let top_k_importance: f64 = indexed_importance
        .iter()
        .take(top_k)
        .map(|(_, imp)| imp)
        .sum();
    let total_importance: f64 = feature_importance_means.iter().sum();

    // Discriminative power as ratio of top-k importance to total
    let discriminative_power = if total_importance > f64::EPSILON {
        top_k_importance / total_importance
    } else {
        0.0
    };

    Ok(discriminative_power)
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::Axis;

    // Mock model function for testing
    fn mock_model_predict(features: &Array2<f64>) -> Array1<f64> {
        features.sum_axis(Axis(1))
    }

    #[test]
    fn test_faithfulness_removal() {
        let predictions = Array1::from_vec(vec![1.0, 2.0, 3.0]);
        let explanations =
            Array2::from_shape_vec((3, 2), vec![0.5, 0.5, 0.3, 0.7, 0.8, 0.2]).unwrap();

        let result =
            calculate_faithfulness_removal(&predictions, &explanations, mock_model_predict, 0.4)
                .unwrap();

        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert_eq!(result.sample_scores.len(), 3);
        assert_eq!(result.method, FaithfulnessMethod::FeatureRemoval);
    }

    #[test]
    fn test_explanation_stability() {
        let explanations = Array2::from_shape_vec(
            (3, 4),
            vec![0.4, 0.3, 0.2, 0.1, 0.5, 0.2, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
        )
        .unwrap();

        let result =
            calculate_explanation_stability(&explanations, StabilityMetric::Correlation).unwrap();

        assert!(result.mean_stability >= -1.0 && result.mean_stability <= 1.0);
        assert_eq!(result.pairwise_scores.shape(), &[3, 3]);
        assert!(result.min_stability <= result.mean_stability);
        assert!(result.mean_stability <= result.max_stability);
    }

    #[test]
    fn test_comprehensibility() {
        let explanations = Array2::from_shape_vec(
            (2, 4),
            vec![
                0.8, 0.1, 0.05, 0.05, // Sparse explanation
                0.25, 0.25, 0.25, 0.25, // Dense explanation
            ],
        )
        .unwrap();

        let result = calculate_comprehensibility(&explanations, 0.1).unwrap();

        assert!(result.score >= 0.0 && result.score <= 1.0);
        assert!(result.sparsity >= 0.0 && result.sparsity <= 1.0);
        assert!(result.complexity >= 0.0 && result.complexity <= 1.0);
        assert!(result.consistency >= 0.0 && result.consistency <= 1.0);
    }

    #[test]
    fn test_pearson_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = pearson_correlation(&x, &y);
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);

        let y_neg = vec![10.0, 8.0, 6.0, 4.0, 2.0]; // Perfect negative correlation
        let correlation_neg = pearson_correlation(&x, &y_neg);
        assert_abs_diff_eq!(correlation_neg, -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = Array1::from_vec(vec![1.0, 0.0, 0.0]);
        let b = Array1::from_vec(vec![1.0, 0.0, 0.0]);

        let similarity = cosine_similarity(&a.view(), &b.view());
        assert_abs_diff_eq!(similarity, 1.0, epsilon = 1e-10);

        let c = Array1::from_vec(vec![0.0, 1.0, 0.0]);
        let similarity_orthogonal = cosine_similarity(&a.view(), &c.view());
        assert_abs_diff_eq!(similarity_orthogonal, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_spearman_correlation() {
        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![5.0, 6.0, 7.0, 8.0, 9.0]; // Perfect rank correlation

        let correlation = spearman_correlation(&x, &y);
        assert_abs_diff_eq!(correlation, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_top_k_jaccard() {
        let a = Array1::from_vec(vec![0.8, 0.6, 0.4, 0.2, 0.1]);
        let b = Array1::from_vec(vec![0.7, 0.5, 0.3, 0.1, 0.05]);

        let jaccard = top_k_jaccard_similarity(&a.view(), &b.view(), 3);
        assert!((0.0..=1.0).contains(&jaccard));
    }

    #[test]
    fn test_ranking_quality() {
        let explanations = Array2::from_shape_vec(
            (3, 4),
            vec![0.4, 0.3, 0.2, 0.1, 0.5, 0.2, 0.2, 0.1, 0.4, 0.3, 0.2, 0.1],
        )
        .unwrap();

        let result = evaluate_ranking_quality(&explanations, None, 2).unwrap();

        assert!(result.consistency >= 0.0 && result.consistency <= 1.0);
        assert!(result.top_k_stability >= 0.0 && result.top_k_stability <= 1.0);
        assert!(result.discriminative_power >= 0.0 && result.discriminative_power <= 1.0);
        assert!(result.ground_truth_correlation.is_none());
    }

    #[test]
    fn test_calculate_variance() {
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let variance = calculate_variance(&values);
        let expected_variance = 2.0; // Variance of 1,2,3,4,5
        assert_abs_diff_eq!(variance, expected_variance, epsilon = 1e-10);
    }

    #[test]
    fn test_calculate_entropy() {
        let values = vec![1.0, 1.0, 1.0, 1.0]; // All same values
        let entropy = calculate_entropy(&values, 4);
        assert_abs_diff_eq!(entropy, 0.0, epsilon = 1e-10);

        let uniform_values = vec![1.0, 2.0, 3.0, 4.0]; // Uniform distribution
        let entropy_uniform = calculate_entropy(&uniform_values, 4);
        assert!(entropy_uniform > 0.0);
    }

    #[test]
    fn test_bootstrap_confidence_interval() {
        let values = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let (lower, upper) = bootstrap_confidence_interval(&values, 0.95, 100);

        let mean = values.mean().unwrap();
        assert!(lower <= mean);
        assert!(mean <= upper);
        assert!(upper > lower);
    }
}
