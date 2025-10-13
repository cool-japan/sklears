//! Fairness and bias detection tools
//!
//! This module provides comprehensive tools for assessing model fairness, detecting bias,
//! and computing standard fairness metrics including demographic parity and equalized odds.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Fairness assessment result
#[derive(Debug, Clone)]
pub struct FairnessResult {
    /// Demographic parity metrics
    pub demographic_parity: DemographicParityResult,
    /// Equalized odds metrics
    pub equalized_odds: EqualizedOddsResult,
    /// Individual fairness metrics
    pub individual_fairness: IndividualFairnessResult,
    /// Bias detection results
    pub bias_detection: BiasDetectionResult,
    /// Overall fairness score
    pub overall_fairness_score: Float,
}

/// Demographic parity analysis result
#[derive(Debug, Clone)]
pub struct DemographicParityResult {
    /// Selection rates by group
    pub selection_rates: HashMap<String, Float>,
    /// Demographic parity difference
    pub parity_difference: Float,
    /// Demographic parity ratio
    pub parity_ratio: Float,
    /// Is demographic parity satisfied (within tolerance)
    pub is_fair: bool,
}

/// Equalized odds analysis result
#[derive(Debug, Clone)]
pub struct EqualizedOddsResult {
    /// True positive rates by group
    pub true_positive_rates: HashMap<String, Float>,
    /// False positive rates by group
    pub false_positive_rates: HashMap<String, Float>,
    /// Equalized odds difference
    pub odds_difference: Float,
    /// Is equalized odds satisfied (within tolerance)
    pub is_fair: bool,
}

/// Individual fairness analysis result
#[derive(Debug, Clone)]
pub struct IndividualFairnessResult {
    /// Lipschitz constant (measure of individual fairness)
    pub lipschitz_constant: Float,
    /// Consistency measure
    pub consistency: Float,
    /// Individual fairness violations
    pub violations: Vec<IndividualFairnessViolation>,
    /// Is individual fairness satisfied
    pub is_fair: bool,
}

/// Individual fairness violation
#[derive(Debug, Clone)]
pub struct IndividualFairnessViolation {
    /// Index of first instance
    pub instance1_idx: usize,
    /// Index of second instance
    pub instance2_idx: usize,
    /// Distance between instances
    pub distance: Float,
    /// Prediction difference
    pub prediction_difference: Float,
    /// Severity of violation
    pub severity: Float,
}

/// Bias detection result
#[derive(Debug, Clone)]
pub struct BiasDetectionResult {
    /// Statistical parity violations
    pub statistical_parity_violations: Vec<BiasViolation>,
    /// Conditional parity violations
    pub conditional_parity_violations: Vec<BiasViolation>,
    /// Predictive parity violations
    pub predictive_parity_violations: Vec<BiasViolation>,
    /// Overall bias score
    pub bias_score: Float,
}

/// Bias violation
#[derive(Debug, Clone)]
pub struct BiasViolation {
    /// Protected attribute involved
    pub protected_attribute: String,
    /// Groups compared
    pub groups: Vec<String>,
    /// Type of violation
    pub violation_type: BiasViolationType,
    /// Magnitude of violation
    pub magnitude: Float,
    /// Statistical significance (p-value)
    pub p_value: Option<Float>,
}

/// Type of bias violation
#[derive(Debug, Clone)]
pub enum BiasViolationType {
    /// Statistical parity violation
    StatisticalParity,
    /// Equalized odds violation
    EqualizedOdds,
    /// Predictive parity violation
    PredictiveParity,
    /// Individual fairness violation
    IndividualFairness,
}

/// Configuration for fairness assessment
#[derive(Debug, Clone)]
pub struct FairnessConfig {
    /// Protected attributes (column names or indices)
    pub protected_attributes: Vec<String>,
    /// Fairness tolerance (acceptable difference)
    pub fairness_tolerance: Float,
    /// Significance level for statistical tests
    pub significance_level: Float,
    /// Distance metric for individual fairness
    pub distance_metric: DistanceMetric,
    /// Number of samples for individual fairness assessment
    pub individual_fairness_samples: usize,
    /// Random seed for reproducibility
    pub random_state: Option<u64>,
}

/// Distance metric for individual fairness
#[derive(Debug, Clone, Copy)]
pub enum DistanceMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine distance
    Cosine,
    /// Custom distance (user-defined)
    Custom,
}

impl Default for FairnessConfig {
    fn default() -> Self {
        Self {
            protected_attributes: Vec::new(),
            fairness_tolerance: 0.1,
            significance_level: 0.05,
            distance_metric: DistanceMetric::Euclidean,
            individual_fairness_samples: 1000,
            random_state: None,
        }
    }
}

/// Assess model fairness comprehensively
///
/// Evaluates multiple fairness criteria including demographic parity, equalized odds,
/// and individual fairness.
pub fn assess_fairness<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    protected_attributes: &ArrayView2<Float>,
    config: &FairnessConfig,
) -> SklResult<FairnessResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    if X.nrows() != y_true.len()
        || X.nrows() != y_pred.len()
        || X.nrows() != protected_attributes.nrows()
    {
        return Err(SklearsError::InvalidInput(
            "All inputs must have same number of samples".to_string(),
        ));
    }

    // Analyze demographic parity
    let demographic_parity = analyze_demographic_parity(y_pred, protected_attributes, config)?;

    // Analyze equalized odds
    let equalized_odds = analyze_equalized_odds(y_true, y_pred, protected_attributes, config)?;

    // Analyze individual fairness
    let individual_fairness =
        analyze_individual_fairness(predict_fn, X, protected_attributes, config)?;

    // Detect bias
    let bias_detection = detect_bias(y_true, y_pred, protected_attributes, config)?;

    // Compute overall fairness score
    let overall_fairness_score = compute_overall_fairness_score(
        &demographic_parity,
        &equalized_odds,
        &individual_fairness,
        &bias_detection,
    );

    Ok(FairnessResult {
        demographic_parity,
        equalized_odds,
        individual_fairness,
        bias_detection,
        overall_fairness_score,
    })
}

/// Analyze demographic parity (statistical parity)
///
/// Ensures that positive predictions are equally likely across protected groups.
pub fn analyze_demographic_parity(
    y_pred: &ArrayView1<Float>,
    protected_attributes: &ArrayView2<Float>,
    config: &FairnessConfig,
) -> SklResult<DemographicParityResult> {
    if protected_attributes.ncols() == 0 {
        return Err(SklearsError::InvalidInput(
            "At least one protected attribute required".to_string(),
        ));
    }

    // Use first protected attribute for simplicity
    let protected_attr = protected_attributes.column(0);

    // Group instances by protected attribute values
    let groups = group_by_attribute(&protected_attr.view())?;
    let mut selection_rates = HashMap::new();

    // Compute selection rate for each group
    for (group_value, indices) in &groups {
        let group_predictions: Vec<Float> = indices.iter().map(|&i| y_pred[i]).collect();
        let positive_count = group_predictions.iter().filter(|&&pred| pred > 0.5).count();
        let selection_rate = positive_count as Float / group_predictions.len() as Float;
        selection_rates.insert(group_value.to_string(), selection_rate);
    }

    // Compute parity metrics
    let rates: Vec<Float> = selection_rates.values().cloned().collect();
    let max_rate = rates.iter().fold(0.0f64, |a, &b| a.max(b));
    let min_rate = rates.iter().fold(Float::INFINITY, |a, &b| a.min(b));

    let parity_difference = max_rate - min_rate;
    let parity_ratio = if max_rate > 0.0 {
        min_rate / max_rate
    } else {
        1.0
    };
    let is_fair = parity_difference <= config.fairness_tolerance;

    Ok(DemographicParityResult {
        selection_rates,
        parity_difference,
        parity_ratio,
        is_fair,
    })
}

/// Analyze equalized odds
///
/// Ensures that true positive and false positive rates are equal across protected groups.
pub fn analyze_equalized_odds(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    protected_attributes: &ArrayView2<Float>,
    config: &FairnessConfig,
) -> SklResult<EqualizedOddsResult> {
    if protected_attributes.ncols() == 0 {
        return Err(SklearsError::InvalidInput(
            "At least one protected attribute required".to_string(),
        ));
    }

    let protected_attr = protected_attributes.column(0);
    let groups = group_by_attribute(&protected_attr.view())?;

    let mut true_positive_rates = HashMap::new();
    let mut false_positive_rates = HashMap::new();

    // Compute TPR and FPR for each group
    for (group_value, indices) in &groups {
        let mut tp = 0;
        let mut fp = 0;
        let mut tn = 0;
        let mut fn_count = 0;

        for &i in indices {
            let true_label = if y_true[i] > 0.5 { 1 } else { 0 };
            let pred_label = if y_pred[i] > 0.5 { 1 } else { 0 };

            match (true_label, pred_label) {
                (1, 1) => tp += 1,
                (1, 0) => fn_count += 1,
                (0, 1) => fp += 1,
                (0, 0) => tn += 1,
                _ => {}
            }
        }

        let tpr = if tp + fn_count > 0 {
            tp as Float / (tp + fn_count) as Float
        } else {
            0.0
        };
        let fpr = if fp + tn > 0 {
            fp as Float / (fp + tn) as Float
        } else {
            0.0
        };

        true_positive_rates.insert(group_value.to_string(), tpr);
        false_positive_rates.insert(group_value.to_string(), fpr);
    }

    // Compute equalized odds difference
    let tpr_values: Vec<Float> = true_positive_rates.values().cloned().collect();
    let fpr_values: Vec<Float> = false_positive_rates.values().cloned().collect();

    let tpr_range = compute_range(&tpr_values);
    let fpr_range = compute_range(&fpr_values);
    let odds_difference = tpr_range.max(fpr_range);

    let is_fair = odds_difference <= config.fairness_tolerance;

    Ok(EqualizedOddsResult {
        true_positive_rates,
        false_positive_rates,
        odds_difference,
        is_fair,
    })
}

/// Analyze individual fairness
///
/// Ensures that similar individuals receive similar predictions.
pub fn analyze_individual_fairness<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    protected_attributes: &ArrayView2<Float>,
    config: &FairnessConfig,
) -> SklResult<IndividualFairnessResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    use scirs2_core::random::SeedableRng;
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let n_samples = X.nrows();
    let n_pairs = std::cmp::min(
        config.individual_fairness_samples,
        n_samples * (n_samples - 1) / 2,
    );

    let mut violations = Vec::new();
    let mut lipschitz_ratios = Vec::new();

    // Sample pairs of instances for individual fairness assessment
    for _ in 0..n_pairs {
        let i = rng.gen_range(0..n_samples);
        let j = rng.gen_range(0..n_samples);

        if i == j {
            continue;
        }

        let instance1 = X.row(i);
        let instance2 = X.row(j);

        // Compute distance between instances
        let distance =
            compute_distance(&instance1.view(), &instance2.view(), config.distance_metric);

        if distance < 1e-10 {
            continue; // Skip identical instances
        }

        // Get predictions
        let pred1_2d = instance1.insert_axis(Axis(0));
        let pred2_2d = instance2.insert_axis(Axis(0));
        let pred1 = predict_fn(&pred1_2d.view())[0];
        let pred2 = predict_fn(&pred2_2d.view())[0];

        let prediction_difference = (pred1 - pred2).abs();
        let lipschitz_ratio = prediction_difference / distance;
        lipschitz_ratios.push(lipschitz_ratio);

        // Check for violation (using fairness tolerance as threshold)
        if lipschitz_ratio > config.fairness_tolerance / distance {
            violations.push(IndividualFairnessViolation {
                instance1_idx: i,
                instance2_idx: j,
                distance,
                prediction_difference,
                severity: lipschitz_ratio,
            });
        }
    }

    // Compute Lipschitz constant (maximum Lipschitz ratio)
    let lipschitz_constant = lipschitz_ratios.iter().fold(0.0f64, |a, &b| a.max(b));

    // Compute consistency (1 - average Lipschitz ratio)
    let avg_lipschitz = if !lipschitz_ratios.is_empty() {
        lipschitz_ratios.iter().sum::<Float>() / lipschitz_ratios.len() as Float
    } else {
        0.0
    };
    let consistency = 1.0 / (1.0 + avg_lipschitz);

    let violation_rate = violations.len() as Float / n_pairs as Float;
    let is_fair = violation_rate < 0.1; // 10% violation threshold

    Ok(IndividualFairnessResult {
        lipschitz_constant,
        consistency,
        violations,
        is_fair,
    })
}

/// Detect bias in model predictions
///
/// Identifies various types of bias including statistical parity, conditional parity,
/// and predictive parity violations.
pub fn detect_bias(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    protected_attributes: &ArrayView2<Float>,
    config: &FairnessConfig,
) -> SklResult<BiasDetectionResult> {
    let mut statistical_parity_violations = Vec::new();
    let mut conditional_parity_violations = Vec::new();
    let mut predictive_parity_violations = Vec::new();

    // Check each protected attribute
    for attr_idx in 0..protected_attributes.ncols() {
        let protected_attr = protected_attributes.column(attr_idx);
        let attr_name = config
            .protected_attributes
            .get(attr_idx)
            .unwrap_or(&format!("attr_{}", attr_idx))
            .clone();

        // Statistical parity check
        if let Ok(violation) =
            check_statistical_parity(y_pred, &protected_attr.view(), &attr_name, config)
        {
            if violation.magnitude > config.fairness_tolerance {
                statistical_parity_violations.push(violation);
            }
        }

        // Conditional parity check (equalized odds)
        if let Ok(violations) =
            check_conditional_parity(y_true, y_pred, &protected_attr.view(), &attr_name, config)
        {
            conditional_parity_violations.extend(violations);
        }

        // Predictive parity check
        if let Ok(violation) =
            check_predictive_parity(y_true, y_pred, &protected_attr.view(), &attr_name, config)
        {
            if violation.magnitude > config.fairness_tolerance {
                predictive_parity_violations.push(violation);
            }
        }
    }

    // Compute overall bias score (0 = no bias, 1 = maximum bias)
    let total_violations = statistical_parity_violations.len()
        + conditional_parity_violations.len()
        + predictive_parity_violations.len();
    let max_possible_violations = protected_attributes.ncols() * 3;
    let bias_score = if max_possible_violations > 0 {
        (total_violations as Float / max_possible_violations as Float).min(1.0)
    } else {
        0.0
    };

    Ok(BiasDetectionResult {
        statistical_parity_violations,
        conditional_parity_violations,
        predictive_parity_violations,
        bias_score,
    })
}

/// Compute fairness metrics for specific groups
pub fn compute_group_fairness_metrics(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    group_indices: &[usize],
) -> SklResult<FairnessMetrics> {
    let mut tp = 0;
    let mut fp = 0;
    let mut tn = 0;
    let mut fn_count = 0;
    let mut positive_predictions = 0;

    for &i in group_indices {
        let true_label = if y_true[i] > 0.5 { 1 } else { 0 };
        let pred_label = if y_pred[i] > 0.5 { 1 } else { 0 };

        if pred_label == 1 {
            positive_predictions += 1;
        }

        match (true_label, pred_label) {
            (1, 1) => tp += 1,
            (1, 0) => fn_count += 1,
            (0, 1) => fp += 1,
            (0, 0) => tn += 1,
            _ => {}
        }
    }

    let total = group_indices.len();
    let selection_rate = positive_predictions as Float / total as Float;
    let accuracy = (tp + tn) as Float / total as Float;
    let precision = if tp + fp > 0 {
        tp as Float / (tp + fp) as Float
    } else {
        0.0
    };
    let recall = if tp + fn_count > 0 {
        tp as Float / (tp + fn_count) as Float
    } else {
        0.0
    };
    let specificity = if fp + tn > 0 {
        tn as Float / (fp + tn) as Float
    } else {
        0.0
    };

    Ok(FairnessMetrics {
        selection_rate,
        accuracy,
        precision,
        recall,
        specificity,
        true_positive_rate: recall,
        false_positive_rate: 1.0 - specificity,
    })
}

/// Fairness metrics for a group
#[derive(Debug, Clone)]
pub struct FairnessMetrics {
    /// Selection rate (positive prediction rate)
    pub selection_rate: Float,
    /// Accuracy
    pub accuracy: Float,
    /// Precision (positive predictive value)
    pub precision: Float,
    /// Recall (true positive rate)
    pub recall: Float,
    /// Specificity (true negative rate)
    pub specificity: Float,
    /// True positive rate
    pub true_positive_rate: Float,
    /// False positive rate
    pub false_positive_rate: Float,
}

// Helper functions

fn group_by_attribute(protected_attr: &ArrayView1<Float>) -> SklResult<HashMap<i32, Vec<usize>>> {
    let mut groups = HashMap::new();

    for (i, &value) in protected_attr.iter().enumerate() {
        let group_key = value.round() as i32; // Discretize continuous values
        groups.entry(group_key).or_insert_with(Vec::new).push(i);
    }

    Ok(groups)
}

fn compute_range(values: &[Float]) -> Float {
    if values.is_empty() {
        return 0.0;
    }

    let max_val = values.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
    let min_val = values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
    max_val - min_val
}

fn compute_distance(
    instance1: &ArrayView1<Float>,
    instance2: &ArrayView1<Float>,
    metric: DistanceMetric,
) -> Float {
    match metric {
        DistanceMetric::Euclidean => {
            let diff = instance1 - instance2;
            diff.mapv(|x| x.powi(2)).sum().sqrt()
        }
        DistanceMetric::Manhattan => {
            let diff = instance1 - instance2;
            diff.mapv(|x| x.abs()).sum()
        }
        DistanceMetric::Cosine => {
            let dot_product = instance1.dot(instance2);
            let norm1 = instance1.mapv(|x| x.powi(2)).sum().sqrt();
            let norm2 = instance2.mapv(|x| x.powi(2)).sum().sqrt();
            if norm1 > 0.0 && norm2 > 0.0 {
                1.0 - dot_product / (norm1 * norm2)
            } else {
                0.0
            }
        }
        DistanceMetric::Custom => {
            // Default to Euclidean for custom
            let diff = instance1 - instance2;
            diff.mapv(|x| x.powi(2)).sum().sqrt()
        }
    }
}

fn check_statistical_parity(
    y_pred: &ArrayView1<Float>,
    protected_attr: &ArrayView1<Float>,
    attr_name: &str,
    config: &FairnessConfig,
) -> SklResult<BiasViolation> {
    let groups = group_by_attribute(protected_attr)?;
    let mut selection_rates = Vec::new();
    let mut group_names = Vec::new();

    for (group_value, indices) in &groups {
        let positive_count = indices.iter().filter(|&&i| y_pred[i] > 0.5).count();
        let selection_rate = positive_count as Float / indices.len() as Float;
        selection_rates.push(selection_rate);
        group_names.push(group_value.to_string());
    }

    let magnitude = compute_range(&selection_rates);

    Ok(BiasViolation {
        protected_attribute: attr_name.to_string(),
        groups: group_names,
        violation_type: BiasViolationType::StatisticalParity,
        magnitude,
        p_value: None, // Would compute with statistical test
    })
}

fn check_conditional_parity(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    protected_attr: &ArrayView1<Float>,
    attr_name: &str,
    config: &FairnessConfig,
) -> SklResult<Vec<BiasViolation>> {
    let groups = group_by_attribute(protected_attr)?;
    let mut violations = Vec::new();
    let mut tpr_values = Vec::new();
    let mut fpr_values = Vec::new();
    let mut group_names = Vec::new();

    for (group_value, indices) in &groups {
        let metrics = compute_group_fairness_metrics(y_true, y_pred, indices)?;

        tpr_values.push(metrics.true_positive_rate);
        fpr_values.push(metrics.false_positive_rate);
        group_names.push(group_value.to_string());
    }

    let tpr_range = compute_range(&tpr_values);
    let fpr_range = compute_range(&fpr_values);

    if tpr_range > config.fairness_tolerance {
        violations.push(BiasViolation {
            protected_attribute: attr_name.to_string(),
            groups: group_names.clone(),
            violation_type: BiasViolationType::EqualizedOdds,
            magnitude: tpr_range,
            p_value: None,
        });
    }

    if fpr_range > config.fairness_tolerance {
        violations.push(BiasViolation {
            protected_attribute: attr_name.to_string(),
            groups: group_names,
            violation_type: BiasViolationType::EqualizedOdds,
            magnitude: fpr_range,
            p_value: None,
        });
    }

    Ok(violations)
}

fn check_predictive_parity(
    y_true: &ArrayView1<Float>,
    y_pred: &ArrayView1<Float>,
    protected_attr: &ArrayView1<Float>,
    attr_name: &str,
    config: &FairnessConfig,
) -> SklResult<BiasViolation> {
    let groups = group_by_attribute(protected_attr)?;
    let mut precision_values = Vec::new();
    let mut group_names = Vec::new();

    for (group_value, indices) in &groups {
        let metrics = compute_group_fairness_metrics(y_true, y_pred, indices)?;

        precision_values.push(metrics.precision);
        group_names.push(group_value.to_string());
    }

    let magnitude = compute_range(&precision_values);

    Ok(BiasViolation {
        protected_attribute: attr_name.to_string(),
        groups: group_names,
        violation_type: BiasViolationType::PredictiveParity,
        magnitude,
        p_value: None,
    })
}

fn compute_overall_fairness_score(
    demographic_parity: &DemographicParityResult,
    equalized_odds: &EqualizedOddsResult,
    individual_fairness: &IndividualFairnessResult,
    bias_detection: &BiasDetectionResult,
) -> Float {
    let mut score = 0.0;
    let mut components = 0;

    // Demographic parity component
    if demographic_parity.is_fair {
        score += 1.0;
    } else {
        score += (1.0 - demographic_parity.parity_difference).max(0.0);
    }
    components += 1;

    // Equalized odds component
    if equalized_odds.is_fair {
        score += 1.0;
    } else {
        score += (1.0 - equalized_odds.odds_difference).max(0.0);
    }
    components += 1;

    // Individual fairness component
    score += individual_fairness.consistency;
    components += 1;

    // Bias detection component
    score += (1.0 - bias_detection.bias_score).max(0.0);
    components += 1;

    score / components as Float
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_demographic_parity_analysis() {
        let y_pred = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let protected_attr = array![[0.0], [0.0], [1.0], [1.0], [0.0], [1.0]];

        let config = FairnessConfig::default();

        let result =
            analyze_demographic_parity(&y_pred.view(), &protected_attr.view(), &config).unwrap();

        assert!(result.selection_rates.len() >= 1);
        assert!(result.parity_difference >= 0.0);
        assert!(result.parity_ratio >= 0.0 && result.parity_ratio <= 1.0);
    }

    #[test]
    fn test_equalized_odds_analysis() {
        let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let protected_attr = array![[0.0], [0.0], [1.0], [1.0], [0.0], [1.0]];

        let config = FairnessConfig::default();

        let result = analyze_equalized_odds(
            &y_true.view(),
            &y_pred.view(),
            &protected_attr.view(),
            &config,
        )
        .unwrap();

        assert!(result.true_positive_rates.len() >= 1);
        assert!(result.false_positive_rates.len() >= 1);
        assert!(result.odds_difference >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_individual_fairness_analysis() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.sum()).collect()
        };

        let X = array![[1.0, 2.0], [1.1, 2.1], [3.0, 4.0]];
        let protected_attr = array![[0.0], [0.0], [1.0]];

        let mut config = FairnessConfig::default();
        config.individual_fairness_samples = 10; // Small for test

        let result =
            analyze_individual_fairness(&predict_fn, &X.view(), &protected_attr.view(), &config)
                .unwrap();

        assert!(result.lipschitz_constant >= 0.0);
        assert!(result.consistency >= 0.0 && result.consistency <= 1.0);
    }

    #[test]
    fn test_bias_detection() {
        let y_true = array![1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0, 0.0, 1.0]; // Some bias
        let protected_attr = array![[0.0], [0.0], [1.0], [1.0], [0.0], [1.0]];

        let config = FairnessConfig {
            protected_attributes: vec!["gender".to_string()],
            ..Default::default()
        };

        let result = detect_bias(
            &y_true.view(),
            &y_pred.view(),
            &protected_attr.view(),
            &config,
        )
        .unwrap();

        assert!(result.bias_score >= 0.0 && result.bias_score <= 1.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_fairness_assessment() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows()
                .into_iter()
                .map(|row| if row.sum() > 3.0 { 1.0 } else { 0.0 })
                .collect()
        };

        let X = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [1.0, 1.0]];
        let y_true = array![0.0, 1.0, 1.0, 0.0];
        let y_pred = array![0.0, 1.0, 1.0, 0.0];
        let protected_attr = array![[0.0], [0.0], [1.0], [1.0]];

        let config = FairnessConfig {
            protected_attributes: vec!["group".to_string()],
            individual_fairness_samples: 5, // Small for test
            ..Default::default()
        };

        let result = assess_fairness(
            &predict_fn,
            &X.view(),
            &y_true.view(),
            &y_pred.view(),
            &protected_attr.view(),
            &config,
        )
        .unwrap();

        assert!(result.overall_fairness_score >= 0.0 && result.overall_fairness_score <= 1.0);
        assert!(result.demographic_parity.parity_difference >= 0.0);
        assert!(result.equalized_odds.odds_difference >= 0.0);
    }

    #[test]
    fn test_group_fairness_metrics() {
        let y_true = array![1.0, 0.0, 1.0, 0.0];
        let y_pred = array![1.0, 0.0, 1.0, 0.0];
        let group_indices = vec![0, 1, 2, 3];

        let metrics =
            compute_group_fairness_metrics(&y_true.view(), &y_pred.view(), &group_indices).unwrap();

        assert_eq!(metrics.accuracy, 1.0); // Perfect predictions
        assert_eq!(metrics.selection_rate, 0.5); // 50% positive predictions
    }
}
