//! Fairness and Ethics Baseline Estimators
//!
//! This module provides fairness-aware baseline estimators for detecting
//! and mitigating bias in machine learning models.
//!
//! The module includes:
//! - [`FairnessAwareBaseline`] - Fairness-aware baseline with bias mitigation
//! - [`DemographicParityBaseline`] - Demographic parity constraint baseline
//! - [`EqualizedOddsBaseline`] - Equalized odds constraint baseline
//! - [`IndividualFairnessBaseline`] - Individual fairness baseline
//! - [`BiasDetectionBaseline`] - Bias detection and measurement baseline

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_core::random::{prelude::*, thread_rng, Rng};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use std::collections::HashMap;

/// Strategy for fairness-aware baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FairnessStrategy {
    /// Pre-processing bias mitigation
    PreProcessing { bias_reduction_factor: f64 },
    /// In-processing fairness constraints
    InProcessing { fairness_weight: f64 },
    /// Post-processing bias correction
    PostProcessing { correction_strength: f64 },
    /// Adversarial debiasing approximation
    AdversarialDebiasing { adversarial_weight: f64 },
    /// Fairness through awareness
    FairnessThruAwareness { awareness_threshold: f64 },
}

/// Strategy for demographic parity baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DemographicParityStrategy {
    /// Equal outcome rates across groups
    EqualOutcomeRates,
    /// Statistical parity constraint
    StatisticalParity { tolerance: f64 },
    /// Disparate impact mitigation
    DisparateImpactMitigation { impact_threshold: f64 },
    /// Group fairness optimization
    GroupFairnessOptimization { group_weights: Vec<f64> },
}

/// Strategy for equalized odds baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum EqualizedOddsStrategy {
    /// Equal TPR and FPR across groups
    EqualTPRFPR,
    /// Equal opportunity (TPR equality)
    EqualOpportunity,
    /// Predictive equality (FPR equality)
    PredictiveEquality,
    /// Conditional statistical parity
    ConditionalStatisticalParity { conditioning_variables: Vec<usize> },
}

/// Strategy for individual fairness baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum IndividualFairnessStrategy {
    /// Lipschitz fairness constraint
    LipschitzFairness { lipschitz_constant: f64 },
    /// Counterfactual fairness
    CounterfactualFairness { counterfactual_threshold: f64 },
    /// Similarity-based fairness
    SimilarityBasedFairness { similarity_metric: SimilarityMetric },
    /// Distance-based fairness
    DistanceBasedFairness { distance_threshold: f64 },
}

/// Similarity metric for individual fairness
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum SimilarityMetric {
    /// Euclidean distance
    Euclidean,
    /// Manhattan distance
    Manhattan,
    /// Cosine similarity
    Cosine,
    /// Hamming distance
    Hamming,
}

/// Strategy for bias detection baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BiasDetectionStrategy {
    /// Statistical bias detection
    StatisticalBias {
        statistical_tests: Vec<StatisticalTest>,
    },
    /// Disparate impact detection
    DisparateImpact { impact_thresholds: Vec<f64> },
    /// Algorithmic bias detection
    AlgorithmicBias { bias_metrics: Vec<BiasMetric> },
    /// Intersectional bias detection
    IntersectionalBias {
        intersection_groups: Vec<Vec<usize>>,
    },
}

/// Statistical test for bias detection
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum StatisticalTest {
    /// Chi-square test for independence
    ChiSquare,
    /// Fisher's exact test
    FisherExact,
    /// Kolmogorov-Smirnov test
    KolmogorovSmirnov,
    /// Mann-Whitney U test
    MannWhitney,
}

/// Bias metric for algorithmic fairness
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum BiasMetric {
    /// Demographic parity difference
    DemographicParityDifference,
    /// Equalized odds difference
    EqualizedOddsDifference,
    /// Calibration difference
    CalibrationDifference,
    /// Individual fairness violation
    IndividualFairnessViolation,
}

/// Fairness-aware baseline estimator
#[derive(Debug, Clone)]
pub struct FairnessAwareBaseline {
    strategy: FairnessStrategy,
    protected_attribute_columns: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted fairness-aware baseline
#[derive(Debug, Clone)]
pub struct FittedFairnessAwareBaseline {
    strategy: FairnessStrategy,
    protected_attribute_columns: Vec<usize>,
    group_statistics: HashMap<String, GroupStatistics>,
    bias_mitigation_params: Array1<f64>,
    fairness_constraints: Vec<FairnessConstraint>,
    random_state: Option<u64>,
}

/// Demographic parity baseline estimator
#[derive(Debug, Clone)]
pub struct DemographicParityBaseline {
    strategy: DemographicParityStrategy,
    protected_attribute_columns: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted demographic parity baseline
#[derive(Debug, Clone)]
pub struct FittedDemographicParityBaseline {
    strategy: DemographicParityStrategy,
    protected_attribute_columns: Vec<usize>,
    group_outcome_rates: HashMap<String, f64>,
    parity_adjustments: HashMap<String, f64>,
    random_state: Option<u64>,
}

/// Equalized odds baseline estimator
#[derive(Debug, Clone)]
pub struct EqualizedOddsBaseline {
    strategy: EqualizedOddsStrategy,
    protected_attribute_columns: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted equalized odds baseline
#[derive(Debug, Clone)]
pub struct FittedEqualizedOddsBaseline {
    strategy: EqualizedOddsStrategy,
    protected_attribute_columns: Vec<usize>,
    group_tpr_fpr: HashMap<String, (f64, f64)>,
    odds_adjustments: HashMap<String, (f64, f64)>,
    random_state: Option<u64>,
}

/// Individual fairness baseline estimator
#[derive(Debug, Clone)]
pub struct IndividualFairnessBaseline {
    strategy: IndividualFairnessStrategy,
    sensitive_features: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted individual fairness baseline
#[derive(Debug, Clone)]
pub struct FittedIndividualFairnessBaseline {
    strategy: IndividualFairnessStrategy,
    sensitive_features: Vec<usize>,
    similarity_matrix: Array2<f64>,
    fairness_violations: Array1<f64>,
    random_state: Option<u64>,
}

/// Bias detection baseline estimator
#[derive(Debug, Clone)]
pub struct BiasDetectionBaseline {
    strategy: BiasDetectionStrategy,
    protected_attribute_columns: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted bias detection baseline
#[derive(Debug, Clone)]
pub struct FittedBiasDetectionBaseline {
    strategy: BiasDetectionStrategy,
    protected_attribute_columns: Vec<usize>,
    bias_metrics: HashMap<String, BiasMetricResult>,
    statistical_test_results: HashMap<String, StatisticalTestResult>,
    bias_detected: bool,
    random_state: Option<u64>,
}

/// Group statistics for fairness analysis
#[derive(Debug, Clone)]
pub struct GroupStatistics {
    /// group_size
    pub group_size: usize,
    /// outcome_rate
    pub outcome_rate: f64,
    /// feature_means
    pub feature_means: Array1<f64>,
    /// feature_stds
    pub feature_stds: Array1<f64>,
}

/// Fairness constraint for optimization
#[derive(Debug, Clone)]
pub struct FairnessConstraint {
    /// constraint_type
    pub constraint_type: String,
    /// target_value
    pub target_value: f64,
    /// tolerance
    pub tolerance: f64,
    /// groups
    pub groups: Vec<String>,
}

/// Result of bias metric computation
#[derive(Debug, Clone)]
pub struct BiasMetricResult {
    /// metric_name
    pub metric_name: String,
    /// value
    pub value: f64,
    /// threshold
    pub threshold: f64,
    /// bias_detected
    pub bias_detected: bool,
    /// affected_groups
    pub affected_groups: Vec<String>,
}

/// Result of statistical test for bias
#[derive(Debug, Clone)]
pub struct StatisticalTestResult {
    /// test_name
    pub test_name: String,
    /// statistic
    pub statistic: f64,
    /// p_value
    pub p_value: f64,
    /// significant
    pub significant: bool,
    /// effect_size
    pub effect_size: f64,
}

impl FairnessAwareBaseline {
    /// Create a new fairness-aware baseline
    pub fn new(strategy: FairnessStrategy, protected_attribute_columns: Vec<usize>) -> Self {
        Self {
            strategy,
            protected_attribute_columns,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>, FittedFairnessAwareBaseline> for FairnessAwareBaseline {
    type Fitted = FittedFairnessAwareBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<FittedFairnessAwareBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Validate protected attribute columns
        for &col in &self.protected_attribute_columns {
            if col >= x.ncols() {
                return Err(SklearsError::InvalidParameter {
                    name: "protected_attribute_columns".to_string(),
                    reason: format!("Column index {} exceeds data dimensions", col),
                });
            }
        }

        // Compute group statistics
        let mut group_statistics = HashMap::new();
        let groups = self.identify_groups(x);

        for (group_id, group_indices) in groups {
            let group_size = group_indices.len();
            let group_outcomes: Vec<i32> = group_indices.iter().map(|&i| y[i]).collect();
            let outcome_rate = group_outcomes
                .iter()
                .map(|&outcome| if outcome > 0 { 1.0 } else { 0.0 })
                .sum::<f64>()
                / group_size as f64;

            let group_features: Vec<f64> = group_indices
                .iter()
                .flat_map(|&i| x.row(i).to_vec())
                .collect();
            let group_feature_matrix =
                Array2::from_shape_vec((group_size, x.ncols()), group_features)?;

            let feature_means = group_feature_matrix.mean_axis(Axis(0)).unwrap();
            let feature_stds = group_feature_matrix.std_axis(Axis(0), 0.0);

            group_statistics.insert(
                group_id,
                GroupStatistics {
                    group_size,
                    outcome_rate,
                    feature_means,
                    feature_stds,
                },
            );
        }

        // Compute bias mitigation parameters
        let bias_mitigation_params = self.compute_bias_mitigation_params(&group_statistics);

        // Define fairness constraints
        let fairness_constraints = self.define_fairness_constraints(&group_statistics);

        Ok(FittedFairnessAwareBaseline {
            strategy: self.strategy,
            protected_attribute_columns: self.protected_attribute_columns,
            group_statistics,
            bias_mitigation_params,
            fairness_constraints,
            random_state: self.random_state,
        })
    }
}

impl FairnessAwareBaseline {
    fn identify_groups(&self, x: &Array2<f64>) -> HashMap<String, Vec<usize>> {
        let mut groups = HashMap::new();

        for i in 0..x.nrows() {
            let mut group_key = String::new();
            for &col in &self.protected_attribute_columns {
                let value = x[[i, col]];
                let group_value = if value > 0.5 { "1" } else { "0" };
                group_key.push_str(&format!("{}:{};", col, group_value));
            }

            groups.entry(group_key).or_insert_with(Vec::new).push(i);
        }

        groups
    }

    fn compute_bias_mitigation_params(
        &self,
        group_stats: &HashMap<String, GroupStatistics>,
    ) -> Array1<f64> {
        match &self.strategy {
            FairnessStrategy::PreProcessing {
                bias_reduction_factor,
            } => Array1::from_elem(group_stats.len(), *bias_reduction_factor),
            FairnessStrategy::InProcessing { fairness_weight } => {
                Array1::from_elem(group_stats.len(), *fairness_weight)
            }
            FairnessStrategy::PostProcessing {
                correction_strength,
            } => Array1::from_elem(group_stats.len(), *correction_strength),
            FairnessStrategy::AdversarialDebiasing { adversarial_weight } => {
                Array1::from_elem(group_stats.len(), *adversarial_weight)
            }
            FairnessStrategy::FairnessThruAwareness {
                awareness_threshold,
            } => Array1::from_elem(group_stats.len(), *awareness_threshold),
        }
    }

    fn define_fairness_constraints(
        &self,
        group_stats: &HashMap<String, GroupStatistics>,
    ) -> Vec<FairnessConstraint> {
        let mut constraints = Vec::new();

        match &self.strategy {
            FairnessStrategy::PreProcessing { .. } => {
                constraints.push(FairnessConstraint {
                    constraint_type: "demographic_parity".to_string(),
                    target_value: 0.5,
                    tolerance: 0.1,
                    groups: group_stats.keys().cloned().collect(),
                });
            }
            FairnessStrategy::InProcessing { fairness_weight } => {
                constraints.push(FairnessConstraint {
                    constraint_type: "equalized_odds".to_string(),
                    target_value: *fairness_weight,
                    tolerance: 0.05,
                    groups: group_stats.keys().cloned().collect(),
                });
            }
            FairnessStrategy::PostProcessing { .. } => {
                constraints.push(FairnessConstraint {
                    constraint_type: "calibration".to_string(),
                    target_value: 0.0,
                    tolerance: 0.02,
                    groups: group_stats.keys().cloned().collect(),
                });
            }
            _ => {}
        }

        constraints
    }
}

impl Predict<Array2<f64>, Array1<i32>> for FittedFairnessAwareBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());
        let mut rng = self.random_state.map_or_else(
            || Box::new(thread_rng()) as Box<dyn RngCore>,
            |seed| Box::new(StdRng::seed_from_u64(seed)),
        );

        for (i, sample) in x.rows().into_iter().enumerate() {
            // Identify the group for this sample
            let mut group_key = String::new();
            for &col in &self.protected_attribute_columns {
                let value = sample[col];
                let group_value = if value > 0.5 { "1" } else { "0" };
                group_key.push_str(&format!("{}:{};", col, group_value));
            }

            let prediction = if let Some(group_stats) = self.group_statistics.get(&group_key) {
                match &self.strategy {
                    FairnessStrategy::PreProcessing { .. } => {
                        // Apply pre-processing bias mitigation
                        let bias_adjusted_rate = group_stats.outcome_rate
                            * self.bias_mitigation_params[i % self.bias_mitigation_params.len()];
                        if rng.gen::<f64>() < bias_adjusted_rate {
                            1
                        } else {
                            0
                        }
                    }
                    FairnessStrategy::InProcessing { .. } => {
                        // Apply in-processing fairness constraints
                        let fairness_adjusted_rate = (group_stats.outcome_rate + 0.5) / 2.0;
                        if rng.gen::<f64>() < fairness_adjusted_rate {
                            1
                        } else {
                            0
                        }
                    }
                    FairnessStrategy::PostProcessing {
                        correction_strength,
                    } => {
                        // Apply post-processing bias correction
                        let base_prediction = if rng.gen::<f64>() < group_stats.outcome_rate {
                            1
                        } else {
                            0
                        };
                        let correction = (0.5 - group_stats.outcome_rate) * correction_strength;
                        let corrected_prob = group_stats.outcome_rate + correction;
                        if rng.gen::<f64>() < corrected_prob {
                            1
                        } else {
                            base_prediction
                        }
                    }
                    FairnessStrategy::AdversarialDebiasing { adversarial_weight } => {
                        // Apply adversarial debiasing
                        let adversarial_adjustment =
                            adversarial_weight * (0.5 - group_stats.outcome_rate);
                        let adjusted_rate = (group_stats.outcome_rate + adversarial_adjustment)
                            .max(0.0)
                            .min(1.0);
                        if rng.gen::<f64>() < adjusted_rate {
                            1
                        } else {
                            0
                        }
                    }
                    FairnessStrategy::FairnessThruAwareness {
                        awareness_threshold,
                    } => {
                        // Apply fairness through awareness
                        if group_stats.outcome_rate.abs() < *awareness_threshold {
                            if rng.gen::<f64>() < 0.5 {
                                1
                            } else {
                                0
                            } // Random fair prediction
                        } else if rng.gen::<f64>() < group_stats.outcome_rate {
                            1
                        } else {
                            0
                        }
                    }
                }
            } else {
                // Default prediction for unknown groups
                if rng.gen::<f64>() < 0.5 {
                    1
                } else {
                    0
                }
            };

            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

impl DemographicParityBaseline {
    /// Create a new demographic parity baseline
    pub fn new(
        strategy: DemographicParityStrategy,
        protected_attribute_columns: Vec<usize>,
    ) -> Self {
        Self {
            strategy,
            protected_attribute_columns,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<i32>, FittedDemographicParityBaseline> for DemographicParityBaseline {
    type Fitted = FittedDemographicParityBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<i32>,
    ) -> Result<FittedDemographicParityBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        // Identify groups and compute outcome rates
        let groups = self.identify_groups(x);
        let mut group_outcome_rates = HashMap::new();
        let mut parity_adjustments = HashMap::new();

        // Compute overall outcome rate
        let overall_outcome_rate = y
            .iter()
            .map(|&outcome| if outcome > 0 { 1.0 } else { 0.0 })
            .sum::<f64>()
            / y.len() as f64;

        for (group_id, group_indices) in groups {
            let group_outcomes: Vec<i32> = group_indices.iter().map(|&i| y[i]).collect();
            let group_outcome_rate = group_outcomes
                .iter()
                .map(|&outcome| if outcome > 0 { 1.0 } else { 0.0 })
                .sum::<f64>()
                / group_outcomes.len() as f64;

            let adjustment = match &self.strategy {
                DemographicParityStrategy::EqualOutcomeRates => {
                    overall_outcome_rate - group_outcome_rate
                }
                DemographicParityStrategy::StatisticalParity { tolerance } => {
                    let difference = (group_outcome_rate - overall_outcome_rate).abs();
                    if difference > *tolerance {
                        overall_outcome_rate - group_outcome_rate
                    } else {
                        0.0
                    }
                }
                DemographicParityStrategy::DisparateImpactMitigation { impact_threshold } => {
                    let impact_ratio = if overall_outcome_rate > 0.0 {
                        group_outcome_rate / overall_outcome_rate
                    } else {
                        1.0
                    };
                    if impact_ratio < *impact_threshold {
                        overall_outcome_rate - group_outcome_rate
                    } else {
                        0.0
                    }
                }
                DemographicParityStrategy::GroupFairnessOptimization { group_weights } => {
                    let weight = if group_weights.is_empty() {
                        1.0
                    } else {
                        group_weights[0] // Simplified: use first weight
                    };
                    weight * (overall_outcome_rate - group_outcome_rate)
                }
            };

            group_outcome_rates.insert(group_id.clone(), group_outcome_rate);
            parity_adjustments.insert(group_id, adjustment);
        }

        Ok(FittedDemographicParityBaseline {
            strategy: self.strategy,
            protected_attribute_columns: self.protected_attribute_columns,
            group_outcome_rates,
            parity_adjustments,
            random_state: self.random_state,
        })
    }
}

impl DemographicParityBaseline {
    fn identify_groups(&self, x: &Array2<f64>) -> HashMap<String, Vec<usize>> {
        let mut groups = HashMap::new();

        for i in 0..x.nrows() {
            let mut group_key = String::new();
            for &col in &self.protected_attribute_columns {
                let value = x[[i, col]];
                let group_value = if value > 0.5 { "1" } else { "0" };
                group_key.push_str(&format!("{}:{};", col, group_value));
            }

            groups.entry(group_key).or_insert_with(Vec::new).push(i);
        }

        groups
    }
}

impl Predict<Array2<f64>, Array1<i32>> for FittedDemographicParityBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<i32>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());
        let mut rng = self.random_state.map_or_else(
            || Box::new(thread_rng()) as Box<dyn RngCore>,
            |seed| Box::new(StdRng::seed_from_u64(seed)),
        );

        for sample in x.rows() {
            // Identify the group for this sample
            let mut group_key = String::new();
            for &col in &self.protected_attribute_columns {
                let value = sample[col];
                let group_value = if value > 0.5 { "1" } else { "0" };
                group_key.push_str(&format!("{}:{};", col, group_value));
            }

            let prediction = if let (Some(&group_rate), Some(&adjustment)) = (
                self.group_outcome_rates.get(&group_key),
                self.parity_adjustments.get(&group_key),
            ) {
                let adjusted_rate = (group_rate + adjustment).max(0.0).min(1.0);
                if rng.gen::<f64>() < adjusted_rate {
                    1
                } else {
                    0
                }
            } else {
                // Default prediction for unknown groups
                if rng.gen::<f64>() < 0.5 {
                    1
                } else {
                    0
                }
            };

            predictions.push(prediction);
        }

        Ok(Array1::from_vec(predictions))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_fairness_aware_baseline() {
        let x = Array2::from_shape_vec(
            (6, 3),
            vec![
                1.0, 0.0, 2.0, 2.0, 0.0, 3.0, 3.0, 1.0, 4.0, 4.0, 1.0, 5.0, 5.0, 0.0, 6.0, 6.0,
                1.0, 7.0,
            ],
        )
        .unwrap();
        let y = array![0, 0, 1, 1, 0, 1];

        let baseline = FairnessAwareBaseline::new(
            FairnessStrategy::PreProcessing {
                bias_reduction_factor: 0.8,
            },
            vec![1], // Column 1 is the protected attribute
        );
        let fitted = baseline.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 6);
        assert!(fitted.group_statistics.len() > 0);
    }

    #[test]
    fn test_demographic_parity_baseline() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 3.0, 0.0, 4.0, 4.0, 1.0, 5.0],
        )
        .unwrap();
        let y = array![0, 1, 0, 1];

        let baseline = DemographicParityBaseline::new(
            DemographicParityStrategy::EqualOutcomeRates,
            vec![1], // Column 1 is the protected attribute
        );
        let fitted = baseline.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(fitted.group_outcome_rates.len() > 0);
    }

    #[test]
    fn test_fairness_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let strategies = vec![
            FairnessStrategy::PreProcessing {
                bias_reduction_factor: 0.8,
            },
            FairnessStrategy::InProcessing {
                fairness_weight: 0.5,
            },
            FairnessStrategy::PostProcessing {
                correction_strength: 0.3,
            },
            FairnessStrategy::AdversarialDebiasing {
                adversarial_weight: 0.4,
            },
            FairnessStrategy::FairnessThruAwareness {
                awareness_threshold: 0.1,
            },
        ];

        for strategy in strategies {
            let baseline = FairnessAwareBaseline::new(strategy, vec![1]).with_random_state(42);
            let fitted = baseline.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }

    #[test]
    fn test_demographic_parity_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 0.0, 2.0, 1.0, 3.0, 0.0, 4.0, 1.0]).unwrap();
        let y = array![0, 1, 0, 1];

        let strategies = vec![
            DemographicParityStrategy::EqualOutcomeRates,
            DemographicParityStrategy::StatisticalParity { tolerance: 0.1 },
            DemographicParityStrategy::DisparateImpactMitigation {
                impact_threshold: 0.8,
            },
            DemographicParityStrategy::GroupFairnessOptimization {
                group_weights: vec![1.0],
            },
        ];

        for strategy in strategies {
            let baseline = DemographicParityBaseline::new(strategy, vec![1]).with_random_state(42);
            let fitted = baseline.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }
}
