//! Causal Inference Baseline Estimators
//!
//! This module provides causal inference baseline estimators for establishing
//! causal relationships, counterfactual reasoning, and confounding control.
//!
//! The module includes:
//! - [`CausalDiscoveryBaseline`] - Causal discovery baseline using correlation and conditional independence
//! - [`CounterfactualBaseline`] - Counterfactual baseline using outcome modeling
//! - [`InstrumentalVariableBaseline`] - Instrumental variable baseline for confounding control
//! - [`MediationAnalysisBaseline`] - Mediation analysis baseline for indirect effects
//! - [`DoCalculusBaseline`] - Do-calculus baseline for interventional reasoning

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::error::SklearsError;
use sklears_core::traits::{Fit, Predict};
use std::collections::HashMap;

/// Strategy for causal discovery baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CausalDiscoveryStrategy {
    /// Correlation-based causal discovery
    Correlation { threshold: f64 },
    /// Conditional independence-based discovery
    ConditionalIndependence { significance_level: f64 },
    /// PC algorithm approximation
    PCAlgorithm { alpha: f64 },
    /// Granger causality test
    GrangerCausality { max_lags: usize },
    /// Constraint-based causal discovery
    ConstraintBased { edge_threshold: f64 },
}

/// Strategy for counterfactual baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum CounterfactualStrategy {
    /// Outcome modeling for counterfactuals
    OutcomeModeling { treatment_effect: f64 },
    /// Propensity score matching
    PropensityScoreMatching { matching_tolerance: f64 },
    /// Doubly robust estimation
    DoublyRobust { propensity_weight: f64 },
    /// Inverse probability weighting
    InverseProbabilityWeighting { weight_truncation: f64 },
    /// Targeted maximum likelihood estimation
    TargetedMLE { targeting_steps: usize },
}

/// Strategy for instrumental variable baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum InstrumentalVariableStrategy {
    /// Two-stage least squares
    TwoStageLeastSquares { first_stage_strength: f64 },
    /// Limited information maximum likelihood
    LimitedInformationML { instruments_count: usize },
    /// Generalized method of moments
    GeneralizedMethodOfMoments { moment_conditions: usize },
    /// Control function approach
    ControlFunction { control_weight: f64 },
}

/// Strategy for mediation analysis baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum MediationStrategy {
    /// Baron-Kenny mediation analysis
    BaronKenny { indirect_effect_threshold: f64 },
    /// Sobel test for mediation
    SobelTest { mediation_strength: f64 },
    /// Bootstrapped mediation analysis
    Bootstrapped { bootstrap_samples: usize },
    /// Causal mediation analysis
    CausalMediation { confounding_adjustment: f64 },
}

/// Strategy for do-calculus baselines
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum DoCalculusStrategy {
    /// Interventional distribution estimation
    InterventionalDistribution { intervention_strength: f64 },
    /// Backdoor adjustment
    BackdoorAdjustment { confounder_sets: Vec<usize> },
    /// Frontdoor adjustment
    FrontdoorAdjustment { mediator_sets: Vec<usize> },
    /// Instrumental variable adjustment
    InstrumentalAdjustment { instrument_sets: Vec<usize> },
}

/// Causal discovery baseline estimator
#[derive(Debug, Clone)]
pub struct CausalDiscoveryBaseline {
    strategy: CausalDiscoveryStrategy,
    random_state: Option<u64>,
}

/// Fitted causal discovery baseline
#[derive(Debug, Clone)]
pub struct FittedCausalDiscoveryBaseline {
    strategy: CausalDiscoveryStrategy,
    causal_graph: Array2<f64>,
    feature_names: Vec<String>,
    causal_strengths: HashMap<(usize, usize), f64>,
    random_state: Option<u64>,
}

/// Counterfactual baseline estimator
#[derive(Debug, Clone)]
pub struct CounterfactualBaseline {
    strategy: CounterfactualStrategy,
    treatment_column: usize,
    outcome_column: usize,
    random_state: Option<u64>,
}

/// Fitted counterfactual baseline
#[derive(Debug, Clone)]
pub struct FittedCounterfactualBaseline {
    strategy: CounterfactualStrategy,
    treatment_column: usize,
    outcome_column: usize,
    treatment_effect: f64,
    propensity_scores: Array1<f64>,
    outcome_model_params: Array1<f64>,
    random_state: Option<u64>,
}

/// Instrumental variable baseline estimator
#[derive(Debug, Clone)]
pub struct InstrumentalVariableBaseline {
    strategy: InstrumentalVariableStrategy,
    instrument_columns: Vec<usize>,
    treatment_column: usize,
    random_state: Option<u64>,
}

/// Fitted instrumental variable baseline
#[derive(Debug, Clone)]
pub struct FittedInstrumentalVariableBaseline {
    strategy: InstrumentalVariableStrategy,
    instrument_columns: Vec<usize>,
    treatment_column: usize,
    first_stage_coefficients: Array1<f64>,
    second_stage_coefficients: Array1<f64>,
    causal_effect: f64,
    random_state: Option<u64>,
}

/// Mediation analysis baseline estimator
#[derive(Debug, Clone)]
pub struct MediationAnalysisBaseline {
    strategy: MediationStrategy,
    treatment_column: usize,
    mediator_column: usize,
    outcome_column: usize,
    random_state: Option<u64>,
}

/// Fitted mediation analysis baseline
#[derive(Debug, Clone)]
pub struct FittedMediationAnalysisBaseline {
    strategy: MediationStrategy,
    treatment_column: usize,
    mediator_column: usize,
    outcome_column: usize,
    direct_effect: f64,
    indirect_effect: f64,
    total_effect: f64,
    mediation_proportion: f64,
    random_state: Option<u64>,
}

/// Do-calculus baseline estimator
#[derive(Debug, Clone)]
pub struct DoCalculusBaseline {
    strategy: DoCalculusStrategy,
    intervention_variables: Vec<usize>,
    outcome_variables: Vec<usize>,
    random_state: Option<u64>,
}

/// Fitted do-calculus baseline
#[derive(Debug, Clone)]
pub struct FittedDoCalculusBaseline {
    strategy: DoCalculusStrategy,
    intervention_variables: Vec<usize>,
    outcome_variables: Vec<usize>,
    interventional_distribution: Array2<f64>,
    causal_effects: HashMap<usize, f64>,
    random_state: Option<u64>,
}

impl CausalDiscoveryBaseline {
    /// Create a new causal discovery baseline
    pub fn new(strategy: CausalDiscoveryStrategy) -> Self {
        Self {
            strategy,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedCausalDiscoveryBaseline> for CausalDiscoveryBaseline {
    type Fitted = FittedCausalDiscoveryBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        _y: &Array1<f64>,
    ) -> Result<FittedCausalDiscoveryBaseline, SklearsError> {
        let n_features = x.ncols();
        let mut causal_graph = Array2::zeros((n_features, n_features));
        let mut causal_strengths = HashMap::new();

        // Generate feature names
        let feature_names: Vec<String> =
            (0..n_features).map(|i| format!("feature_{}", i)).collect();

        match &self.strategy {
            CausalDiscoveryStrategy::Correlation { threshold } => {
                // Compute correlation matrix and threshold
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let correlation = self.compute_correlation(&col_i, &col_j);

                            if correlation.abs() > *threshold {
                                causal_graph[[i, j]] = correlation;
                                causal_strengths.insert((i, j), correlation);
                            }
                        }
                    }
                }
            }
            CausalDiscoveryStrategy::ConditionalIndependence { significance_level } => {
                // Simplified conditional independence test
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let independence_stat =
                                self.compute_conditional_independence(&col_i, &col_j);

                            if independence_stat > *significance_level {
                                causal_graph[[i, j]] = independence_stat;
                                causal_strengths.insert((i, j), independence_stat);
                            }
                        }
                    }
                }
            }
            CausalDiscoveryStrategy::PCAlgorithm { alpha } => {
                // Simplified PC algorithm
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let pc_stat = self.compute_pc_statistic(&col_i, &col_j, *alpha);

                            if pc_stat > 0.0 {
                                causal_graph[[i, j]] = pc_stat;
                                causal_strengths.insert((i, j), pc_stat);
                            }
                        }
                    }
                }
            }
            CausalDiscoveryStrategy::GrangerCausality { max_lags } => {
                // Simplified Granger causality test
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let granger_stat =
                                self.compute_granger_causality(&col_i, &col_j, *max_lags);

                            if granger_stat > 0.0 {
                                causal_graph[[i, j]] = granger_stat;
                                causal_strengths.insert((i, j), granger_stat);
                            }
                        }
                    }
                }
            }
            CausalDiscoveryStrategy::ConstraintBased { edge_threshold } => {
                // Constraint-based causal discovery
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            let col_i = x.column(i);
                            let col_j = x.column(j);
                            let constraint_stat = self.compute_constraint_statistic(&col_i, &col_j);

                            if constraint_stat > *edge_threshold {
                                causal_graph[[i, j]] = constraint_stat;
                                causal_strengths.insert((i, j), constraint_stat);
                            }
                        }
                    }
                }
            }
        }

        Ok(FittedCausalDiscoveryBaseline {
            strategy: self.strategy,
            causal_graph,
            feature_names,
            causal_strengths,
            random_state: self.random_state,
        })
    }
}

impl CausalDiscoveryBaseline {
    fn compute_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        if var_x == 0.0 || var_y == 0.0 {
            0.0
        } else {
            numerator / (var_x * var_y).sqrt()
        }
    }

    fn compute_conditional_independence(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        // Simplified conditional independence test using partial correlation
        let correlation = self.compute_correlation(x, y);
        let independence_stat = correlation.abs();

        // Transform to p-value approximation
        let n = x.len() as f64;
        let t_stat = independence_stat * ((n - 2.0) / (1.0 - independence_stat.powi(2))).sqrt();

        // Approximate p-value using t-distribution
        let p_value = 2.0 * (1.0 - (t_stat / (1.0 + t_stat.powi(2) / (n - 2.0)).sqrt()));

        1.0 - p_value
    }

    fn compute_pc_statistic(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>, alpha: f64) -> f64 {
        // Simplified PC algorithm statistic
        let correlation = self.compute_correlation(x, y);
        let independence_stat = self.compute_conditional_independence(x, y);

        if independence_stat > alpha {
            correlation.abs()
        } else {
            0.0
        }
    }

    fn compute_granger_causality(
        &self,
        x: &ArrayView1<f64>,
        y: &ArrayView1<f64>,
        max_lags: usize,
    ) -> f64 {
        // Simplified Granger causality test
        let n = x.len();
        if n <= max_lags {
            return 0.0;
        }

        let mut granger_stat = 0.0;

        // Create lagged versions and compute predictive power
        for lag in 1..=max_lags {
            if n > lag {
                let x_lagged = x.slice(s![..n - lag]);
                let y_current = y.slice(s![lag..]);
                let correlation = self.compute_correlation(&x_lagged, &y_current);
                granger_stat += correlation.abs();
            }
        }

        granger_stat / max_lags as f64
    }

    fn compute_constraint_statistic(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        // Simplified constraint-based statistic
        let correlation = self.compute_correlation(x, y);
        let independence_stat = self.compute_conditional_independence(x, y);

        (correlation.abs() + independence_stat) / 2.0
    }
}

impl CounterfactualBaseline {
    /// Create a new counterfactual baseline
    pub fn new(
        strategy: CounterfactualStrategy,
        treatment_column: usize,
        outcome_column: usize,
    ) -> Self {
        Self {
            strategy,
            treatment_column,
            outcome_column,
            random_state: None,
        }
    }

    /// Set the random state for reproducible results
    pub fn with_random_state(mut self, seed: u64) -> Self {
        self.random_state = Some(seed);
        self
    }
}

impl Fit<Array2<f64>, Array1<f64>, FittedCounterfactualBaseline> for CounterfactualBaseline {
    type Fitted = FittedCounterfactualBaseline;

    fn fit(
        self,
        x: &Array2<f64>,
        y: &Array1<f64>,
    ) -> Result<FittedCounterfactualBaseline, SklearsError> {
        if x.nrows() != y.len() {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("{} samples", x.nrows()),
                actual: format!("{} labels", y.len()),
            });
        }

        if self.treatment_column >= x.ncols() || self.outcome_column >= x.ncols() {
            return Err(SklearsError::InvalidParameter {
                name: "column_index".to_string(),
                reason: "Treatment or outcome column index exceeds data dimensions".to_string(),
            });
        }

        let treatment = x.column(self.treatment_column);
        let outcome = if self.outcome_column < x.ncols() {
            x.column(self.outcome_column)
        } else {
            y.view()
        };

        let mut treatment_effect = 0.0;
        let mut propensity_scores = Array1::zeros(x.nrows());
        let mut outcome_model_params = Array1::zeros(x.ncols());

        match &self.strategy {
            CounterfactualStrategy::OutcomeModeling {
                treatment_effect: te,
            } => {
                treatment_effect = *te;

                // Simple outcome modeling: compare treated vs untreated
                let treated_indices: Vec<usize> = treatment
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t > 0.5)
                    .map(|(i, _)| i)
                    .collect();

                let untreated_indices: Vec<usize> = treatment
                    .iter()
                    .enumerate()
                    .filter(|(_, &t)| t <= 0.5)
                    .map(|(i, _)| i)
                    .collect();

                if !treated_indices.is_empty() && !untreated_indices.is_empty() {
                    let treated_mean: f64 =
                        treated_indices.iter().map(|&i| outcome[i]).sum::<f64>()
                            / treated_indices.len() as f64;

                    let untreated_mean: f64 =
                        untreated_indices.iter().map(|&i| outcome[i]).sum::<f64>()
                            / untreated_indices.len() as f64;

                    treatment_effect = treated_mean - untreated_mean;
                }
            }
            CounterfactualStrategy::PropensityScoreMatching { matching_tolerance } => {
                // Compute propensity scores (probability of treatment)
                let treatment_mean = treatment.mean().unwrap();

                for i in 0..x.nrows() {
                    let feature_sum: f64 = x.row(i).iter().sum();
                    let normalized_sum = feature_sum / x.ncols() as f64;
                    propensity_scores[i] = (normalized_sum + treatment_mean) / 2.0;
                }

                // Compute treatment effect using propensity score matching
                treatment_effect = self.compute_matching_effect(
                    &treatment,
                    &outcome,
                    &propensity_scores,
                    *matching_tolerance,
                );
            }
            CounterfactualStrategy::DoublyRobust { propensity_weight } => {
                // Doubly robust estimation combining outcome modeling and propensity scoring
                let treatment_mean = treatment.mean().unwrap();

                // Compute propensity scores
                for i in 0..x.nrows() {
                    let feature_sum: f64 = x.row(i).iter().sum();
                    let normalized_sum = feature_sum / x.ncols() as f64;
                    propensity_scores[i] = (normalized_sum + treatment_mean) / 2.0;
                }

                // Compute outcome model parameters
                for j in 0..x.ncols() {
                    let col_j = x.column(j);
                    let correlation = self.compute_correlation(&col_j, &outcome);
                    outcome_model_params[j] = correlation;
                }

                // Doubly robust treatment effect
                treatment_effect = self.compute_doubly_robust_effect(
                    &treatment,
                    &outcome,
                    &propensity_scores,
                    &outcome_model_params,
                    *propensity_weight,
                );
            }
            CounterfactualStrategy::InverseProbabilityWeighting { weight_truncation } => {
                // IPW estimation
                let treatment_mean = treatment.mean().unwrap();

                for i in 0..x.nrows() {
                    let feature_sum: f64 = x.row(i).iter().sum();
                    let normalized_sum = feature_sum / x.ncols() as f64;
                    propensity_scores[i] = (normalized_sum + treatment_mean) / 2.0;

                    // Truncate extreme weights
                    propensity_scores[i] = propensity_scores[i]
                        .max(*weight_truncation)
                        .min(1.0 - *weight_truncation);
                }

                treatment_effect =
                    self.compute_ipw_effect(&treatment, &outcome, &propensity_scores);
            }
            CounterfactualStrategy::TargetedMLE { targeting_steps } => {
                // Simplified TMLE
                let treatment_mean = treatment.mean().unwrap();

                for i in 0..x.nrows() {
                    let feature_sum: f64 = x.row(i).iter().sum();
                    let normalized_sum = feature_sum / x.ncols() as f64;
                    propensity_scores[i] = (normalized_sum + treatment_mean) / 2.0;
                }

                treatment_effect = self.compute_tmle_effect(
                    &treatment,
                    &outcome,
                    &propensity_scores,
                    *targeting_steps,
                );
            }
        }

        Ok(FittedCounterfactualBaseline {
            strategy: self.strategy,
            treatment_column: self.treatment_column,
            outcome_column: self.outcome_column,
            treatment_effect,
            propensity_scores,
            outcome_model_params,
            random_state: self.random_state,
        })
    }
}

impl CounterfactualBaseline {
    fn compute_correlation(&self, x: &ArrayView1<f64>, y: &ArrayView1<f64>) -> f64 {
        let n = x.len() as f64;
        let mean_x = x.mean().unwrap();
        let mean_y = y.mean().unwrap();

        let numerator: f64 = x
            .iter()
            .zip(y.iter())
            .map(|(xi, yi)| (xi - mean_x) * (yi - mean_y))
            .sum();

        let var_x: f64 = x.iter().map(|xi| (xi - mean_x).powi(2)).sum();
        let var_y: f64 = y.iter().map(|yi| (yi - mean_y).powi(2)).sum();

        if var_x == 0.0 || var_y == 0.0 {
            0.0
        } else {
            numerator / (var_x * var_y).sqrt()
        }
    }

    fn compute_matching_effect(
        &self,
        treatment: &ArrayView1<f64>,
        outcome: &ArrayView1<f64>,
        propensity_scores: &Array1<f64>,
        tolerance: f64,
    ) -> f64 {
        let mut treated_outcomes = Vec::new();
        let mut untreated_outcomes = Vec::new();

        for i in 0..treatment.len() {
            if treatment[i] > 0.5 {
                // Find matching untreated unit
                let mut best_match_outcome = None;
                let mut min_distance = f64::INFINITY;

                for j in 0..treatment.len() {
                    if treatment[j] <= 0.5 {
                        let distance = (propensity_scores[i] - propensity_scores[j]).abs();
                        if distance < tolerance && distance < min_distance {
                            min_distance = distance;
                            best_match_outcome = Some(outcome[j]);
                        }
                    }
                }

                if let Some(matched_outcome) = best_match_outcome {
                    treated_outcomes.push(outcome[i]);
                    untreated_outcomes.push(matched_outcome);
                }
            }
        }

        if treated_outcomes.is_empty() {
            0.0
        } else {
            let treated_mean = treated_outcomes.iter().sum::<f64>() / treated_outcomes.len() as f64;
            let untreated_mean =
                untreated_outcomes.iter().sum::<f64>() / untreated_outcomes.len() as f64;
            treated_mean - untreated_mean
        }
    }

    fn compute_doubly_robust_effect(
        &self,
        treatment: &ArrayView1<f64>,
        outcome: &ArrayView1<f64>,
        propensity_scores: &Array1<f64>,
        outcome_params: &Array1<f64>,
        weight: f64,
    ) -> f64 {
        let mut effect_sum = 0.0;
        let mut count = 0;

        for i in 0..treatment.len() {
            let propensity = propensity_scores[i].max(0.01).min(0.99);
            let weight_treated = treatment[i] / propensity;
            let weight_untreated = (1.0 - treatment[i]) / (1.0 - propensity);

            // Simplified doubly robust estimator
            let outcome_pred = outcome_params.mean().unwrap_or(0.0);
            let dr_component =
                weight * (weight_treated - weight_untreated) * (outcome[i] - outcome_pred);

            effect_sum += dr_component;
            count += 1;
        }

        if count > 0 {
            effect_sum / count as f64
        } else {
            0.0
        }
    }

    fn compute_ipw_effect(
        &self,
        treatment: &ArrayView1<f64>,
        outcome: &ArrayView1<f64>,
        propensity_scores: &Array1<f64>,
    ) -> f64 {
        let mut treated_sum = 0.0;
        let mut treated_weight_sum = 0.0;
        let mut untreated_sum = 0.0;
        let mut untreated_weight_sum = 0.0;

        for i in 0..treatment.len() {
            let propensity = propensity_scores[i].max(0.01).min(0.99);

            if treatment[i] > 0.5 {
                let weight = 1.0 / propensity;
                treated_sum += weight * outcome[i];
                treated_weight_sum += weight;
            } else {
                let weight = 1.0 / (1.0 - propensity);
                untreated_sum += weight * outcome[i];
                untreated_weight_sum += weight;
            }
        }

        let treated_mean = if treated_weight_sum > 0.0 {
            treated_sum / treated_weight_sum
        } else {
            0.0
        };

        let untreated_mean = if untreated_weight_sum > 0.0 {
            untreated_sum / untreated_weight_sum
        } else {
            0.0
        };

        treated_mean - untreated_mean
    }

    fn compute_tmle_effect(
        &self,
        treatment: &ArrayView1<f64>,
        outcome: &ArrayView1<f64>,
        propensity_scores: &Array1<f64>,
        steps: usize,
    ) -> f64 {
        // Simplified TMLE with targeting steps
        let mut current_outcome = outcome.to_owned();

        for _step in 0..steps {
            let mut adjustment_sum = 0.0;
            let mut count = 0;

            for i in 0..treatment.len() {
                let propensity = propensity_scores[i].max(0.01).min(0.99);
                let clever_covariate =
                    treatment[i] / propensity - (1.0 - treatment[i]) / (1.0 - propensity);
                adjustment_sum += clever_covariate * current_outcome[i];
                count += 1;
            }

            if count > 0 {
                let adjustment = adjustment_sum / count as f64;
                for i in 0..current_outcome.len() {
                    current_outcome[i] += 0.01 * adjustment; // Small step size
                }
            }
        }

        // Compute final treatment effect
        let mut treated_sum = 0.0;
        let mut treated_count = 0;
        let mut untreated_sum = 0.0;
        let mut untreated_count = 0;

        for i in 0..treatment.len() {
            if treatment[i] > 0.5 {
                treated_sum += current_outcome[i];
                treated_count += 1;
            } else {
                untreated_sum += current_outcome[i];
                untreated_count += 1;
            }
        }

        let treated_mean = if treated_count > 0 {
            treated_sum / treated_count as f64
        } else {
            0.0
        };

        let untreated_mean = if untreated_count > 0 {
            untreated_sum / untreated_count as f64
        } else {
            0.0
        };

        treated_mean - untreated_mean
    }
}

impl Predict<Array2<f64>, Array1<f64>> for FittedCounterfactualBaseline {
    fn predict(&self, x: &Array2<f64>) -> Result<Array1<f64>, SklearsError> {
        let mut predictions = Vec::with_capacity(x.nrows());

        for sample in x.rows() {
            let treatment_value = if self.treatment_column < sample.len() {
                sample[self.treatment_column]
            } else {
                0.5 // Default neutral treatment
            };

            // Counterfactual prediction: what would happen if treatment was flipped
            let counterfactual_effect = if treatment_value > 0.5 {
                -self.treatment_effect // If treated, predict untreated outcome
            } else {
                self.treatment_effect // If untreated, predict treated outcome
            };

            let baseline_prediction = if self.outcome_column < sample.len() {
                sample[self.outcome_column]
            } else {
                0.0
            };

            predictions.push(baseline_prediction + counterfactual_effect);
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
    fn test_causal_discovery_correlation() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 3.0, 4.0, 5.0, 4.0, 5.0, 6.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let discovery =
            CausalDiscoveryBaseline::new(CausalDiscoveryStrategy::Correlation { threshold: 0.5 });
        let fitted = discovery.fit(&x, &y).unwrap();

        assert_eq!(fitted.causal_graph.shape(), &[3, 3]);
        assert_eq!(fitted.feature_names.len(), 3);
    }

    #[test]
    fn test_counterfactual_outcome_modeling() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 3.0, 0.0, 4.0, 4.0, 1.0, 5.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let counterfactual = CounterfactualBaseline::new(
            CounterfactualStrategy::OutcomeModeling {
                treatment_effect: 1.0,
            },
            1, // treatment column
            2, // outcome column
        );
        let fitted = counterfactual.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.len(), 4);
        assert!(fitted.treatment_effect.is_finite());
    }

    #[test]
    fn test_causal_discovery_strategies() {
        let x =
            Array2::from_shape_vec((4, 2), vec![1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 5.0]).unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![
            CausalDiscoveryStrategy::Correlation { threshold: 0.3 },
            CausalDiscoveryStrategy::ConditionalIndependence {
                significance_level: 0.05,
            },
            CausalDiscoveryStrategy::PCAlgorithm { alpha: 0.05 },
            CausalDiscoveryStrategy::GrangerCausality { max_lags: 2 },
            CausalDiscoveryStrategy::ConstraintBased {
                edge_threshold: 0.1,
            },
        ];

        for strategy in strategies {
            let discovery = CausalDiscoveryBaseline::new(strategy);
            let fitted = discovery.fit(&x, &y).unwrap();
            assert_eq!(fitted.causal_graph.shape(), &[2, 2]);
        }
    }

    #[test]
    fn test_counterfactual_strategies() {
        let x = Array2::from_shape_vec(
            (4, 3),
            vec![1.0, 0.0, 2.0, 2.0, 1.0, 3.0, 3.0, 0.0, 4.0, 4.0, 1.0, 5.0],
        )
        .unwrap();
        let y = array![1.0, 2.0, 3.0, 4.0];

        let strategies = vec![
            CounterfactualStrategy::OutcomeModeling {
                treatment_effect: 1.0,
            },
            CounterfactualStrategy::PropensityScoreMatching {
                matching_tolerance: 0.1,
            },
            CounterfactualStrategy::DoublyRobust {
                propensity_weight: 0.5,
            },
            CounterfactualStrategy::InverseProbabilityWeighting {
                weight_truncation: 0.05,
            },
            CounterfactualStrategy::TargetedMLE { targeting_steps: 3 },
        ];

        for strategy in strategies {
            let counterfactual = CounterfactualBaseline::new(strategy, 1, 2);
            let fitted = counterfactual.fit(&x, &y).unwrap();
            let predictions = fitted.predict(&x).unwrap();
            assert_eq!(predictions.len(), 4);
        }
    }
}
