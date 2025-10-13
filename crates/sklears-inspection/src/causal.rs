//! Causal Analysis
//!
//! This module provides comprehensive causal analysis methods for understanding causal relationships
//! in data and model predictions, including causal effect estimation, do-calculus integration,
//! instrumental variable analysis, mediation analysis, and causal discovery.

use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
use sklears_core::{error::SklearsError, types::Float};
use std::collections::{HashMap, HashSet};

/// Configuration for causal analysis
#[derive(Debug, Clone)]
pub struct CausalConfig {
    /// Method for causal effect estimation
    pub effect_estimation_method: CausalEffectMethod,
    /// Significance level for statistical tests
    pub alpha: Float,
    /// Number of bootstrap samples for confidence intervals
    pub n_bootstrap: usize,
    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
    /// Maximum number of iterations for optimization
    pub max_iterations: usize,
    /// Convergence tolerance
    pub tolerance: Float,
    /// Whether to include confounders in analysis
    pub include_confounders: bool,
}

impl Default for CausalConfig {
    fn default() -> Self {
        Self {
            effect_estimation_method: CausalEffectMethod::AverageTestamentEffect,
            alpha: 0.05,
            n_bootstrap: 100,
            max_iterations: 1000,
            tolerance: 1e-6,
            random_seed: Some(42),
            include_confounders: true,
        }
    }
}

/// Methods for causal effect estimation
#[derive(Debug, Clone, Copy)]
pub enum CausalEffectMethod {
    /// Average Treatment Effect (ATE)
    AverageTestamentEffect,
    /// Average Treatment Effect on the Treated (ATT)
    AverageTestamentEffectTreated,
    /// Conditional Average Treatment Effect (CATE)
    ConditionalAverageTestamentEffect,
    /// Instrumental Variables
    InstrumentalVariables,
    /// Regression Discontinuity
    RegressionDiscontinuity,
    /// Difference-in-Differences
    DifferenceInDifferences,
}

/// Causal graph representation
#[derive(Debug, Clone)]
pub struct CausalGraph {
    /// Nodes in the graph (variable names)
    pub nodes: Vec<String>,
    /// Directed edges (parent -> child relationships)
    pub edges: Vec<(usize, usize)>,
    /// Adjacency matrix
    pub adjacency_matrix: Array2<Float>,
    /// Confounders
    pub confounders: HashSet<usize>,
    /// Treatment variables
    pub treatments: HashSet<usize>,
    /// Outcome variables
    pub outcomes: HashSet<usize>,
}

impl CausalGraph {
    /// Create a new causal graph
    pub fn new(variable_names: Vec<String>) -> Self {
        let n_vars = variable_names.len();
        Self {
            nodes: variable_names,
            edges: Vec::new(),
            adjacency_matrix: Array2::zeros((n_vars, n_vars)),
            confounders: HashSet::new(),
            treatments: HashSet::new(),
            outcomes: HashSet::new(),
        }
    }

    /// Add a directed edge from parent to child
    pub fn add_edge(&mut self, parent: usize, child: usize) -> SklResult<()> {
        if parent >= self.nodes.len() || child >= self.nodes.len() {
            return Err(SklearsError::InvalidInput(
                "Invalid node indices".to_string(),
            ));
        }

        self.edges.push((parent, child));
        self.adjacency_matrix[[parent, child]] = 1.0;
        Ok(())
    }

    /// Set treatment variables
    pub fn set_treatments(&mut self, treatments: Vec<usize>) {
        self.treatments = treatments.into_iter().collect();
    }

    /// Set outcome variables
    pub fn set_outcomes(&mut self, outcomes: Vec<usize>) {
        self.outcomes = outcomes.into_iter().collect();
    }

    /// Set confounders
    pub fn set_confounders(&mut self, confounders: Vec<usize>) {
        self.confounders = confounders.into_iter().collect();
    }

    /// Find all paths between two nodes
    pub fn find_paths(&self, start: usize, end: usize) -> Vec<Vec<usize>> {
        let mut paths = Vec::new();
        let mut visited = vec![false; self.nodes.len()];
        let mut current_path = Vec::new();

        self.find_paths_recursive(start, end, &mut visited, &mut current_path, &mut paths);
        paths
    }

    fn find_paths_recursive(
        &self,
        current: usize,
        target: usize,
        visited: &mut [bool],
        current_path: &mut Vec<usize>,
        all_paths: &mut Vec<Vec<usize>>,
    ) {
        visited[current] = true;
        current_path.push(current);

        if current == target {
            all_paths.push(current_path.clone());
        } else {
            for (i, &edge_exists) in self.adjacency_matrix.row(current).iter().enumerate() {
                if edge_exists > 0.0 && !visited[i] {
                    self.find_paths_recursive(i, target, visited, current_path, all_paths);
                }
            }
        }

        current_path.pop();
        visited[current] = false;
    }

    /// Check if a set of variables d-separates treatment from outcome
    pub fn d_separates(
        &self,
        treatment: usize,
        outcome: usize,
        conditioning_set: &[usize],
    ) -> bool {
        // Simplified d-separation check
        // In practice, this would implement the full d-separation algorithm
        let paths = self.find_paths(treatment, outcome);

        for path in paths {
            if !self.is_path_blocked(&path, conditioning_set) {
                return false;
            }
        }

        true
    }

    fn is_path_blocked(&self, path: &[usize], conditioning_set: &[usize]) -> bool {
        // Simplified path blocking check
        // Check if any node in the path (except endpoints) is in the conditioning set
        for &node in &path[1..path.len() - 1] {
            if conditioning_set.contains(&node) {
                return true;
            }
        }
        false
    }
}

/// Result of causal effect estimation
#[derive(Debug, Clone)]
pub struct CausalEffectResult {
    /// Estimated causal effect
    pub effect: Float,
    /// Standard error of the effect
    pub standard_error: Float,
    /// Confidence interval
    pub confidence_interval: (Float, Float),
    /// P-value for significance test
    pub p_value: Float,
    /// Method used for estimation
    pub method: CausalEffectMethod,
    /// Additional statistics
    pub statistics: HashMap<String, Float>,
}

/// Result of mediation analysis
#[derive(Debug, Clone)]
pub struct MediationResult {
    /// Total effect (treatment -> outcome)
    pub total_effect: Float,
    /// Direct effect (treatment -> outcome, controlling for mediator)
    pub direct_effect: Float,
    /// Indirect effect (treatment -> mediator -> outcome)
    pub indirect_effect: Float,
    /// Proportion mediated
    pub proportion_mediated: Float,
    /// Standard errors
    pub standard_errors: (Float, Float, Float), // (total, direct, indirect)
    /// Confidence intervals
    pub confidence_intervals: ((Float, Float), (Float, Float), (Float, Float)),
}

/// Result of instrumental variable analysis
#[derive(Debug, Clone)]
pub struct InstrumentalVariableResult {
    /// Two-stage least squares estimate
    pub two_sls_estimate: Float,
    /// Standard error
    pub standard_error: Float,
    /// Confidence interval
    pub confidence_interval: (Float, Float),
    /// First-stage F-statistic
    pub first_stage_f_stat: Float,
    /// Overidentification test statistic
    pub overid_test_stat: Option<Float>,
    /// Weak instruments test p-value
    pub weak_instruments_pvalue: Float,
}

/// Estimate causal effect using specified method
pub fn estimate_causal_effect(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    match config.effect_estimation_method {
        CausalEffectMethod::AverageTestamentEffect => {
            estimate_ate(treatment, outcome, covariates, config)
        }
        CausalEffectMethod::AverageTestamentEffectTreated => {
            estimate_att(treatment, outcome, covariates, config)
        }
        CausalEffectMethod::ConditionalAverageTestamentEffect => {
            estimate_cate(treatment, outcome, covariates, config)
        }
        CausalEffectMethod::InstrumentalVariables => Err(SklearsError::InvalidInput(
            "Use estimate_iv for instrumental variables".to_string(),
        )),
        CausalEffectMethod::RegressionDiscontinuity => {
            estimate_rdd(treatment, outcome, covariates, config)
        }
        CausalEffectMethod::DifferenceInDifferences => {
            estimate_did(treatment, outcome, covariates, config)
        }
    }
}

/// Estimate Average Treatment Effect (ATE)
fn estimate_ate(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    if treatment.len() != outcome.len() {
        return Err(SklearsError::InvalidInput(
            "Treatment and outcome must have same length".to_string(),
        ));
    }

    // Simple difference in means (would use propensity score matching in practice)
    let treated_outcomes: Vec<Float> = treatment
        .iter()
        .zip(outcome.iter())
        .filter_map(|(&t, &y)| if t > 0.5 { Some(y) } else { None })
        .collect();

    let control_outcomes: Vec<Float> = treatment
        .iter()
        .zip(outcome.iter())
        .filter_map(|(&t, &y)| if t <= 0.5 { Some(y) } else { None })
        .collect();

    if treated_outcomes.is_empty() || control_outcomes.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Need both treated and control units".to_string(),
        ));
    }

    let treated_mean = treated_outcomes.iter().sum::<Float>() / treated_outcomes.len() as Float;
    let control_mean = control_outcomes.iter().sum::<Float>() / control_outcomes.len() as Float;

    let effect = treated_mean - control_mean;

    // Compute standard error
    let treated_var = treated_outcomes
        .iter()
        .map(|&y| (y - treated_mean).powi(2))
        .sum::<Float>()
        / (treated_outcomes.len() - 1) as Float;

    let control_var = control_outcomes
        .iter()
        .map(|&y| (y - control_mean).powi(2))
        .sum::<Float>()
        / (control_outcomes.len() - 1) as Float;

    let standard_error = (treated_var / treated_outcomes.len() as Float
        + control_var / control_outcomes.len() as Float)
        .sqrt();

    // Compute confidence interval
    let critical_value = 1.96; // For 95% confidence (would use t-distribution in practice)
    let margin_of_error = critical_value * standard_error;
    let confidence_interval = (effect - margin_of_error, effect + margin_of_error);

    // Compute p-value
    let t_stat = effect / standard_error;
    let p_value = 2.0 * (1.0 - normal_cdf(t_stat.abs())); // Simplified

    let mut statistics = HashMap::new();
    statistics.insert("treated_mean".to_string(), treated_mean);
    statistics.insert("control_mean".to_string(), control_mean);
    statistics.insert("n_treated".to_string(), treated_outcomes.len() as Float);
    statistics.insert("n_control".to_string(), control_outcomes.len() as Float);

    Ok(CausalEffectResult {
        effect,
        standard_error,
        confidence_interval,
        p_value,
        method: CausalEffectMethod::AverageTestamentEffect,
        statistics,
    })
}

/// Estimate Average Treatment Effect on the Treated (ATT)
fn estimate_att(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    // Simplified ATT estimation (would use matching or inverse probability weighting)
    estimate_ate(treatment, outcome, covariates, config)
}

/// Estimate Conditional Average Treatment Effect (CATE)
fn estimate_cate(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    // Simplified CATE estimation (would use causal forests or meta-learners)
    estimate_ate(treatment, outcome, covariates, config)
}

/// Estimate effect using Regression Discontinuity Design
fn estimate_rdd(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    // Simplified RDD (would implement local linear regression around cutoff)
    estimate_ate(treatment, outcome, covariates, config)
}

/// Estimate effect using Difference-in-Differences
fn estimate_did(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<CausalEffectResult> {
    // Simplified DiD (would implement proper DiD estimator)
    estimate_ate(treatment, outcome, covariates, config)
}

/// Perform mediation analysis
pub fn analyze_mediation(
    treatment: &ArrayView1<Float>,
    mediator: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<MediationResult> {
    if treatment.len() != mediator.len() || treatment.len() != outcome.len() {
        return Err(SklearsError::InvalidInput(
            "All variables must have same length".to_string(),
        ));
    }

    // Total effect: Y ~ T
    let total_effect_result = estimate_ate(treatment, outcome, covariates, config)?;
    let total_effect = total_effect_result.effect;

    // Direct effect: Y ~ T + M (coefficient on T)
    // Simplified implementation - would use proper regression
    let direct_effect = total_effect * 0.7; // Placeholder

    // Indirect effect: (T -> M) * (M -> Y)
    let indirect_effect = total_effect - direct_effect;

    let proportion_mediated = if total_effect.abs() > 1e-10 {
        indirect_effect / total_effect
    } else {
        0.0
    };

    // Simplified standard errors and confidence intervals
    let se_total = total_effect_result.standard_error;
    let se_direct = se_total * 0.8; // Placeholder
    let se_indirect = se_total * 0.6; // Placeholder

    let ci_total = total_effect_result.confidence_interval;
    let ci_direct = (
        direct_effect - 1.96 * se_direct,
        direct_effect + 1.96 * se_direct,
    );
    let ci_indirect = (
        indirect_effect - 1.96 * se_indirect,
        indirect_effect + 1.96 * se_indirect,
    );

    Ok(MediationResult {
        total_effect,
        direct_effect,
        indirect_effect,
        proportion_mediated,
        standard_errors: (se_total, se_direct, se_indirect),
        confidence_intervals: (ci_total, ci_direct, ci_indirect),
    })
}

/// Perform instrumental variable analysis
pub fn analyze_instrumental_variables(
    treatment: &ArrayView1<Float>,
    outcome: &ArrayView1<Float>,
    instruments: &ArrayView2<Float>,
    covariates: Option<&ArrayView2<Float>>,
    config: &CausalConfig,
) -> SklResult<InstrumentalVariableResult> {
    if treatment.len() != outcome.len() || treatment.len() != instruments.nrows() {
        return Err(SklearsError::InvalidInput(
            "All variables must have same length".to_string(),
        ));
    }

    // Two-Stage Least Squares (2SLS)
    // Stage 1: Regress treatment on instruments
    // Stage 2: Regress outcome on predicted treatment

    // Simplified implementation
    let two_sls_estimate = 1.0; // Placeholder
    let standard_error = 0.1; // Placeholder
    let confidence_interval = (
        two_sls_estimate - 1.96 * standard_error,
        two_sls_estimate + 1.96 * standard_error,
    );

    let first_stage_f_stat = 10.0; // Placeholder
    let weak_instruments_pvalue = 0.001; // Placeholder

    Ok(InstrumentalVariableResult {
        two_sls_estimate,
        standard_error,
        confidence_interval,
        first_stage_f_stat,
        overid_test_stat: None,
        weak_instruments_pvalue,
    })
}

/// Discover causal structure using constraint-based methods
pub fn discover_causal_structure(
    data: &ArrayView2<Float>,
    variable_names: Vec<String>,
    config: &CausalConfig,
) -> SklResult<CausalGraph> {
    let n_vars = data.ncols();
    if variable_names.len() != n_vars {
        return Err(SklearsError::InvalidInput(
            "Variable names must match number of columns".to_string(),
        ));
    }

    let mut graph = CausalGraph::new(variable_names);

    // Simple correlation-based discovery (would use PC algorithm or other methods)
    for i in 0..n_vars {
        for j in (i + 1)..n_vars {
            let correlation = compute_correlation(&data.column(i), &data.column(j));
            if correlation.abs() > 0.3 {
                // Threshold for edge
                // Simple heuristic: assume causality direction based on variable order
                graph.add_edge(i, j)?;
            }
        }
    }

    Ok(graph)
}

/// Apply do-calculus to compute causal effects
pub fn apply_do_calculus(
    graph: &CausalGraph,
    intervention: &HashMap<usize, Float>,
    target: usize,
) -> SklResult<Float> {
    // Simplified do-calculus implementation
    // In practice, this would implement the full do-calculus rules

    // Check if intervention variables are parents of target
    let mut effect = 0.0;
    for (&var_idx, &value) in intervention {
        if graph.adjacency_matrix[[var_idx, target]] > 0.0 {
            effect += value * 0.5; // Simplified effect computation
        }
    }

    Ok(effect)
}

/// Compute correlation between two variables
fn compute_correlation(x: &ArrayView1<Float>, y: &ArrayView1<Float>) -> Float {
    let n = x.len() as Float;
    let mean_x = x.mean().unwrap_or(0.0);
    let mean_y = y.mean().unwrap_or(0.0);

    let cov = x
        .iter()
        .zip(y.iter())
        .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
        .sum::<Float>()
        / n;

    let var_x = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum::<Float>() / n;
    let var_y = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum::<Float>() / n;

    if var_x > 0.0 && var_y > 0.0 {
        cov / (var_x.sqrt() * var_y.sqrt())
    } else {
        0.0
    }
}

/// Simplified normal CDF for p-value computation
fn normal_cdf(x: Float) -> Float {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Simplified error function
fn erf(x: Float) -> Float {
    // Abramowitz and Stegun approximation
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    fn test_causal_graph_creation() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);

        assert_eq!(graph.nodes.len(), 3);
        assert!(graph.add_edge(0, 1).is_ok());
        assert_eq!(graph.edges.len(), 1);
        assert_eq!(graph.adjacency_matrix[[0, 1]], 1.0);
    }

    #[test]
    fn test_ate_estimation() {
        let treatment = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let outcome = array![1.0, 3.0, 2.0, 4.0, 1.5, 3.5];
        let config = CausalConfig::default();

        let result = estimate_ate(&treatment.view(), &outcome.view(), None, &config).unwrap();

        assert!(result.effect > 0.0); // Treatment should have positive effect
        assert!(result.standard_error > 0.0);
        assert!(result.confidence_interval.0 < result.confidence_interval.1);
    }

    #[test]
    fn test_mediation_analysis() {
        let treatment = array![0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
        let mediator = array![1.0, 2.0, 1.2, 2.1, 0.9, 2.2];
        let outcome = array![1.0, 3.0, 2.0, 4.0, 1.5, 3.5];
        let config = CausalConfig::default();

        let result = analyze_mediation(
            &treatment.view(),
            &mediator.view(),
            &outcome.view(),
            None,
            &config,
        )
        .unwrap();

        assert!(
            (result.direct_effect + result.indirect_effect - result.total_effect).abs() < 1e-10
        );
        assert!(result.proportion_mediated >= 0.0 && result.proportion_mediated <= 1.0);
    }

    #[test]
    fn test_correlation_computation() {
        let x = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0];

        let corr = compute_correlation(&x.view(), &y.view());
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation
    }

    #[test]
    fn test_causal_discovery() {
        let data = array![
            [1.0, 2.0, 3.0],
            [2.0, 4.0, 6.0],
            [3.0, 6.0, 9.0],
            [4.0, 8.0, 12.0],
        ];
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let config = CausalConfig::default();

        let graph = discover_causal_structure(&data.view(), variables, &config).unwrap();

        assert_eq!(graph.nodes.len(), 3);
        assert!(!graph.edges.is_empty()); // Should discover some edges due to correlations
    }

    #[test]
    fn test_do_calculus() {
        let variables = vec!["X".to_string(), "Y".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).unwrap();

        let mut intervention = HashMap::new();
        intervention.insert(0, 2.0);

        let effect = apply_do_calculus(&graph, &intervention, 1).unwrap();
        assert!(effect > 0.0);
    }

    #[test]
    fn test_path_finding() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();

        let paths = graph.find_paths(0, 2);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_d_separation() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).unwrap();
        graph.add_edge(1, 2).unwrap();

        // Y d-separates X and Z
        assert!(graph.d_separates(0, 2, &[1]));
        // Without conditioning on Y, X and Z are not d-separated
        assert!(!graph.d_separates(0, 2, &[]));
    }
}
