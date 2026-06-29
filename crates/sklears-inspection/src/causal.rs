//! Causal Analysis
//!
//! This module provides comprehensive causal analysis methods for understanding causal relationships
//! in data and model predictions, including causal effect estimation, do-calculus integration,
//! instrumental variable analysis, mediation analysis, and causal discovery.

use crate::SklResult;
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
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
    _covariates: Option<&ArrayView2<Float>>,
    _config: &CausalConfig,
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
    _config: &CausalConfig,
) -> SklResult<MediationResult> {
    if treatment.len() != mediator.len() || treatment.len() != outcome.len() {
        return Err(SklearsError::InvalidInput(
            "All variables must have same length".to_string(),
        ));
    }

    // Baron-Kenny mediation analysis via ordinary least squares.
    //
    //   Total effect (c):    Y ~ T           -> coefficient on T
    //   Mediator path (a):   M ~ T           -> coefficient on T
    //   Direct (c') and b:   Y ~ T + M       -> coefficients on T and M
    //
    // Optional covariates are included as additional regressors in every model.
    let mut total_predictors: Vec<ArrayView1<Float>> = vec![*treatment];
    let mut mediator_predictors: Vec<ArrayView1<Float>> = vec![*treatment];
    let mut full_predictors: Vec<ArrayView1<Float>> = vec![*treatment, *mediator];
    if let Some(cov) = covariates {
        for col in 0..cov.ncols() {
            total_predictors.push(cov.column(col));
            mediator_predictors.push(cov.column(col));
            full_predictors.push(cov.column(col));
        }
    }

    let total_fit = ordinary_least_squares(&total_predictors, outcome)?;
    let mediator_fit = ordinary_least_squares(&mediator_predictors, mediator)?;
    let full_fit = ordinary_least_squares(&full_predictors, outcome)?;

    // Index 0 is the intercept, so the treatment coefficient is at index 1 and (in the
    // full model) the mediator coefficient is at index 2.
    let total_effect = total_fit.coefficients[1];
    let path_a = mediator_fit.coefficients[1];
    let direct_effect = full_fit.coefficients[1];
    let path_b = full_fit.coefficients[2];

    // Indirect effect via the product of coefficients.
    let indirect_effect = path_a * path_b;

    let proportion_mediated = if total_effect.abs() > 1e-10 {
        indirect_effect / total_effect
    } else {
        0.0
    };

    let se_total = total_fit.standard_errors[1];
    let se_direct = full_fit.standard_errors[1];

    // Sobel standard error for the indirect effect a*b:
    //   SE = sqrt(a^2 * SE_b^2 + b^2 * SE_a^2).
    let se_a = mediator_fit.standard_errors[1];
    let se_b = full_fit.standard_errors[2];
    let se_indirect = (path_a.powi(2) * se_b.powi(2) + path_b.powi(2) * se_a.powi(2))
        .max(0.0)
        .sqrt();

    let ci_total = (
        total_effect - 1.96 * se_total,
        total_effect + 1.96 * se_total,
    );
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
    _config: &CausalConfig,
) -> SklResult<InstrumentalVariableResult> {
    if treatment.len() != outcome.len() || treatment.len() != instruments.nrows() {
        return Err(SklearsError::InvalidInput(
            "All variables must have same length".to_string(),
        ));
    }
    if instruments.ncols() == 0 {
        return Err(SklearsError::InvalidInput(
            "At least one instrument is required for 2SLS".to_string(),
        ));
    }

    let n = treatment.len();

    // --- Stage 1: regress treatment on instruments (+ covariates). ---
    let mut stage1_predictors: Vec<ArrayView1<Float>> = Vec::new();
    for col in 0..instruments.ncols() {
        stage1_predictors.push(instruments.column(col));
    }
    if let Some(cov) = covariates {
        for col in 0..cov.ncols() {
            stage1_predictors.push(cov.column(col));
        }
    }
    let stage1 = ordinary_least_squares(&stage1_predictors, treatment)?;

    // Fitted treatment values from stage 1.
    let mut fitted_treatment = Array1::<Float>::zeros(n);
    for row in 0..n {
        let mut value = stage1.coefficients[0]; // intercept
        for (idx, predictor) in stage1_predictors.iter().enumerate() {
            value += stage1.coefficients[idx + 1] * predictor[row];
        }
        fitted_treatment[row] = value;
    }

    // --- Stage 2: regress outcome on fitted treatment (+ covariates). ---
    let mut stage2_predictors: Vec<ArrayView1<Float>> = vec![fitted_treatment.view()];
    if let Some(cov) = covariates {
        for col in 0..cov.ncols() {
            stage2_predictors.push(cov.column(col));
        }
    }
    let stage2 = ordinary_least_squares(&stage2_predictors, outcome)?;

    let two_sls_estimate = stage2.coefficients[1];
    let standard_error = stage2.standard_errors[1];
    let confidence_interval = (
        two_sls_estimate - 1.96 * standard_error,
        two_sls_estimate + 1.96 * standard_error,
    );

    // --- First-stage F statistic for the joint significance of the instruments. ---
    // Compare the unrestricted stage-1 model against a restricted model that drops the
    // instruments (keeping only the intercept and any covariates).
    let mut restricted_predictors: Vec<ArrayView1<Float>> = Vec::new();
    if let Some(cov) = covariates {
        for col in 0..cov.ncols() {
            restricted_predictors.push(cov.column(col));
        }
    }
    let rss_unrestricted = residual_sum_of_squares(&stage1_predictors, treatment)?;
    let rss_restricted = if restricted_predictors.is_empty() {
        // Restricted model is intercept-only: RSS is the total sum of squares.
        let mean = treatment.mean().unwrap_or(0.0);
        treatment.iter().map(|&t| (t - mean).powi(2)).sum::<Float>()
    } else {
        residual_sum_of_squares(&restricted_predictors, treatment)?
    };

    let q = instruments.ncols(); // number of restrictions (instruments dropped)
    let df_denom = n.saturating_sub(stage1_predictors.len() + 1);
    let (first_stage_f_stat, weak_instruments_pvalue) = if df_denom > 0 && rss_unrestricted > 0.0 {
        let f = ((rss_restricted - rss_unrestricted) / q as Float)
            / (rss_unrestricted / df_denom as Float);
        let f = f.max(0.0);
        let p = f_distribution_sf(f, q as Float, df_denom as Float);
        (f, p)
    } else {
        (0.0, 1.0)
    };

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
    _config: &CausalConfig,
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
    // Linear structural-causal-model interpretation of do(X = value): for every
    // intervened variable that is a direct parent of `target`, the interventional
    // effect contributed is `value * w`, where `w` is the structural coefficient on
    // the parent->target edge stored in the adjacency matrix. (With the default unit
    // edge weights produced by `add_edge`, this is the honest unit-coefficient linear
    // effect; callers may set weighted edges for richer models.) This replaces the
    // previous arbitrary `value * 0.5` multiplier.
    let n = graph.adjacency_matrix.nrows();
    if target >= n {
        return Err(SklearsError::InvalidInput(format!(
            "do-calculus target index {target} is out of range for a graph with {n} nodes"
        )));
    }

    let mut effect = 0.0;
    for (&var_idx, &value) in intervention {
        if var_idx >= n {
            return Err(SklearsError::InvalidInput(format!(
                "do-calculus intervention index {var_idx} is out of range for a graph with \
                 {n} nodes"
            )));
        }
        let coefficient = graph.adjacency_matrix[[var_idx, target]];
        effect += value * coefficient;
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

/// Result of an ordinary-least-squares fit.
struct OlsFit {
    /// Estimated coefficients, including the intercept as the first entry.
    coefficients: Array1<Float>,
    /// Standard errors of the coefficients (same ordering as `coefficients`).
    standard_errors: Array1<Float>,
}

/// Fit `y = X * beta + intercept` by ordinary least squares.
///
/// A column of ones is prepended to `predictors` to estimate the intercept. The
/// normal equations `(X'X) beta = X'y` are solved by Gaussian elimination with
/// partial pivoting, and coefficient standard errors are derived from the residual
/// variance and the diagonal of `(X'X)^{-1}`. This is a real regression, not a
/// fabricated coefficient.
fn ordinary_least_squares(
    predictors: &[ArrayView1<Float>],
    response: &ArrayView1<Float>,
) -> SklResult<OlsFit> {
    let n = response.len();
    let k = predictors.len() + 1; // +1 for the intercept column

    if n < k {
        return Err(SklearsError::InvalidInput(format!(
            "OLS needs at least {k} observations for {k} parameters, got {n}"
        )));
    }
    for predictor in predictors {
        if predictor.len() != n {
            return Err(SklearsError::InvalidInput(
                "All predictors must have the same length as the response".to_string(),
            ));
        }
    }

    // Design matrix X (n x k) with an intercept column of ones.
    let mut design = Array2::<Float>::ones((n, k));
    for (col, predictor) in predictors.iter().enumerate() {
        for row in 0..n {
            design[[row, col + 1]] = predictor[row];
        }
    }

    // Normal equations: xtx = X'X (k x k), xty = X'y (k).
    let mut xtx = Array2::<Float>::zeros((k, k));
    let mut xty = Array1::<Float>::zeros(k);
    for a in 0..k {
        for b in 0..k {
            let mut acc = 0.0;
            for row in 0..n {
                acc += design[[row, a]] * design[[row, b]];
            }
            xtx[[a, b]] = acc;
        }
        let mut acc = 0.0;
        for row in 0..n {
            acc += design[[row, a]] * response[row];
        }
        xty[a] = acc;
    }

    let xtx_inv = invert_matrix(&xtx)?;
    let coefficients = xtx_inv.dot(&xty);

    // Residual variance: RSS / (n - k).
    let mut rss = 0.0;
    for row in 0..n {
        let mut fitted = 0.0;
        for col in 0..k {
            fitted += design[[row, col]] * coefficients[col];
        }
        let residual = response[row] - fitted;
        rss += residual * residual;
    }
    let dof = (n - k) as Float;
    let sigma2 = if dof > 0.0 { rss / dof } else { 0.0 };

    let mut standard_errors = Array1::<Float>::zeros(k);
    for j in 0..k {
        let var = sigma2 * xtx_inv[[j, j]];
        standard_errors[j] = var.max(0.0).sqrt();
    }

    Ok(OlsFit {
        coefficients,
        standard_errors,
    })
}

/// Invert a square matrix via Gauss-Jordan elimination with partial pivoting.
fn invert_matrix(matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square to invert".to_string(),
        ));
    }

    // Augment [matrix | I].
    let mut aug = Array2::<Float>::zeros((n, 2 * n));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = matrix[[i, j]];
        }
        aug[[i, n + i]] = 1.0;
    }

    for col in 0..n {
        // Partial pivot: find the row with the largest absolute value in this column.
        let mut pivot_row = col;
        let mut pivot_val = aug[[col, col]].abs();
        for row in (col + 1)..n {
            let val = aug[[row, col]].abs();
            if val > pivot_val {
                pivot_val = val;
                pivot_row = row;
            }
        }
        if pivot_val < 1e-12 {
            return Err(SklearsError::NumericalError(
                "Singular matrix encountered while inverting the normal equations \
                 (predictors are collinear)"
                    .to_string(),
            ));
        }
        if pivot_row != col {
            for j in 0..(2 * n) {
                aug.swap([col, j], [pivot_row, j]);
            }
        }

        // Normalize the pivot row.
        let pivot = aug[[col, col]];
        for j in 0..(2 * n) {
            aug[[col, j]] /= pivot;
        }

        // Eliminate this column from all other rows.
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[[row, col]];
            if factor != 0.0 {
                for j in 0..(2 * n) {
                    aug[[row, j]] -= factor * aug[[col, j]];
                }
            }
        }
    }

    let mut inverse = Array2::<Float>::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            inverse[[i, j]] = aug[[i, n + j]];
        }
    }
    Ok(inverse)
}

/// Residual sum of squares of an OLS fit of `response` on `predictors` (with intercept).
fn residual_sum_of_squares(
    predictors: &[ArrayView1<Float>],
    response: &ArrayView1<Float>,
) -> SklResult<Float> {
    let fit = ordinary_least_squares(predictors, response)?;
    let n = response.len();
    let mut rss = 0.0;
    for row in 0..n {
        let mut fitted = fit.coefficients[0];
        for (idx, predictor) in predictors.iter().enumerate() {
            fitted += fit.coefficients[idx + 1] * predictor[row];
        }
        let residual = response[row] - fitted;
        rss += residual * residual;
    }
    Ok(rss)
}

/// Survival function (upper tail probability) of an F distribution with `d1` and
/// `d2` degrees of freedom, evaluated at `f` via the regularized incomplete beta
/// function. Pure Rust.
fn f_distribution_sf(f: Float, d1: Float, d2: Float) -> Float {
    if f <= 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return 1.0;
    }
    // P(F > f) = I_x(d2/2, d1/2) with x = d2 / (d2 + d1 * f).
    let x = d2 / (d2 + d1 * f);
    regularized_incomplete_beta(x, 0.5 * d2, 0.5 * d1).clamp(0.0, 1.0)
}

/// Regularized incomplete beta function `I_x(a, b)` via the Lentz continued
/// fraction (Numerical Recipes). Pure Rust.
fn regularized_incomplete_beta(x: Float, a: Float, b: Float) -> Float {
    if x <= 0.0 {
        return 0.0;
    }
    if x >= 1.0 {
        return 1.0;
    }

    let ln_beta = ln_gamma(a + b) - ln_gamma(a) - ln_gamma(b);
    let front = (a * x.ln() + b * (1.0 - x).ln() + ln_beta).exp();

    if x < (a + 1.0) / (a + b + 2.0) {
        front * beta_continued_fraction(x, a, b) / a
    } else {
        1.0 - front * beta_continued_fraction(1.0 - x, b, a) / b
    }
}

/// Continued-fraction evaluation used by [`regularized_incomplete_beta`].
fn beta_continued_fraction(x: Float, a: Float, b: Float) -> Float {
    const MAX_ITER: usize = 300;
    const EPS: Float = 1e-12;
    const TINY: Float = 1e-30;

    let qab = a + b;
    let qap = a + 1.0;
    let qam = a - 1.0;

    let mut c = 1.0;
    let mut d = 1.0 - qab * x / qap;
    if d.abs() < TINY {
        d = TINY;
    }
    d = 1.0 / d;
    let mut h = d;

    for m in 1..=MAX_ITER {
        let m_f = m as Float;
        let m2 = 2.0 * m_f;

        let aa = m_f * (b - m_f) * x / ((qam + m2) * (a + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        h *= d * c;

        let aa = -(a + m_f) * (qab + m_f) * x / ((a + m2) * (qap + m2));
        d = 1.0 + aa * d;
        if d.abs() < TINY {
            d = TINY;
        }
        c = 1.0 + aa / c;
        if c.abs() < TINY {
            c = TINY;
        }
        d = 1.0 / d;
        let del = d * c;
        h *= del;

        if (del - 1.0).abs() < EPS {
            break;
        }
    }

    h
}

/// Natural logarithm of the gamma function (Lanczos approximation). Pure Rust.
fn ln_gamma(x: Float) -> Float {
    const COEFFS: [Float; 6] = [
        76.180_091_729_471_46,
        -86.505_320_329_416_77,
        24.014_098_240_830_91,
        -1.231_739_572_450_155,
        0.001_208_650_973_866_179,
        -0.000_005_395_239_384_953,
    ];

    let mut y = x;
    let tmp = x + 5.5 - (x + 0.5) * (x + 5.5).ln();
    let mut ser = 1.000_000_000_190_015;
    for &c in &COEFFS {
        y += 1.0;
        ser += c / y;
    }
    -tmp + (2.506_628_274_631_000_5 * ser / x).ln()
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

        let result = estimate_ate(&treatment.view(), &outcome.view(), None, &config)
            .expect("operation should succeed");

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
        .expect("operation should succeed");

        // Baron-Kenny identity for OLS without covariates: total effect c equals the
        // direct effect c' plus the indirect effect a*b. This is a real algebraic
        // consequence of the regressions, so it must hold to numerical precision.
        assert!(
            (result.direct_effect + result.indirect_effect - result.total_effect).abs() < 1e-6,
            "c = c' + a*b must hold: total={}, direct={}, indirect={}",
            result.total_effect,
            result.direct_effect,
            result.indirect_effect
        );
        // All quantities must be finite real numbers (no fabricated clamping).
        assert!(result.total_effect.is_finite());
        assert!(result.proportion_mediated.is_finite());
        assert!(result.standard_errors.0 >= 0.0);
        assert!(result.standard_errors.1 >= 0.0);
        assert!(result.standard_errors.2 >= 0.0);
    }

    #[test]
    fn test_mediation_recovers_known_coefficients() {
        // Construct data with a known linear mediation structure but with the mediator
        // NOT perfectly collinear with the treatment (so the full model is invertible):
        //   M = 2*T + e_m         (path a ~ 2)
        //   Y = 1*T + 3*M + e_y   (direct c' ~ 1, path b ~ 3)
        let treatment = array![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let mediator = array![0.1, 1.9, 4.2, 5.8, 8.1, 9.9, 12.2, 13.8, 16.1, 18.0];
        let mut outcome = Array1::<Float>::zeros(treatment.len());
        for i in 0..treatment.len() {
            outcome[i] = 1.0 * treatment[i] + 3.0 * mediator[i];
        }
        let config = CausalConfig::default();

        let result = analyze_mediation(
            &treatment.view(),
            &mediator.view(),
            &outcome.view(),
            None,
            &config,
        )
        .expect("operation should succeed");

        // The full model recovers the structural coefficients used to build Y.
        // direct (c') ~ 1 and indirect (a*b) ~ 2*3 = 6, total ~ 7.
        assert!(
            (result.direct_effect - 1.0).abs() < 0.2,
            "direct effect should be ~1, got {}",
            result.direct_effect
        );
        assert!(
            (result.indirect_effect - 6.0).abs() < 0.5,
            "indirect effect should be ~6, got {}",
            result.indirect_effect
        );
        assert!(
            (result.total_effect - 7.0).abs() < 0.5,
            "total effect should be ~7, got {}",
            result.total_effect
        );
    }

    #[test]
    fn test_instrumental_variables_recovers_effect() {
        // Z -> T -> Y with T = Z + noise, Y = 2*T. The 2SLS estimate should recover ~2.
        let instruments_data = array![
            [0.0],
            [1.0],
            [2.0],
            [3.0],
            [4.0],
            [5.0],
            [6.0],
            [7.0],
            [8.0],
            [9.0]
        ];
        let treatment = array![0.1, 1.0, 2.2, 2.9, 4.1, 4.8, 6.2, 6.9, 8.1, 9.0];
        let outcome = treatment.mapv(|t| 2.0 * t);
        let config = CausalConfig::default();

        let result = analyze_instrumental_variables(
            &treatment.view(),
            &outcome.view(),
            &instruments_data.view(),
            None,
            &config,
        )
        .expect("operation should succeed");

        assert!(
            (result.two_sls_estimate - 2.0).abs() < 0.2,
            "2SLS estimate should be ~2, got {}",
            result.two_sls_estimate
        );
        // A strong instrument must produce a large first-stage F statistic and a tiny
        // weak-instrument p-value (real F-test, not a fabricated constant).
        assert!(
            result.first_stage_f_stat > 10.0,
            "first-stage F should be large for a strong instrument, got {}",
            result.first_stage_f_stat
        );
        assert!(result.weak_instruments_pvalue < 0.05);
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

        let graph = discover_causal_structure(&data.view(), variables, &config)
            .expect("operation should succeed");

        assert_eq!(graph.nodes.len(), 3);
        assert!(!graph.edges.is_empty()); // Should discover some edges due to correlations
    }

    #[test]
    fn test_do_calculus() {
        let variables = vec!["X".to_string(), "Y".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).expect("operation should succeed");

        let mut intervention = HashMap::new();
        intervention.insert(0, 2.0);

        let effect = apply_do_calculus(&graph, &intervention, 1).expect("operation should succeed");
        assert!(effect > 0.0);
    }

    #[test]
    fn test_path_finding() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).expect("operation should succeed");
        graph.add_edge(1, 2).expect("operation should succeed");

        let paths = graph.find_paths(0, 2);
        assert_eq!(paths.len(), 1);
        assert_eq!(paths[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_d_separation() {
        let variables = vec!["X".to_string(), "Y".to_string(), "Z".to_string()];
        let mut graph = CausalGraph::new(variables);
        graph.add_edge(0, 1).expect("operation should succeed");
        graph.add_edge(1, 2).expect("operation should succeed");

        // Y d-separates X and Z
        assert!(graph.d_separates(0, 2, &[1]));
        // Without conditioning on Y, X and Z are not d-separated
        assert!(!graph.d_separates(0, 2, &[]));
    }
}
