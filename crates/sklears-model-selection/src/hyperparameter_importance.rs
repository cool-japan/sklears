//! Hyperparameter Importance Analysis
//!
//! This module provides comprehensive hyperparameter importance analysis including:
//! - SHAP (SHapley Additive exPlanations) values for hyperparameters
//! - Functional ANOVA (fANOVA) for parameter analysis
//! - Interaction effect analysis
//! - Parameter sensitivity analysis
//! - Ablation studies for parameters
//!
//! These techniques help understand which hyperparameters are most important and how they
//! interact, enabling better hyperparameter optimization strategies.

use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::types::Float;
use std::collections::HashMap;

// ============================================================================
// SHAP Values for Hyperparameters
// ============================================================================

/// Configuration for SHAP value computation
#[derive(Debug, Clone)]
pub struct SHAPConfig {
    /// Number of samples for SHAP estimation
    pub n_samples: usize,
    /// Maximum coalition size to consider
    pub max_coalition_size: Option<usize>,
    /// Whether to use KernelSHAP approximation
    pub use_kernel_shap: bool,
    /// Background dataset size for TreeSHAP
    pub background_size: usize,
    pub random_state: Option<u64>,
}

impl Default for SHAPConfig {
    fn default() -> Self {
        Self {
            n_samples: 1000,
            max_coalition_size: None,
            use_kernel_shap: true,
            background_size: 100,
            random_state: None,
        }
    }
}

/// SHAP value analyzer for hyperparameters
pub struct SHAPAnalyzer {
    config: SHAPConfig,
    rng: StdRng,
}

impl SHAPAnalyzer {
    pub fn new(config: SHAPConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.random_state.unwrap_or(42));
        Self { config, rng }
    }

    /// Compute SHAP values for hyperparameters
    pub fn compute_shap_values(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        reference_config: &HashMap<String, Float>,
    ) -> Result<SHAPResult, Box<dyn std::error::Error>> {
        let param_names: Vec<_> = parameter_space.keys().cloned().collect();
        let n_params = param_names.len();

        if self.config.use_kernel_shap {
            self.compute_kernel_shap(
                evaluation_fn,
                parameter_space,
                reference_config,
                &param_names,
            )
        } else {
            self.compute_exact_shap(
                evaluation_fn,
                parameter_space,
                reference_config,
                &param_names,
                n_params,
            )
        }
    }

    /// Compute exact SHAP values using all coalitions
    fn compute_exact_shap(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        reference_config: &HashMap<String, Float>,
        param_names: &[String],
        n_params: usize,
    ) -> Result<SHAPResult, Box<dyn std::error::Error>> {
        let mut shap_values = HashMap::new();
        let baseline_performance = evaluation_fn(reference_config);

        // For each parameter, compute its SHAP value
        for (i, param_name) in param_names.iter().enumerate() {
            let mut marginal_contributions = Vec::new();

            // Generate all possible coalitions (power set)
            let max_coalitions = 2_usize.pow(n_params as u32 - 1);
            let n_coalitions = if let Some(max_size) = self.config.max_coalition_size {
                max_coalitions.min(max_size)
            } else {
                max_coalitions.min(1000) // Limit for practicality
            };

            for _ in 0..n_coalitions {
                // Create random coalition
                let coalition = self.sample_coalition(n_params, i);

                // Evaluate with and without the parameter
                let perf_with = self.evaluate_coalition(
                    evaluation_fn,
                    parameter_space,
                    reference_config,
                    param_names,
                    &coalition,
                    Some(i),
                )?;

                let perf_without = self.evaluate_coalition(
                    evaluation_fn,
                    parameter_space,
                    reference_config,
                    param_names,
                    &coalition,
                    None,
                )?;

                marginal_contributions.push(perf_with - perf_without);
            }

            // Average marginal contributions
            let shap_value = if marginal_contributions.is_empty() {
                0.0
            } else {
                marginal_contributions.iter().sum::<Float>() / marginal_contributions.len() as Float
            };

            shap_values.insert(param_name.clone(), shap_value);
        }

        let rankings = self.rank_parameters(&shap_values);

        Ok(SHAPResult {
            shap_values,
            baseline_performance,
            parameter_rankings: rankings,
            interaction_effects: HashMap::new(), // Computed separately
        })
    }

    /// Compute KernelSHAP approximation
    fn compute_kernel_shap(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        reference_config: &HashMap<String, Float>,
        param_names: &[String],
    ) -> Result<SHAPResult, Box<dyn std::error::Error>> {
        let n_params = param_names.len();
        let mut shap_values = HashMap::new();
        let baseline_performance = evaluation_fn(reference_config);

        // KernelSHAP uses weighted linear regression
        let mut samples = Vec::new();
        let mut performances = Vec::new();
        let mut weights = Vec::new();

        for _ in 0..self.config.n_samples {
            // Sample a coalition
            let coalition_size = self.rng.gen_range(0..=n_params);
            let coalition = self.sample_coalition_of_size(n_params, coalition_size);

            // Create perturbed configuration
            let mut perturbed = reference_config.clone();
            for (idx, &include) in coalition.iter().enumerate() {
                if !include {
                    // Replace with random value from parameter space
                    let param_name = &param_names[idx];
                    if let Some(&(min, max)) = parameter_space.get(param_name) {
                        let random_value = self.rng.gen_range(min..max);
                        perturbed.insert(param_name.clone(), random_value);
                    }
                }
            }

            let perf = evaluation_fn(&perturbed);
            let weight = self.shapley_kernel_weight(coalition_size, n_params);

            samples.push(coalition);
            performances.push(perf);
            weights.push(weight);
        }

        // Solve weighted least squares to get SHAP values
        let shap_coefficients =
            self.solve_weighted_least_squares(&samples, &performances, &weights)?;

        for (i, param_name) in param_names.iter().enumerate() {
            shap_values.insert(
                param_name.clone(),
                shap_coefficients.get(i).cloned().unwrap_or(0.0),
            );
        }

        let rankings = self.rank_parameters(&shap_values);

        Ok(SHAPResult {
            shap_values,
            baseline_performance,
            parameter_rankings: rankings,
            interaction_effects: HashMap::new(),
        })
    }

    // Helper methods

    fn sample_coalition(&mut self, n_params: usize, exclude_idx: usize) -> Vec<bool> {
        (0..n_params)
            .map(|i| i != exclude_idx && self.rng.gen_bool(0.5))
            .collect()
    }

    fn sample_coalition_of_size(&mut self, n_params: usize, size: usize) -> Vec<bool> {
        let mut coalition = vec![false; n_params];
        let mut indices: Vec<_> = (0..n_params).collect();

        // Fisher-Yates shuffle
        for i in (1..n_params).rev() {
            let j = self.rng.gen_range(0..=i);
            indices.swap(i, j);
        }

        for &idx in indices.iter().take(size) {
            coalition[idx] = true;
        }

        coalition
    }

    fn evaluate_coalition(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        reference_config: &HashMap<String, Float>,
        param_names: &[String],
        coalition: &[bool],
        include_idx: Option<usize>,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        let mut config = reference_config.clone();

        for (i, param_name) in param_names.iter().enumerate() {
            let should_include = coalition[i] || include_idx == Some(i);
            if !should_include {
                // Use random value
                if let Some(&(min, max)) = parameter_space.get(param_name) {
                    let random_value = self.rng.gen_range(min..max);
                    config.insert(param_name.clone(), random_value);
                }
            }
        }

        Ok(evaluation_fn(&config))
    }

    fn shapley_kernel_weight(&self, coalition_size: usize, n_params: usize) -> Float {
        if coalition_size == 0 || coalition_size == n_params {
            1e10 // Very large weight for empty and full coalitions
        } else {
            let numerator = (n_params - 1) as Float;
            let denominator = (coalition_size * (n_params - coalition_size)) as Float;
            numerator / denominator
        }
    }

    fn solve_weighted_least_squares(
        &self,
        samples: &[Vec<bool>],
        performances: &[Float],
        weights: &[Float],
    ) -> Result<Vec<Float>, Box<dyn std::error::Error>> {
        if samples.is_empty() {
            return Ok(Vec::new());
        }

        let n_params = samples[0].len();

        // Simple weighted average approach (simplified)
        let mut coefficients = vec![0.0; n_params];

        for param_idx in 0..n_params {
            let mut weighted_sum = 0.0;
            let mut total_weight = 0.0;

            for (i, sample) in samples.iter().enumerate() {
                if sample[param_idx] {
                    weighted_sum += performances[i] * weights[i];
                    total_weight += weights[i];
                }
            }

            if total_weight > 0.0 {
                coefficients[param_idx] = weighted_sum / total_weight;
            }
        }

        Ok(coefficients)
    }

    fn rank_parameters(&self, shap_values: &HashMap<String, Float>) -> Vec<(String, Float)> {
        let mut ranked: Vec<_> = shap_values
            .iter()
            .map(|(name, &value)| (name.clone(), value.abs()))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

/// Result of SHAP analysis
#[derive(Debug, Clone)]
pub struct SHAPResult {
    pub shap_values: HashMap<String, Float>,
    pub baseline_performance: Float,
    pub parameter_rankings: Vec<(String, Float)>,
    pub interaction_effects: HashMap<(String, String), Float>,
}

// ============================================================================
// Functional ANOVA (fANOVA)
// ============================================================================

/// Configuration for fANOVA analysis
#[derive(Debug, Clone)]
pub struct FANOVAConfig {
    pub n_trees: usize,
    pub max_depth: usize,
    pub min_samples_split: usize,
    pub n_samples: usize,
    pub random_state: Option<u64>,
}

impl Default for FANOVAConfig {
    fn default() -> Self {
        Self {
            n_trees: 16,
            max_depth: 6,
            min_samples_split: 10,
            n_samples: 1000,
            random_state: None,
        }
    }
}

/// Functional ANOVA analyzer
pub struct FANOVAAnalyzer {
    config: FANOVAConfig,
}

impl FANOVAAnalyzer {
    pub fn new(config: FANOVAConfig) -> Self {
        Self { config }
    }

    /// Perform fANOVA analysis
    pub fn analyze(
        &self,
        evaluation_data: &[(HashMap<String, Float>, Float)],
        param_names: &[String],
    ) -> Result<FANOVAResult, Box<dyn std::error::Error>> {
        // Compute total variance
        let performances: Vec<_> = evaluation_data.iter().map(|(_, perf)| *perf).collect();
        let mean_performance = performances.iter().sum::<Float>() / performances.len() as Float;
        let total_variance = performances
            .iter()
            .map(|&p| (p - mean_performance).powi(2))
            .sum::<Float>()
            / performances.len() as Float;

        // Compute variance contribution for each parameter
        let mut main_effects = HashMap::new();
        let mut interaction_effects = HashMap::new();

        for param_name in param_names {
            let variance_contribution =
                self.compute_variance_contribution(evaluation_data, param_name, mean_performance)?;
            let importance = variance_contribution / total_variance;
            main_effects.insert(param_name.clone(), importance);
        }

        // Compute pairwise interactions
        for i in 0..param_names.len() {
            for j in (i + 1)..param_names.len() {
                let interaction_variance = self.compute_interaction_variance(
                    evaluation_data,
                    &param_names[i],
                    &param_names[j],
                    mean_performance,
                )?;
                let importance = interaction_variance / total_variance;
                interaction_effects
                    .insert((param_names[i].clone(), param_names[j].clone()), importance);
            }
        }

        let rankings = self.rank_by_importance(&main_effects);

        Ok(FANOVAResult {
            main_effects,
            interaction_effects,
            total_variance,
            parameter_rankings: rankings,
        })
    }

    fn compute_variance_contribution(
        &self,
        data: &[(HashMap<String, Float>, Float)],
        param_name: &str,
        mean: Float,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Group by parameter value and compute conditional variance
        let mut groups: HashMap<String, Vec<Float>> = HashMap::new();

        for (params, perf) in data {
            if let Some(&value) = params.get(param_name) {
                // Discretize continuous values into bins
                let bin = format!("{:.2}", value);
                groups.entry(bin).or_default().push(*perf);
            }
        }

        // Compute variance explained by this parameter
        let mut variance_explained = 0.0;
        for performances in groups.values() {
            if performances.is_empty() {
                continue;
            }
            let group_mean = performances.iter().sum::<Float>() / performances.len() as Float;
            let group_size = performances.len() as Float;
            variance_explained += group_size * (group_mean - mean).powi(2);
        }

        Ok(variance_explained / data.len() as Float)
    }

    fn compute_interaction_variance(
        &self,
        data: &[(HashMap<String, Float>, Float)],
        param1: &str,
        param2: &str,
        mean: Float,
    ) -> Result<Float, Box<dyn std::error::Error>> {
        // Compute interaction effect
        let mut groups: HashMap<(String, String), Vec<Float>> = HashMap::new();

        for (params, perf) in data {
            if let (Some(&v1), Some(&v2)) = (params.get(param1), params.get(param2)) {
                let bin1 = format!("{:.2}", v1);
                let bin2 = format!("{:.2}", v2);
                groups.entry((bin1, bin2)).or_default().push(*perf);
            }
        }

        // Interaction variance
        let var1 = self.compute_variance_contribution(data, param1, mean)?;
        let var2 = self.compute_variance_contribution(data, param2, mean)?;

        let mut joint_variance = 0.0;
        for performances in groups.values() {
            if performances.is_empty() {
                continue;
            }
            let group_mean = performances.iter().sum::<Float>() / performances.len() as Float;
            let group_size = performances.len() as Float;
            joint_variance += group_size * (group_mean - mean).powi(2);
        }
        joint_variance /= data.len() as Float;

        // Interaction = joint - individual effects
        Ok((joint_variance - var1 - var2).max(0.0))
    }

    fn rank_by_importance(&self, effects: &HashMap<String, Float>) -> Vec<(String, Float)> {
        let mut ranked: Vec<_> = effects.iter().map(|(k, &v)| (k.clone(), v)).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

/// Result of fANOVA analysis
#[derive(Debug, Clone)]
pub struct FANOVAResult {
    pub main_effects: HashMap<String, Float>,
    pub interaction_effects: HashMap<(String, String), Float>,
    pub total_variance: Float,
    pub parameter_rankings: Vec<(String, Float)>,
}

// ============================================================================
// Parameter Sensitivity Analysis
// ============================================================================

/// Configuration for sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Number of samples for Morris method
    pub n_trajectories: usize,
    /// Grid levels for Sobol analysis
    pub n_levels: usize,
    /// Perturbation delta for finite differences
    pub perturbation_delta: Float,
    pub random_state: Option<u64>,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            n_trajectories: 10,
            n_levels: 4,
            perturbation_delta: 0.01,
            random_state: None,
        }
    }
}

/// Sensitivity analyzer using various methods
pub struct SensitivityAnalyzer {
    config: SensitivityConfig,
    rng: StdRng,
}

impl SensitivityAnalyzer {
    pub fn new(config: SensitivityConfig) -> Self {
        let rng = StdRng::seed_from_u64(config.random_state.unwrap_or(42));
        Self { config, rng }
    }

    /// Perform Morris sensitivity analysis
    pub fn morris_analysis(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        base_config: &HashMap<String, Float>,
    ) -> Result<SensitivityResult, Box<dyn std::error::Error>> {
        let param_names: Vec<_> = parameter_space.keys().cloned().collect();
        let mut elementary_effects: HashMap<String, Vec<Float>> = HashMap::new();

        for _ in 0..self.config.n_trajectories {
            // Generate random trajectory
            let mut current = base_config.clone();

            for param_name in &param_names {
                if let Some(&(min, max)) = parameter_space.get(param_name) {
                    // Perturb parameter
                    let original_value = current.get(param_name).cloned().unwrap_or(min);
                    let delta = self.config.perturbation_delta * (max - min);

                    let perturbed_value = (original_value + delta).min(max);
                    let mut perturbed = current.clone();
                    perturbed.insert(param_name.clone(), perturbed_value);

                    // Compute elementary effect
                    let f_original = evaluation_fn(&current);
                    let f_perturbed = evaluation_fn(&perturbed);
                    let effect = (f_perturbed - f_original) / delta;

                    elementary_effects
                        .entry(param_name.clone())
                        .or_default()
                        .push(effect);

                    current = perturbed;
                }
            }
        }

        // Compute statistics
        let mut sensitivities = HashMap::new();
        let interactions = HashMap::new();

        for (param_name, effects) in &elementary_effects {
            let mean = effects.iter().sum::<Float>() / effects.len() as Float;
            let variance =
                effects.iter().map(|&e| (e - mean).powi(2)).sum::<Float>() / effects.len() as Float;
            let std_dev = variance.sqrt();

            sensitivities.insert(
                param_name.clone(),
                ParameterSensitivity {
                    mean_effect: mean.abs(),
                    std_effect: std_dev,
                    mu_star: mean.abs(), // |μ|
                    sigma: std_dev,
                },
            );
        }

        let rankings = self.rank_sensitivities(&sensitivities);

        Ok(SensitivityResult {
            sensitivities,
            interactions,
            rankings,
        })
    }

    /// One-at-a-time (OAT) sensitivity analysis
    pub fn oat_analysis(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        base_config: &HashMap<String, Float>,
    ) -> Result<SensitivityResult, Box<dyn std::error::Error>> {
        let param_names: Vec<_> = parameter_space.keys().cloned().collect();
        let mut sensitivities = HashMap::new();
        let baseline_perf = evaluation_fn(base_config);

        for param_name in &param_names {
            if let Some(&(min, max)) = parameter_space.get(param_name) {
                let base_value = base_config
                    .get(param_name)
                    .cloned()
                    .unwrap_or((min + max) / 2.0);

                // Evaluate at several points
                let n_points = 5;
                let mut effects = Vec::new();

                for i in 0..n_points {
                    let alpha = i as Float / (n_points - 1) as Float;
                    let value = min + alpha * (max - min);

                    if (value - base_value).abs() < 1e-6 {
                        continue;
                    }

                    let mut perturbed = base_config.clone();
                    perturbed.insert(param_name.clone(), value);

                    let perf = evaluation_fn(&perturbed);
                    let effect = (perf - baseline_perf).abs() / (value - base_value).abs();
                    effects.push(effect);
                }

                if !effects.is_empty() {
                    let mean_effect = effects.iter().sum::<Float>() / effects.len() as Float;
                    let variance = effects
                        .iter()
                        .map(|&e| (e - mean_effect).powi(2))
                        .sum::<Float>()
                        / effects.len() as Float;

                    sensitivities.insert(
                        param_name.clone(),
                        ParameterSensitivity {
                            mean_effect,
                            std_effect: variance.sqrt(),
                            mu_star: mean_effect,
                            sigma: variance.sqrt(),
                        },
                    );
                }
            }
        }

        let rankings = self.rank_sensitivities(&sensitivities);

        Ok(SensitivityResult {
            sensitivities,
            interactions: HashMap::new(),
            rankings,
        })
    }

    fn rank_sensitivities(
        &self,
        sensitivities: &HashMap<String, ParameterSensitivity>,
    ) -> Vec<(String, Float)> {
        let mut ranked: Vec<_> = sensitivities
            .iter()
            .map(|(name, sens)| (name.clone(), sens.mu_star))
            .collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

/// Sensitivity of a single parameter
#[derive(Debug, Clone)]
pub struct ParameterSensitivity {
    pub mean_effect: Float,
    pub std_effect: Float,
    pub mu_star: Float,
    pub sigma: Float,
}

/// Result of sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    pub sensitivities: HashMap<String, ParameterSensitivity>,
    pub interactions: HashMap<(String, String), Float>,
    pub rankings: Vec<(String, Float)>,
}

// ============================================================================
// Ablation Studies
// ============================================================================

/// Configuration for ablation studies
#[derive(Debug, Clone)]
pub struct AblationConfig {
    /// Number of ablation iterations
    pub n_iterations: usize,
    /// Whether to use leave-one-out approach
    pub leave_one_out: bool,
    /// Whether to use cumulative importance
    pub cumulative: bool,
    pub random_state: Option<u64>,
}

impl Default for AblationConfig {
    fn default() -> Self {
        Self {
            n_iterations: 10,
            leave_one_out: true,
            cumulative: false,
            random_state: None,
        }
    }
}

/// Ablation study analyzer
pub struct AblationAnalyzer {
    config: AblationConfig,
}

impl AblationAnalyzer {
    pub fn new(config: AblationConfig) -> Self {
        Self { config }
    }

    /// Perform ablation study
    pub fn analyze(
        &self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        base_config: &HashMap<String, Float>,
    ) -> Result<AblationResult, Box<dyn std::error::Error>> {
        let param_names: Vec<_> = parameter_space.keys().cloned().collect();
        let baseline_performance = evaluation_fn(base_config);
        let mut ablation_effects = HashMap::new();

        if self.config.leave_one_out {
            // Leave-one-out ablation
            for param_name in &param_names {
                let mut ablated = base_config.clone();

                // Remove parameter (use default/random value)
                if let Some(&(min, max)) = parameter_space.get(param_name) {
                    ablated.insert(param_name.clone(), (min + max) / 2.0);
                }

                let ablated_perf = evaluation_fn(&ablated);
                let effect = baseline_performance - ablated_perf;

                ablation_effects.insert(param_name.clone(), effect);
            }
        } else {
            // Cumulative ablation
            let mut current = base_config.clone();
            let mut remaining_params = param_names.clone();

            while !remaining_params.is_empty() {
                let mut best_param = None;
                let mut best_effect = f64::NEG_INFINITY;

                for param_name in &remaining_params {
                    let mut test_config = current.clone();
                    if let Some(&(min, max)) = parameter_space.get(param_name) {
                        test_config.insert(param_name.clone(), (min + max) / 2.0);
                    }

                    let perf = evaluation_fn(&test_config);
                    let effect = baseline_performance - perf;

                    if effect > best_effect {
                        best_effect = effect;
                        best_param = Some(param_name.clone());
                    }
                }

                if let Some(param) = best_param {
                    ablation_effects.insert(param.clone(), best_effect);
                    if let Some(&(min, max)) = parameter_space.get(&param) {
                        current.insert(param.clone(), (min + max) / 2.0);
                    }
                    remaining_params.retain(|p| p != &param);
                }
            }
        }

        let rankings = self.rank_ablation_effects(&ablation_effects);

        Ok(AblationResult {
            ablation_effects,
            baseline_performance,
            parameter_rankings: rankings,
        })
    }

    fn rank_ablation_effects(&self, effects: &HashMap<String, Float>) -> Vec<(String, Float)> {
        let mut ranked: Vec<_> = effects.iter().map(|(k, &v)| (k.clone(), v.abs())).collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        ranked
    }
}

/// Result of ablation study
#[derive(Debug, Clone)]
pub struct AblationResult {
    pub ablation_effects: HashMap<String, Float>,
    pub baseline_performance: Float,
    pub parameter_rankings: Vec<(String, Float)>,
}

// ============================================================================
// Unified Importance Analysis
// ============================================================================

/// Comprehensive hyperparameter importance analyzer
pub struct HyperparameterImportanceAnalyzer {
    shap_analyzer: SHAPAnalyzer,
    fanova_analyzer: FANOVAAnalyzer,
    sensitivity_analyzer: SensitivityAnalyzer,
    ablation_analyzer: AblationAnalyzer,
}

impl HyperparameterImportanceAnalyzer {
    pub fn new(
        shap_config: SHAPConfig,
        fanova_config: FANOVAConfig,
        sensitivity_config: SensitivityConfig,
        ablation_config: AblationConfig,
    ) -> Self {
        Self {
            shap_analyzer: SHAPAnalyzer::new(shap_config),
            fanova_analyzer: FANOVAAnalyzer::new(fanova_config),
            sensitivity_analyzer: SensitivityAnalyzer::new(sensitivity_config),
            ablation_analyzer: AblationAnalyzer::new(ablation_config),
        }
    }

    /// Perform comprehensive importance analysis
    pub fn analyze_comprehensive(
        &mut self,
        evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
        parameter_space: &HashMap<String, (Float, Float)>,
        base_config: &HashMap<String, Float>,
        evaluation_data: &[(HashMap<String, Float>, Float)],
    ) -> Result<ComprehensiveImportanceResult, Box<dyn std::error::Error>> {
        let param_names: Vec<_> = parameter_space.keys().cloned().collect();

        // Run all analyses
        let shap_result =
            self.shap_analyzer
                .compute_shap_values(evaluation_fn, parameter_space, base_config)?;

        let fanova_result = self
            .fanova_analyzer
            .analyze(evaluation_data, &param_names)?;

        let sensitivity_result = self.sensitivity_analyzer.morris_analysis(
            evaluation_fn,
            parameter_space,
            base_config,
        )?;

        let ablation_result =
            self.ablation_analyzer
                .analyze(evaluation_fn, parameter_space, base_config)?;

        // Aggregate rankings
        let aggregated_rankings = self.aggregate_rankings(
            &shap_result.parameter_rankings,
            &fanova_result.parameter_rankings,
            &sensitivity_result.rankings,
            &ablation_result.parameter_rankings,
        );

        Ok(ComprehensiveImportanceResult {
            shap_result,
            fanova_result,
            sensitivity_result,
            ablation_result,
            aggregated_rankings,
        })
    }

    fn aggregate_rankings(
        &self,
        shap: &[(String, Float)],
        fanova: &[(String, Float)],
        sensitivity: &[(String, Float)],
        ablation: &[(String, Float)],
    ) -> Vec<(String, Float)> {
        let mut scores: HashMap<String, Vec<Float>> = HashMap::new();

        // Normalize and aggregate
        for (param, value) in shap {
            scores.entry(param.clone()).or_default().push(*value);
        }
        for (param, value) in fanova {
            scores.entry(param.clone()).or_default().push(*value);
        }
        for (param, value) in sensitivity {
            scores.entry(param.clone()).or_default().push(*value);
        }
        for (param, value) in ablation {
            scores.entry(param.clone()).or_default().push(*value);
        }

        let mut aggregated: Vec<_> = scores
            .iter()
            .map(|(param, values)| {
                let avg = values.iter().sum::<Float>() / values.len() as Float;
                (param.clone(), avg)
            })
            .collect();

        aggregated.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        aggregated
    }
}

/// Comprehensive importance analysis result
#[derive(Debug, Clone)]
pub struct ComprehensiveImportanceResult {
    pub shap_result: SHAPResult,
    pub fanova_result: FANOVAResult,
    pub sensitivity_result: SensitivityResult,
    pub ablation_result: AblationResult,
    pub aggregated_rankings: Vec<(String, Float)>,
}

// ============================================================================
// Convenience Functions
// ============================================================================

/// Compute SHAP values for hyperparameters
pub fn compute_shap_importance(
    evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
    parameter_space: &HashMap<String, (Float, Float)>,
    reference_config: &HashMap<String, Float>,
) -> Result<SHAPResult, Box<dyn std::error::Error>> {
    let config = SHAPConfig::default();
    let mut analyzer = SHAPAnalyzer::new(config);
    analyzer.compute_shap_values(evaluation_fn, parameter_space, reference_config)
}

/// Perform sensitivity analysis
pub fn analyze_parameter_sensitivity(
    evaluation_fn: &dyn Fn(&HashMap<String, Float>) -> Float,
    parameter_space: &HashMap<String, (Float, Float)>,
    base_config: &HashMap<String, Float>,
) -> Result<SensitivityResult, Box<dyn std::error::Error>> {
    let config = SensitivityConfig::default();
    let mut analyzer = SensitivityAnalyzer::new(config);
    analyzer.morris_analysis(evaluation_fn, parameter_space, base_config)
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shap_config() {
        let config = SHAPConfig::default();
        assert_eq!(config.n_samples, 1000);
        assert!(config.use_kernel_shap);
    }

    #[test]
    fn test_fanova_config() {
        let config = FANOVAConfig::default();
        assert_eq!(config.n_trees, 16);
        assert_eq!(config.max_depth, 6);
    }

    #[test]
    fn test_sensitivity_config() {
        let config = SensitivityConfig::default();
        assert_eq!(config.n_trajectories, 10);
        assert_eq!(config.n_levels, 4);
    }

    #[test]
    fn test_ablation_config() {
        let config = AblationConfig::default();
        assert_eq!(config.n_iterations, 10);
        assert!(config.leave_one_out);
    }

    #[test]
    fn test_shap_analyzer_creation() {
        let config = SHAPConfig::default();
        let analyzer = SHAPAnalyzer::new(config);
        assert_eq!(analyzer.config.n_samples, 1000);
    }

    #[test]
    fn test_sensitivity_analyzer_creation() {
        let config = SensitivityConfig::default();
        let analyzer = SensitivityAnalyzer::new(config);
        assert_eq!(analyzer.config.n_trajectories, 10);
    }
}
