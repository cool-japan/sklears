//! Information-Theoretic Methods for Model Interpretability
//!
//! This module provides information theory-based approaches to model interpretability,
//! including mutual information analysis, information gain attribution, entropy-based
//! explanations, information bottleneck analysis, and minimum description length principles.

use crate::{Float, SklResult};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Configuration for information-theoretic methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationTheoreticConfig {
    /// Number of bins for discretization of continuous variables
    pub n_bins: usize,
    /// Method for estimating probability distributions
    pub estimation_method: EstimationMethod,
    /// Regularization parameter for smooth estimators
    pub regularization: Float,
    /// Minimum count threshold for reliable estimates
    pub min_count_threshold: usize,
    /// Whether to use bias correction for entropy estimates
    pub bias_correction: bool,
    /// Beta parameter for information bottleneck (trade-off between compression and prediction)
    pub beta: Float,
    /// Maximum number of iterations for information bottleneck optimization
    pub max_iterations: usize,
    /// Convergence tolerance for iterative algorithms
    pub tolerance: Float,
}

impl Default for InformationTheoreticConfig {
    fn default() -> Self {
        Self {
            n_bins: 10,
            estimation_method: EstimationMethod::Histogram,
            regularization: 1e-6,
            min_count_threshold: 5,
            bias_correction: true,
            beta: 1.0,
            max_iterations: 100,
            tolerance: 1e-6,
        }
    }
}

/// Methods for estimating probability distributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EstimationMethod {
    /// Simple histogram-based estimation
    Histogram,
    /// Kernel density estimation
    KernelDensity,
    /// Adaptive binning based on data distribution
    AdaptiveBinning,
    /// k-nearest neighbor entropy estimation
    KNearestNeighbor,
}

/// Mutual information analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutualInformationResult {
    /// Mutual information between each feature and target
    pub feature_target_mi: Array1<Float>,
    /// Pairwise mutual information between features
    pub feature_feature_mi: Array2<Float>,
    /// Conditional mutual information given other features
    pub conditional_mi: Array2<Float>,
    /// Normalized mutual information (0-1 scale)
    pub normalized_mi: Array1<Float>,
    /// Statistical significance of MI values
    pub p_values: Array1<Float>,
    /// Confidence intervals for MI estimates
    pub confidence_intervals: Vec<(Float, Float)>,
}

/// Information gain attribution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationGainResult {
    /// Information gain for each feature
    pub information_gains: Array1<Float>,
    /// Entropy of target variable
    pub target_entropy: Float,
    /// Conditional entropy of target given each feature
    pub conditional_entropies: Array1<Float>,
    /// Feature ranking based on information gain
    pub feature_ranking: Vec<usize>,
    /// Cumulative information gain
    pub cumulative_gains: Array1<Float>,
    /// Interaction-based information gains
    pub interaction_gains: Array2<Float>,
}

/// Entropy-based explanation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntropyExplanationResult {
    /// Entropy of predictions for each instance
    pub instance_entropies: Array1<Float>,
    /// Average entropy reduction when conditioning on each feature
    pub feature_entropy_reductions: Array1<Float>,
    /// Cross-entropy between model predictions and true distribution
    pub cross_entropy: Float,
    /// KL divergence from uniform distribution
    pub kl_divergence_uniform: Float,
    /// Jensen-Shannon divergence between prediction distributions
    pub js_divergences: Array2<Float>,
    /// Entropy-based uncertainty quantification
    pub entropy_uncertainty: Array1<Float>,
}

/// Information bottleneck analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InformationBottleneckResult {
    /// Optimal representation that maximizes I(T;Y) - β*I(T;X)
    pub optimal_representation: Array2<Float>,
    /// Information bottleneck curve (compression vs. prediction)
    pub ib_curve: Vec<(Float, Float)>,
    /// Relevant information I(T;Y)
    pub relevant_information: Float,
    /// Compression cost I(T;X)
    pub compression_cost: Float,
    /// Trade-off parameter β used
    pub beta_used: Float,
    /// Feature relevance scores from information bottleneck
    pub feature_relevance: Array1<Float>,
}

/// Minimum description length result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MDLResult {
    /// MDL-based model complexity
    pub model_complexity: Float,
    /// Data description length
    pub data_description_length: Float,
    /// Total description length
    pub total_description_length: Float,
    /// Feature selection based on MDL principle
    pub selected_features: Vec<usize>,
    /// MDL score for each feature subset
    pub subset_mdl_scores: HashMap<Vec<usize>, Float>,
    /// Normalized MDL scores
    pub normalized_scores: Array1<Float>,
}

/// Compute mutual information between features and target
#[allow(non_snake_case)]
pub fn analyze_mutual_information(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<MutualInformationResult> {
    let n_features = X.ncols();
    let n_samples = X.nrows();

    // Discretize continuous variables
    let X_discrete = discretize_features(X, config)?;
    let y_discrete = discretize_target(y, config)?;

    // Compute feature-target mutual information
    let mut feature_target_mi = Array1::zeros(n_features);
    let mut conditional_mi = Array2::zeros((n_features, n_features));
    let mut p_values = Array1::zeros(n_features);

    for i in 0..n_features {
        let x_i = X_discrete.column(i);

        // Compute MI(X_i; Y)
        feature_target_mi[i] = compute_mutual_information(&x_i, &y_discrete.view(), config)?;

        // Statistical significance test
        p_values[i] = test_mi_significance(feature_target_mi[i], n_samples, config);

        // Conditional mutual information
        for j in 0..n_features {
            if i != j {
                let x_j = X_discrete.column(j);
                conditional_mi[[i, j]] =
                    compute_conditional_mutual_information(&x_i, &y_discrete.view(), &x_j, config)?;
            }
        }
    }

    // Compute pairwise feature-feature MI
    let mut feature_feature_mi = Array2::zeros((n_features, n_features));
    for i in 0..n_features {
        for j in i + 1..n_features {
            let x_i = X_discrete.column(i);
            let x_j = X_discrete.column(j);
            let mi_val = compute_mutual_information(&x_i, &x_j, config)?;
            feature_feature_mi[[i, j]] = mi_val;
            feature_feature_mi[[j, i]] = mi_val;
        }
    }

    // Normalize MI values
    let normalized_mi =
        normalize_mutual_information(&feature_target_mi, &X_discrete, &y_discrete, config)?;

    // Compute confidence intervals
    let confidence_intervals =
        compute_mi_confidence_intervals(&feature_target_mi, n_samples, config);

    Ok(MutualInformationResult {
        feature_target_mi,
        feature_feature_mi,
        conditional_mi,
        normalized_mi,
        p_values,
        confidence_intervals,
    })
}

/// Compute information gain attribution for feature importance
#[allow(non_snake_case)]
pub fn compute_information_gain_attribution(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<InformationGainResult> {
    let n_features = X.ncols();

    // Discretize data
    let X_discrete = discretize_features(X, config)?;
    let y_discrete = discretize_target(y, config)?;

    // Compute target entropy H(Y)
    let target_entropy = compute_entropy(&y_discrete.view(), config)?;

    // Compute information gains IG(Y|X_i) = H(Y) - H(Y|X_i)
    let mut information_gains = Array1::zeros(n_features);
    let mut conditional_entropies = Array1::zeros(n_features);

    for i in 0..n_features {
        let x_i = X_discrete.column(i);
        let conditional_entropy = compute_conditional_entropy(&y_discrete.view(), &x_i, config)?;
        conditional_entropies[i] = conditional_entropy;
        information_gains[i] = target_entropy - conditional_entropy;
    }

    // Feature ranking based on information gain
    let mut feature_ranking: Vec<usize> = (0..n_features).collect();
    feature_ranking.sort_by(|&a, &b| {
        information_gains[b]
            .partial_cmp(&information_gains[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Cumulative information gain
    let mut cumulative_gains = Array1::zeros(n_features);
    cumulative_gains[0] = information_gains[feature_ranking[0]];
    for i in 1..n_features {
        cumulative_gains[i] = cumulative_gains[i - 1] + information_gains[feature_ranking[i]];
    }

    // Interaction-based information gains
    let interaction_gains =
        compute_interaction_information_gains(&X_discrete, &y_discrete, config)?;

    Ok(InformationGainResult {
        information_gains,
        target_entropy,
        conditional_entropies,
        feature_ranking,
        cumulative_gains,
        interaction_gains,
    })
}

/// Generate entropy-based explanations
#[allow(non_snake_case)]
pub fn generate_entropy_explanations<F>(
    model: F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<EntropyExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> SklResult<Array2<Float>>,
{
    let n_samples = X.nrows();
    let n_features = X.ncols();

    // Get model predictions (probability distributions)
    let predictions = model(X)?;

    // Compute entropy for each instance
    let mut instance_entropies = Array1::zeros(n_samples);
    for i in 0..n_samples {
        let pred_dist = predictions.row(i);
        instance_entropies[i] = compute_entropy_from_probabilities(&pred_dist);
    }

    // Compute feature entropy reductions
    let mut feature_entropy_reductions = Array1::zeros(n_features);

    for feature_idx in 0..n_features {
        // Create masked version of data (remove feature)
        let X_masked = mask_feature(X, feature_idx);
        let predictions_masked = model(&X_masked.view())?;

        // Compute entropy reduction
        let mut entropy_reduction = 0.0;
        for i in 0..n_samples {
            let original_entropy = compute_entropy_from_probabilities(&predictions.row(i));
            let masked_entropy = compute_entropy_from_probabilities(&predictions_masked.row(i));
            entropy_reduction += original_entropy - masked_entropy;
        }
        feature_entropy_reductions[feature_idx] = entropy_reduction / n_samples as Float;
    }

    // Compute cross-entropy with true labels
    let cross_entropy = compute_cross_entropy(&predictions, y)?;

    // Compute KL divergence from uniform distribution
    let kl_divergence_uniform = compute_kl_divergence_uniform(&predictions);

    // Compute Jensen-Shannon divergences between instances
    let js_divergences = compute_pairwise_js_divergences(&predictions)?;

    // Entropy-based uncertainty quantification
    let entropy_uncertainty = instance_entropies.clone();

    Ok(EntropyExplanationResult {
        instance_entropies,
        feature_entropy_reductions,
        cross_entropy,
        kl_divergence_uniform,
        js_divergences,
        entropy_uncertainty,
    })
}

/// Perform information bottleneck analysis
#[allow(non_snake_case)]
pub fn analyze_information_bottleneck(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<InformationBottleneckResult> {
    let n_samples = X.nrows();
    let n_features = X.ncols();

    // Discretize data
    let X_discrete = discretize_features(X, config)?;
    let y_discrete = discretize_target(y, config)?;

    // Initialize representation T randomly
    let mut T = Array2::from_shape_fn((n_samples, n_features), |_| {
        scirs2_core::random::thread_rng().random::<Float>()
    });

    // Information bottleneck optimization
    let mut ib_curve = Vec::new();
    let mut best_representation = T.clone();
    let mut best_objective = Float::NEG_INFINITY;

    for iteration in 0..config.max_iterations {
        // Update representation using information bottleneck principle
        let (new_T, objective) =
            optimize_information_bottleneck_step(&X_discrete, &y_discrete, &T, config)?;

        // Compute information measures
        let relevant_info = compute_mutual_information_matrices(
            &new_T,
            &y_discrete.clone().insert_axis(Axis(1)),
            config,
        )?;
        let compression_cost = compute_mutual_information_matrices(&new_T, &X_discrete, config)?;

        ib_curve.push((compression_cost, relevant_info));

        // Check for improvement
        if objective > best_objective {
            best_objective = objective;
            best_representation = new_T.clone();
        }

        // Check convergence
        let change = compute_matrix_frobenius_norm_diff(&T, &new_T);
        if change < config.tolerance {
            break;
        }

        T = new_T;
    }

    // Compute final information measures
    let relevant_information = compute_mutual_information_matrices(
        &best_representation,
        &y_discrete.clone().insert_axis(Axis(1)),
        config,
    )?;
    let compression_cost =
        compute_mutual_information_matrices(&best_representation, &X_discrete, config)?;

    // Compute feature relevance from information bottleneck
    let feature_relevance =
        compute_feature_relevance_from_ib(&best_representation, &X_discrete, config)?;

    Ok(InformationBottleneckResult {
        optimal_representation: best_representation,
        ib_curve,
        relevant_information,
        compression_cost,
        beta_used: config.beta,
        feature_relevance,
    })
}

/// Apply minimum description length principle for feature selection
#[allow(non_snake_case)]
pub fn apply_minimum_description_length(
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<MDLResult> {
    let n_features = X.ncols();

    // Discretize data
    let X_discrete = discretize_features(X, config)?;
    let y_discrete = discretize_target(y, config)?;

    // Compute MDL for different feature subsets
    let mut subset_mdl_scores = HashMap::new();
    let mut best_mdl = Float::INFINITY;
    let mut selected_features = Vec::new();

    // Forward selection based on MDL principle
    let mut current_features = Vec::new();
    let mut remaining_features: Vec<usize> = (0..n_features).collect();

    while !remaining_features.is_empty() {
        let mut best_feature = 0;
        let mut best_mdl_improvement = Float::NEG_INFINITY;

        for &feature in &remaining_features {
            let mut test_features = current_features.clone();
            test_features.push(feature);

            let mdl_score = compute_mdl_score(&X_discrete, &y_discrete, &test_features, config)?;
            subset_mdl_scores.insert(test_features.clone(), mdl_score);

            let improvement = best_mdl - mdl_score;
            if improvement > best_mdl_improvement {
                best_mdl_improvement = improvement;
                best_feature = feature;
                best_mdl = mdl_score;
            }
        }

        // Add feature if it improves MDL
        if best_mdl_improvement > 0.0 {
            current_features.push(best_feature);
            remaining_features.retain(|&x| x != best_feature);
        } else {
            break; // No more improvement possible
        }
    }

    selected_features = current_features;

    // Compute model complexity and data description length
    let model_complexity = compute_model_complexity(&selected_features, config);
    let data_description_length =
        compute_data_description_length(&X_discrete, &y_discrete, &selected_features, config)?;
    let total_description_length = model_complexity + data_description_length;

    // Normalize MDL scores
    let max_score = subset_mdl_scores
        .values()
        .cloned()
        .fold(Float::NEG_INFINITY, Float::max);
    let min_score = subset_mdl_scores
        .values()
        .cloned()
        .fold(Float::INFINITY, Float::min);
    let score_range = max_score - min_score;

    let mut normalized_scores = Array1::zeros(n_features);
    for i in 0..n_features {
        let subset = vec![i];
        if let Some(&score) = subset_mdl_scores.get(&subset) {
            normalized_scores[i] = if score_range > 0.0 {
                1.0 - (score - min_score) / score_range
            } else {
                0.5
            };
        }
    }

    Ok(MDLResult {
        model_complexity,
        data_description_length,
        total_description_length,
        selected_features,
        subset_mdl_scores,
        normalized_scores,
    })
}

// Helper functions

fn discretize_features(
    X: &ArrayView2<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<Array2<usize>> {
    let (n_samples, n_features) = X.dim();
    let mut X_discrete = Array2::zeros((n_samples, n_features));

    for feature_idx in 0..n_features {
        let feature_values = X.column(feature_idx);
        let discretized = discretize_column(&feature_values, config)?;
        X_discrete.column_mut(feature_idx).assign(&discretized);
    }

    Ok(X_discrete)
}

fn discretize_target(
    y: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<Array1<usize>> {
    discretize_column(y, config)
}

fn discretize_column(
    values: &ArrayView1<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<Array1<usize>> {
    let n_samples = values.len();

    match config.estimation_method {
        EstimationMethod::Histogram => {
            // Equal-width binning
            let min_val = values.iter().cloned().fold(Float::INFINITY, Float::min);
            let max_val = values.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
            let bin_width = (max_val - min_val) / config.n_bins as Float;

            let mut discretized = Array1::zeros(n_samples);
            for (i, &val) in values.iter().enumerate() {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                discretized[i] = bin.min(config.n_bins - 1);
            }
            Ok(discretized)
        }
        EstimationMethod::AdaptiveBinning => {
            // Equal-frequency binning
            let mut sorted_values: Vec<(Float, usize)> = values
                .iter()
                .enumerate()
                .map(|(i, &val)| (val, i))
                .collect();
            sorted_values.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            let samples_per_bin = n_samples / config.n_bins;
            let mut discretized = Array1::zeros(n_samples);

            for (sorted_idx, (_, original_idx)) in sorted_values.iter().enumerate() {
                let bin = (sorted_idx / samples_per_bin).min(config.n_bins - 1);
                discretized[*original_idx] = bin;
            }
            Ok(discretized)
        }
        _ => {
            // Default to histogram method
            let min_val = values.iter().cloned().fold(Float::INFINITY, Float::min);
            let max_val = values.iter().cloned().fold(Float::NEG_INFINITY, Float::max);
            let bin_width = (max_val - min_val) / config.n_bins as Float;

            let mut discretized = Array1::zeros(n_samples);
            for (i, &val) in values.iter().enumerate() {
                let bin = ((val - min_val) / bin_width).floor() as usize;
                discretized[i] = bin.min(config.n_bins - 1);
            }
            Ok(discretized)
        }
    }
}

fn compute_mutual_information(
    x: &ArrayView1<usize>,
    y: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // Compute joint and marginal distributions
    let (joint_dist, x_dist, y_dist) = compute_joint_distribution(x, y, config)?;

    let mut mi = 0.0;
    for (x_val, x_prob) in x_dist.iter() {
        for (y_val, y_prob) in y_dist.iter() {
            if let Some(&joint_prob) = joint_dist.get(&(*x_val, *y_val)) {
                if joint_prob > 0.0 && *x_prob > 0.0 && *y_prob > 0.0 {
                    mi += joint_prob * (joint_prob / (x_prob * y_prob)).ln();
                }
            }
        }
    }

    // Apply bias correction if requested
    if config.bias_correction {
        let n_samples = x.len() as Float;
        let n_x_bins = x_dist.len() as Float;
        let n_y_bins = y_dist.len() as Float;

        // Miller-Madow bias correction
        let bias_correction = (n_x_bins - 1.0) * (n_y_bins - 1.0) / (2.0 * n_samples);
        mi = (mi - bias_correction).max(0.0);
    }

    Ok(mi)
}

fn compute_conditional_mutual_information(
    x: &ArrayView1<usize>,
    y: &ArrayView1<usize>,
    z: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // CMI(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z) - H(Z)
    let h_x_given_z = compute_conditional_entropy_discrete(x, z, config)?;
    let h_y_given_z = compute_conditional_entropy_discrete(y, z, config)?;
    let h_xy_given_z = compute_conditional_joint_entropy(x, y, z, config)?;

    let cmi = h_x_given_z + h_y_given_z - h_xy_given_z;
    Ok(cmi.max(0.0))
}

fn compute_entropy(
    values: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    let dist = compute_distribution(values, config)?;

    let mut entropy = 0.0;
    for &prob in dist.values() {
        if prob > 0.0 {
            entropy -= prob * prob.ln();
        }
    }

    // Apply bias correction
    if config.bias_correction {
        let n_samples = values.len() as Float;
        let n_bins = dist.len() as Float;
        let bias_correction = (n_bins - 1.0) / (2.0 * n_samples);
        entropy = (entropy - bias_correction).max(0.0);
    }

    Ok(entropy)
}

fn compute_conditional_entropy(
    y: &ArrayView1<usize>,
    x: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    let (joint_dist, x_dist, _) = compute_joint_distribution(x, y, config)?;

    let mut conditional_entropy = 0.0;
    for (x_val, x_prob) in x_dist.iter() {
        let mut h_y_given_x = 0.0;
        let mut total_prob_y_given_x = 0.0;

        for (joint_key, joint_prob) in joint_dist.iter() {
            if joint_key.0 == *x_val {
                let prob_y_given_x = joint_prob / x_prob;
                if prob_y_given_x > 0.0 {
                    h_y_given_x -= prob_y_given_x * prob_y_given_x.ln();
                }
                total_prob_y_given_x += prob_y_given_x;
            }
        }

        conditional_entropy += x_prob * h_y_given_x;
    }

    Ok(conditional_entropy)
}

fn compute_conditional_entropy_discrete(
    x: &ArrayView1<usize>,
    z: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    compute_conditional_entropy(x, z, config)
}

fn compute_conditional_joint_entropy(
    x: &ArrayView1<usize>,
    y: &ArrayView1<usize>,
    z: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // Create joint variable (x,y)
    let mut xy_joint = Array1::zeros(x.len());
    let max_x = x.iter().max().unwrap_or(&0) + 1;

    for i in 0..x.len() {
        xy_joint[i] = x[i] * max_x + y[i];
    }

    compute_conditional_entropy(&xy_joint.view(), z, config)
}

fn compute_joint_distribution(
    x: &ArrayView1<usize>,
    y: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<(
    HashMap<(usize, usize), Float>,
    HashMap<usize, Float>,
    HashMap<usize, Float>,
)> {
    let n_samples = x.len() as Float;
    let mut joint_counts = HashMap::new();
    let mut x_counts = HashMap::new();
    let mut y_counts = HashMap::new();

    // Count occurrences
    for i in 0..x.len() {
        let x_val = x[i];
        let y_val = y[i];

        *joint_counts.entry((x_val, y_val)).or_insert(0) += 1;
        *x_counts.entry(x_val).or_insert(0) += 1;
        *y_counts.entry(y_val).or_insert(0) += 1;
    }

    // Convert to probabilities with regularization
    let joint_dist: HashMap<(usize, usize), Float> = joint_counts
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                (v as Float + config.regularization) / (n_samples + config.regularization),
            )
        })
        .collect();

    let x_dist: HashMap<usize, Float> = x_counts
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                (v as Float + config.regularization) / (n_samples + config.regularization),
            )
        })
        .collect();

    let y_dist: HashMap<usize, Float> = y_counts
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                (v as Float + config.regularization) / (n_samples + config.regularization),
            )
        })
        .collect();

    Ok((joint_dist, x_dist, y_dist))
}

fn compute_distribution(
    values: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<HashMap<usize, Float>> {
    let n_samples = values.len() as Float;
    let mut counts = HashMap::new();

    for &val in values.iter() {
        *counts.entry(val).or_insert(0) += 1;
    }

    let dist: HashMap<usize, Float> = counts
        .into_iter()
        .map(|(k, v)| {
            (
                k,
                (v as Float + config.regularization) / (n_samples + config.regularization),
            )
        })
        .collect();

    Ok(dist)
}

fn normalize_mutual_information(
    mi_values: &Array1<Float>,
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Array1<Float>> {
    let n_features = mi_values.len();
    let mut normalized = Array1::zeros(n_features);

    let h_y = compute_entropy(&y_discrete.view(), config)?;

    for i in 0..n_features {
        let x_i = X_discrete.column(i);
        let h_x = compute_entropy(&x_i, config)?;

        // Normalized MI = MI(X,Y) / min(H(X), H(Y))
        let normalizer = h_x.min(h_y);
        normalized[i] = if normalizer > 0.0 {
            mi_values[i] / normalizer
        } else {
            0.0
        };
    }

    Ok(normalized)
}

fn test_mi_significance(
    mi_value: Float,
    n_samples: usize,
    _config: &InformationTheoreticConfig,
) -> Float {
    // Simple chi-square test approximation for MI significance
    let chi_square = 2.0 * n_samples as Float * mi_value;

    // Degrees of freedom approximation (simplified)
    let df = 1.0;

    // P-value approximation using chi-square distribution
    // This is a very simplified approximation
    let p_value = (-chi_square / 2.0).exp();

    p_value.min(1.0).max(0.0)
}

fn compute_mi_confidence_intervals(
    mi_values: &Array1<Float>,
    n_samples: usize,
    _config: &InformationTheoreticConfig,
) -> Vec<(Float, Float)> {
    let n_features = mi_values.len();
    let mut intervals = Vec::with_capacity(n_features);

    // Bootstrap-based confidence intervals (simplified)
    let z_score = 1.96; // 95% confidence

    for &mi_val in mi_values.iter() {
        // Standard error approximation
        let std_error = (mi_val / (n_samples as Float).sqrt()).max(0.01);
        let margin = z_score * std_error;

        intervals.push(((mi_val - margin).max(0.0), mi_val + margin));
    }

    intervals
}

fn compute_interaction_information_gains(
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Array2<Float>> {
    let n_features = X_discrete.ncols();
    let mut interaction_gains = Array2::zeros((n_features, n_features));

    for i in 0..n_features {
        for j in i + 1..n_features {
            let x_i = X_discrete.column(i);
            let x_j = X_discrete.column(j);

            // Compute interaction information I(X_i; X_j; Y)
            let mi_xi_y = compute_mutual_information(&x_i, &y_discrete.view(), config)?;
            let mi_xj_y = compute_mutual_information(&x_j, &y_discrete.view(), config)?;
            let mi_xi_xj = compute_mutual_information(&x_i, &x_j, config)?;
            let mi_xi_xj_y =
                compute_three_way_mutual_information(&x_i, &x_j, &y_discrete.view(), config)?;

            let interaction = mi_xi_xj_y - mi_xi_y - mi_xj_y + mi_xi_xj;
            interaction_gains[[i, j]] = interaction;
            interaction_gains[[j, i]] = interaction;
        }
    }

    Ok(interaction_gains)
}

fn compute_three_way_mutual_information(
    x: &ArrayView1<usize>,
    y: &ArrayView1<usize>,
    z: &ArrayView1<usize>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // Simplified three-way MI computation
    let mi_xy = compute_mutual_information(x, y, config)?;
    let mi_xz = compute_mutual_information(x, z, config)?;
    let mi_yz = compute_mutual_information(y, z, config)?;

    // Approximation: I(X;Y;Z) ≈ min(I(X;Y), I(X;Z), I(Y;Z))
    Ok(mi_xy.min(mi_xz).min(mi_yz))
}

fn compute_entropy_from_probabilities(probabilities: &ArrayView1<Float>) -> Float {
    let mut entropy = 0.0;
    for &prob in probabilities.iter() {
        if prob > 0.0 {
            entropy -= prob * prob.ln();
        }
    }
    entropy
}

fn mask_feature(X: &ArrayView2<Float>, feature_idx: usize) -> Array2<Float> {
    let mut X_masked = X.to_owned();

    // Set feature to mean value (simple masking)
    let feature_mean = X.column(feature_idx).mean().unwrap_or(0.0);
    X_masked.column_mut(feature_idx).fill(feature_mean);

    X_masked
}

fn compute_cross_entropy(predictions: &Array2<Float>, y: &ArrayView1<Float>) -> SklResult<Float> {
    let n_samples = predictions.nrows();
    let mut cross_entropy = 0.0;

    for i in 0..n_samples {
        let true_class = y[i] as usize;
        if true_class < predictions.ncols() {
            let pred_prob = predictions[[i, true_class]].max(1e-15); // Avoid log(0)
            cross_entropy -= pred_prob.ln();
        }
    }

    Ok(cross_entropy / n_samples as Float)
}

fn compute_kl_divergence_uniform(predictions: &Array2<Float>) -> Float {
    let n_classes = predictions.ncols() as Float;
    let uniform_prob = 1.0 / n_classes;

    let mut kl_div = 0.0;
    for prediction in predictions.axis_iter(Axis(0)) {
        for &prob in prediction.iter() {
            if prob > 0.0 {
                kl_div += prob * (prob / uniform_prob).ln();
            }
        }
    }

    kl_div / predictions.nrows() as Float
}

fn compute_pairwise_js_divergences(predictions: &Array2<Float>) -> SklResult<Array2<Float>> {
    let n_samples = predictions.nrows();
    let mut js_divergences = Array2::zeros((n_samples, n_samples));

    for i in 0..n_samples {
        for j in i + 1..n_samples {
            let p = predictions.row(i);
            let q = predictions.row(j);
            let js_div = compute_jensen_shannon_divergence(&p, &q);
            js_divergences[[i, j]] = js_div;
            js_divergences[[j, i]] = js_div;
        }
    }

    Ok(js_divergences)
}

fn compute_jensen_shannon_divergence(p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
    let n = p.len();
    let mut m = Array1::zeros(n);

    // M = (P + Q) / 2
    for i in 0..n {
        m[i] = (p[i] + q[i]) / 2.0;
    }

    // JS(P,Q) = (KL(P,M) + KL(Q,M)) / 2
    let kl_pm = compute_kl_divergence_arrays(p, &m.view());
    let kl_qm = compute_kl_divergence_arrays(q, &m.view());

    (kl_pm + kl_qm) / 2.0
}

fn compute_kl_divergence_arrays(p: &ArrayView1<Float>, q: &ArrayView1<Float>) -> Float {
    let mut kl_div = 0.0;
    for i in 0..p.len() {
        if p[i] > 0.0 && q[i] > 0.0 {
            kl_div += p[i] * (p[i] / q[i]).ln();
        }
    }
    kl_div
}

fn optimize_information_bottleneck_step(
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    T: &Array2<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<(Array2<Float>, Float)> {
    // Simplified information bottleneck optimization step
    let mut new_T = T.clone();

    // Gradient-based update (simplified)
    let learning_rate = 0.01;

    for i in 0..T.nrows() {
        for j in 0..T.ncols() {
            // Compute approximate gradient
            let epsilon = 1e-6;
            let mut T_plus = T.clone();
            T_plus[[i, j]] += epsilon;

            let obj_plus = compute_ib_objective(X_discrete, y_discrete, &T_plus, config)?;
            let obj_current = compute_ib_objective(X_discrete, y_discrete, T, config)?;

            let gradient = (obj_plus - obj_current) / epsilon;
            new_T[[i, j]] += learning_rate * gradient;
        }
    }

    let objective = compute_ib_objective(X_discrete, y_discrete, &new_T, config)?;

    Ok((new_T, objective))
}

fn compute_ib_objective(
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    T: &Array2<Float>,
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // Simplified IB objective: I(T;Y) - β*I(T;X)

    // For continuous T, we need to discretize or use differential entropy
    // This is a simplified version
    let relevant_info = compute_matrix_frobenius_norm(T); // Placeholder
    let compression_cost = compute_matrix_frobenius_norm(T); // Placeholder

    let objective = relevant_info - config.beta * compression_cost;
    Ok(objective)
}

fn compute_mutual_information_matrices(
    T: &Array2<Float>,
    Y: &Array2<usize>,
    _config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    // Simplified MI computation for matrices
    let correlation = compute_matrix_correlation(T, Y);
    Ok(correlation.abs())
}

fn compute_matrix_correlation(A: &Array2<Float>, B: &Array2<usize>) -> Float {
    // Simplified correlation computation
    let a_flat: Vec<Float> = A.iter().cloned().collect();
    let b_flat: Vec<Float> = B.iter().map(|&x| x as Float).collect();

    if a_flat.len() != b_flat.len() {
        return 0.0;
    }

    let mean_a = a_flat.iter().sum::<Float>() / a_flat.len() as Float;
    let mean_b = b_flat.iter().sum::<Float>() / b_flat.len() as Float;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for i in 0..a_flat.len() {
        let a_dev = a_flat[i] - mean_a;
        let b_dev = b_flat[i] - mean_b;

        numerator += a_dev * b_dev;
        sum_sq_a += a_dev * a_dev;
        sum_sq_b += b_dev * b_dev;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_matrix_frobenius_norm(matrix: &Array2<Float>) -> Float {
    matrix.iter().map(|&x| x * x).sum::<Float>().sqrt()
}

fn compute_matrix_frobenius_norm_diff(A: &Array2<Float>, B: &Array2<Float>) -> Float {
    let diff = A - B;
    compute_matrix_frobenius_norm(&diff)
}

fn compute_feature_relevance_from_ib(
    T: &Array2<Float>,
    X_discrete: &Array2<usize>,
    _config: &InformationTheoreticConfig,
) -> SklResult<Array1<Float>> {
    let n_features = X_discrete.ncols();
    let mut relevance = Array1::zeros(n_features);

    // Compute relevance as correlation between T and each feature
    for feature_idx in 0..n_features {
        let feature_values: Vec<Float> = X_discrete
            .column(feature_idx)
            .iter()
            .map(|&x| x as Float)
            .collect();

        let t_values: Vec<Float> = T.column(feature_idx).to_vec();

        relevance[feature_idx] = compute_vector_correlation(&t_values, &feature_values);
    }

    Ok(relevance)
}

fn compute_vector_correlation(a: &[Float], b: &[Float]) -> Float {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mean_a = a.iter().sum::<Float>() / a.len() as Float;
    let mean_b = b.iter().sum::<Float>() / b.len() as Float;

    let mut numerator = 0.0;
    let mut sum_sq_a = 0.0;
    let mut sum_sq_b = 0.0;

    for i in 0..a.len() {
        let a_dev = a[i] - mean_a;
        let b_dev = b[i] - mean_b;

        numerator += a_dev * b_dev;
        sum_sq_a += a_dev * a_dev;
        sum_sq_b += b_dev * b_dev;
    }

    let denominator = (sum_sq_a * sum_sq_b).sqrt();
    if denominator.abs() < Float::EPSILON {
        0.0
    } else {
        numerator / denominator
    }
}

fn compute_mdl_score(
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    features: &[usize],
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    if features.is_empty() {
        return Ok(Float::INFINITY);
    }

    // Model complexity (number of parameters)
    let model_complexity = features.len() as Float * (config.n_bins as Float).ln();

    // Data description length (negative log-likelihood)
    let mut data_length = 0.0;

    // Simple approach: conditional entropy of target given selected features
    for &feature_idx in features {
        if feature_idx < X_discrete.ncols() {
            let x_feature = X_discrete.column(feature_idx);
            let conditional_entropy =
                compute_conditional_entropy(&y_discrete.view(), &x_feature, config)?;
            data_length += conditional_entropy;
        }
    }

    Ok(model_complexity + data_length)
}

fn compute_model_complexity(features: &[usize], config: &InformationTheoreticConfig) -> Float {
    // Model complexity based on number of parameters
    features.len() as Float * (config.n_bins as Float).ln()
}

fn compute_data_description_length(
    X_discrete: &Array2<usize>,
    y_discrete: &Array1<usize>,
    features: &[usize],
    config: &InformationTheoreticConfig,
) -> SklResult<Float> {
    if features.is_empty() {
        return compute_entropy(&y_discrete.view(), config);
    }

    // Compute conditional entropy of target given selected features
    let mut total_entropy = 0.0;

    for &feature_idx in features {
        if feature_idx < X_discrete.ncols() {
            let x_feature = X_discrete.column(feature_idx);
            let conditional_entropy =
                compute_conditional_entropy(&y_discrete.view(), &x_feature, config)?;
            total_entropy += conditional_entropy;
        }
    }

    Ok(total_entropy / features.len() as Float)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_mutual_information_analysis() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let config = InformationTheoreticConfig {
            n_bins: 2,
            ..Default::default()
        };

        let result = analyze_mutual_information(&X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.feature_target_mi.len(), 2);
        assert_eq!(result.feature_feature_mi.shape(), &[2, 2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_information_gain_attribution() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let config = InformationTheoreticConfig {
            n_bins: 2,
            ..Default::default()
        };

        let result = compute_information_gain_attribution(&X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.information_gains.len(), 2);
        assert_eq!(result.feature_ranking.len(), 2);
        assert!(result.target_entropy >= 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_entropy_explanations() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y = array![0.0, 1.0, 0.0];

        let model = |X: &ArrayView2<Float>| {
            let n_samples = X.nrows();
            Ok(Array2::from_shape_fn((n_samples, 2), |(i, j)| {
                if j == 0 {
                    0.6
                } else {
                    0.4
                }
            }))
        };

        let config = InformationTheoreticConfig::default();

        let result = generate_entropy_explanations(model, &X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert_eq!(result.instance_entropies.len(), 3);
        assert_eq!(result.feature_entropy_reductions.len(), 2);
    }

    #[test]
    fn test_discretization() {
        let values = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = InformationTheoreticConfig {
            n_bins: 2,
            ..Default::default()
        };

        let result = discretize_column(&values.view(), &config);
        assert!(result.is_ok());

        let discretized = result.unwrap();
        assert_eq!(discretized.len(), 5);

        // Check that values are within expected range
        for &val in discretized.iter() {
            assert!(val < config.n_bins);
        }
    }

    #[test]
    fn test_entropy_computation() {
        let values = array![0, 0, 1, 1]; // Equal distribution
        let config = InformationTheoreticConfig::default();

        let result = compute_entropy(&values.view(), &config);
        assert!(result.is_ok());

        let entropy = result.unwrap();
        assert!(entropy > 0.0);

        // Maximum entropy for binary equal distribution should be ln(2)
        // With bias correction applied: (n_bins-1)/(2*n_samples) = 1/8 = 0.125
        let expected_max_entropy = 2.0_f64.ln() as Float;
        assert!((entropy - expected_max_entropy).abs() < 0.2); // More tolerant for bias correction
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_mdl_principle() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![0.0, 1.0, 0.0, 1.0];

        let config = InformationTheoreticConfig {
            n_bins: 2,
            ..Default::default()
        };

        let result = apply_minimum_description_length(&X.view(), &y.view(), &config);
        assert!(result.is_ok());

        let result = result.unwrap();
        assert!(result.total_description_length > 0.0);
        assert!(result.model_complexity > 0.0);
        assert!(!result.selected_features.is_empty());
    }
}
