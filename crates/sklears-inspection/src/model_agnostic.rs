//! Model-agnostic explanation methods

// ✅ SciRS2 Policy Compliant Imports
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::collections::HashMap;

/// Result of model-agnostic explanation
#[derive(Debug, Clone)]
pub struct ModelAgnosticResult {
    /// Global feature importance scores
    pub global_importance: Array1<Float>,
    /// Local explanation for the specific instance
    pub local_explanation: LocalExplanation,
    /// Model behavior summary
    pub model_summary: ModelSummary,
    /// Explanation metadata
    pub metadata: ExplanationMetadata,
}

/// Local explanation for a specific instance
#[derive(Debug, Clone)]
pub struct LocalExplanation {
    /// Feature contributions to the prediction
    pub feature_contributions: Array1<Float>,
    /// Baseline prediction (when all features are at reference values)
    pub baseline_prediction: Float,
    /// Actual prediction for the instance
    pub prediction: Float,
    /// Confidence/certainty of the explanation
    pub confidence: Float,
    /// Top contributing features (indices and contributions)
    pub top_features: Vec<(usize, Float)>,
}

/// Summary of model behavior
#[derive(Debug, Clone)]
pub struct ModelSummary {
    /// Model complexity estimate
    pub complexity_score: Float,
    /// Prediction variance across samples
    pub prediction_variance: Float,
    /// Feature interaction strength
    pub interaction_strength: Float,
    /// Model stability (consistency across perturbations)
    pub stability: Float,
    /// Non-linearity measure
    pub nonlinearity: Float,
}

/// Metadata about the explanation process
#[derive(Debug, Clone)]
pub struct ExplanationMetadata {
    /// Method used for explanation
    pub method: String,
    /// Number of perturbations used
    pub n_perturbations: usize,
    /// Computation time in milliseconds
    pub computation_time_ms: u64,
    /// Explanation quality metrics
    pub quality_metrics: QualityMetrics,
}

/// Quality metrics for explanations
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Fidelity: how well explanation approximates the model
    pub fidelity: Float,
    /// Stability: consistency across similar inputs
    pub stability: Float,
    /// Sparsity: proportion of non-zero contributions
    pub sparsity: Float,
    /// Comprehensibility: simplified complexity measure
    pub comprehensibility: Float,
}

/// Configuration for model-agnostic explanations
#[derive(Debug, Clone)]
pub struct ModelAgnosticConfig {
    /// Number of perturbations for explanation
    pub n_perturbations: usize,
    /// Perturbation strategy
    pub perturbation_strategy: PerturbationStrategy,
    /// Sample strategy for reference baseline
    pub sampling_strategy: SamplingStrategy,
    /// Whether to compute global importance
    pub compute_global: bool,
    /// Whether to analyze feature interactions
    pub analyze_interactions: bool,
    /// Number of top features to return
    pub n_top_features: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Explanation method to use
    pub method: ExplanationMethod,
}

/// Perturbation strategies for generating explanations
#[derive(Debug, Clone, Copy)]
pub enum PerturbationStrategy {
    Gaussian {
        std: Float,
    },
    /// Uniform sampling within feature ranges
    Uniform,
    /// Sample from training data distribution
    Empirical,
    /// Targeted perturbations based on feature importance
    Targeted,
    /// Marginal sampling (independent features)
    Marginal,
}

/// Sampling strategies for baseline reference
#[derive(Debug, Clone, Copy)]
pub enum SamplingStrategy {
    /// Use training data mean as baseline
    Mean,
    /// Use training data median as baseline
    Median,
    /// Use most frequent values as baseline
    Mode,
    /// Use zero vector as baseline
    Zero,
    /// Sample from training distribution
    Empirical,
}

/// Explanation methods
#[derive(Debug, Clone, Copy)]
pub enum ExplanationMethod {
    /// LIME-style linear approximation
    LinearApproximation,
    /// Permutation-based importance
    Permutation,
    /// Shapley value estimation
    Shapley,
    /// Integrated gradients (for differentiable models)
    IntegratedGradients,
    /// Occlusion-based analysis
    Occlusion,
}

impl Default for ModelAgnosticConfig {
    fn default() -> Self {
        Self {
            n_perturbations: 5000,
            perturbation_strategy: PerturbationStrategy::Gaussian { std: 0.1 },
            sampling_strategy: SamplingStrategy::Mean,
            compute_global: true,
            analyze_interactions: false,
            n_top_features: 5,
            random_state: None,
            method: ExplanationMethod::LinearApproximation,
        }
    }
}

///
/// let predict_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row.iter().sum::<f64>() / row.len() as f64)
///         .collect()
/// };
///
/// let instance = array![1.0, 2.0, 3.0];
/// let X_train = array![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];
///
/// let result = explain_model_agnostic(
///     &predict_fn,
///     &instance.view(),
///     &X_train.view(),
///     &ModelAgnosticConfig::default(),
/// ).unwrap();
///
/// assert_eq!(result.local_explanation.feature_contributions.len(), 3);
/// assert!(result.local_explanation.confidence >= 0.0);
/// ```
pub fn explain_model_agnostic<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    X_train: &ArrayView2<Float>,
    config: &ModelAgnosticConfig,
) -> SklResult<ModelAgnosticResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let start_time = std::time::Instant::now();
    let n_features = instance.len();

    if n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "Instance must have at least one feature".to_string(),
        ));
    }

    if X_train.is_empty() {
        return Err(SklearsError::InvalidInput(
            "Training data cannot be empty".to_string(),
        ));
    }

    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    // Get original prediction
    let instance_2d = instance.insert_axis(Axis(0));
    let original_prediction = predict_fn(&instance_2d.view())[0];

    // Generate baseline reference
    let baseline = generate_baseline(X_train, config.sampling_strategy);
    let baseline_2d = baseline.view().insert_axis(Axis(0));
    let baseline_prediction = predict_fn(&baseline_2d.view())[0];

    // Generate local explanation
    let local_explanation = match config.method {
        ExplanationMethod::LinearApproximation => generate_linear_approximation_explanation(
            predict_fn,
            instance,
            &baseline.view(),
            baseline_prediction,
            original_prediction,
            X_train,
            &mut rng,
            config,
        )?,
        ExplanationMethod::Permutation => generate_permutation_explanation(
            predict_fn,
            instance,
            original_prediction,
            X_train,
            &mut rng,
            config,
        )?,
        ExplanationMethod::Shapley => generate_shapley_explanation(
            predict_fn,
            instance,
            &baseline.view(),
            baseline_prediction,
            &mut rng,
            config,
        )?,
        ExplanationMethod::Occlusion => generate_occlusion_explanation(
            predict_fn,
            instance,
            &baseline.view(),
            original_prediction,
            baseline_prediction,
            config,
        )?,
        ExplanationMethod::IntegratedGradients => {
            // Approximate integrated gradients with finite differences
            generate_integrated_gradients_explanation(
                predict_fn,
                instance,
                &baseline.view(),
                config,
            )?
        }
    };

    // Generate global importance if requested
    let global_importance = if config.compute_global {
        compute_global_importance(predict_fn, X_train, &mut rng, config)?
    } else {
        Array1::zeros(n_features)
    };

    // Analyze model behavior
    let model_summary = analyze_model_behavior(predict_fn, X_train, &mut rng, config)?;

    // Compute explanation quality
    let quality_metrics = compute_explanation_quality(
        predict_fn,
        instance,
        &local_explanation,
        X_train,
        &mut rng,
        config,
    )?;

    let computation_time_ms = start_time.elapsed().as_millis() as u64;

    let metadata = ExplanationMetadata {
        method: format!("{:?}", config.method),
        n_perturbations: config.n_perturbations,
        computation_time_ms,
        quality_metrics,
    };

    Ok(ModelAgnosticResult {
        global_importance,
        local_explanation,
        model_summary,
        metadata,
    })
}

/// Generate baseline reference point
fn generate_baseline(X_train: &ArrayView2<Float>, strategy: SamplingStrategy) -> Array1<Float> {
    let n_features = X_train.ncols();
    let mut baseline = Array1::zeros(n_features);

    match strategy {
        SamplingStrategy::Mean => {
            for j in 0..n_features {
                baseline[j] = X_train.column(j).mean().unwrap_or(0.0);
            }
        }
        SamplingStrategy::Median => {
            for j in 0..n_features {
                let mut column_values: Vec<Float> = X_train.column(j).to_vec();
                column_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let median_idx = column_values.len() / 2;
                baseline[j] = column_values[median_idx];
            }
        }
        SamplingStrategy::Mode => {
            for j in 0..n_features {
                // For simplicity, use most frequent rounded value
                let column_values: Vec<i32> = X_train
                    .column(j)
                    .iter()
                    .map(|&x| (x * 10.0).round() as i32)
                    .collect();

                let mut frequency = HashMap::new();
                for &val in &column_values {
                    *frequency.entry(val).or_insert(0) += 1;
                }

                let mode = frequency
                    .iter()
                    .max_by_key(|(_, &count)| count)
                    .map(|(&val, _)| val)
                    .unwrap_or(0);

                baseline[j] = mode as Float / 10.0;
            }
        }
        SamplingStrategy::Zero => {
            // baseline is already zero
        }
        SamplingStrategy::Empirical => {
            // Use mean for simplicity (could sample randomly)
            for j in 0..n_features {
                baseline[j] = X_train.column(j).mean().unwrap_or(0.0);
            }
        }
    }

    baseline
}

/// Generate linear approximation explanation (LIME-style)
fn generate_linear_approximation_explanation<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
    baseline_prediction: Float,
    original_prediction: Float,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<LocalExplanation>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut perturbations = Array2::zeros((config.n_perturbations, n_features));
    let mut predictions = Vec::new();
    let mut weights = Vec::new();

    // Generate perturbations
    for i in 0..config.n_perturbations {
        let perturbation =
            generate_perturbation(instance, X_train, rng, config.perturbation_strategy);
        perturbations.row_mut(i).assign(&perturbation);

        let pert_2d = perturbation.view().insert_axis(Axis(0));
        let prediction = predict_fn(&pert_2d.view())[0];
        predictions.push(prediction);

        // Weight by proximity to original instance
        let distance = compute_l2_distance(instance, &perturbation.view());
        let weight = (-distance.powi(2) / 0.5).exp(); // Gaussian kernel
        weights.push(weight);
    }

    // Fit linear model using weighted least squares
    let feature_contributions = fit_weighted_linear_model(
        &perturbations.view(),
        &predictions,
        &weights,
        instance,
        baseline,
    )?;

    // Compute explanation confidence
    let confidence = compute_explanation_confidence(&predictions, &weights);

    // Get top features
    let mut indexed_contributions: Vec<(usize, Float)> = feature_contributions
        .iter()
        .enumerate()
        .map(|(idx, &contrib)| (idx, contrib.abs()))
        .collect();
    indexed_contributions
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_features = indexed_contributions
        .into_iter()
        .take(config.n_top_features)
        .collect();

    Ok(LocalExplanation {
        feature_contributions,
        baseline_prediction,
        prediction: original_prediction,
        confidence,
        top_features,
    })
}

/// Generate perturbation based on strategy
fn generate_perturbation(
    instance: &ArrayView1<Float>,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    strategy: PerturbationStrategy,
) -> Array1<Float> {
    let n_features = instance.len();
    let mut perturbation = instance.to_owned();

    match strategy {
        PerturbationStrategy::Gaussian { std } => {
            for i in 0..n_features {
                perturbation[i] += rng.gen::<Float>() * std * 2.0 - std;
            }
        }
        PerturbationStrategy::Uniform => {
            for i in 0..n_features {
                let col_values: Vec<Float> = X_train.column(i).to_vec();
                let min_val = col_values.iter().fold(Float::INFINITY, |a, &b| a.min(b));
                let max_val = col_values
                    .iter()
                    .fold(Float::NEG_INFINITY, |a, &b| a.max(b));
                perturbation[i] = rng.gen_range(min_val..max_val);
            }
        }
        PerturbationStrategy::Empirical => {
            for i in 0..n_features {
                let col_values: Vec<Float> = X_train.column(i).to_vec();
                if !col_values.is_empty() {
                    perturbation[i] = col_values[rng.gen_range(0..col_values.len())];
                }
            }
        }
        PerturbationStrategy::Targeted => {
            // Perturb only a subset of features
            let n_perturb = rng.gen_range(1..n_features + 1);
            let mut indices: Vec<usize> = (0..n_features).collect();
            indices.sort_by(|_, _| {
                if rng.gen::<bool>() {
                    std::cmp::Ordering::Less
                } else {
                    std::cmp::Ordering::Greater
                }
            });

            for &idx in indices.iter().take(n_perturb) {
                let col_values: Vec<Float> = X_train.column(idx).to_vec();
                if !col_values.is_empty() {
                    perturbation[idx] = col_values[rng.gen_range(0..col_values.len())];
                }
            }
        }
        PerturbationStrategy::Marginal => {
            // Sample each feature independently from its marginal distribution
            for i in 0..n_features {
                let col_values: Vec<Float> = X_train.column(i).to_vec();
                if !col_values.is_empty() {
                    perturbation[i] = col_values[rng.gen_range(0..col_values.len())];
                }
            }
        }
    }

    perturbation
}

/// Compute L2 distance between two instances
fn compute_l2_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<Float>()
        .sqrt()
}

/// Fit weighted linear model for feature contributions
fn fit_weighted_linear_model(
    X: &ArrayView2<Float>,
    y: &[Float],
    weights: &[Float],
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
) -> SklResult<Array1<Float>> {
    let n_features = X.ncols();
    let mut feature_contributions = Array1::zeros(n_features);

    // Simple approach: compute weighted correlation-based contributions
    for j in 0..n_features {
        let mut weighted_sum_xy = 0.0;
        let mut weighted_sum_x = 0.0;
        let mut weighted_sum_y = 0.0;
        let mut weighted_sum_xx = 0.0;
        let mut total_weight = 0.0;

        for i in 0..X.nrows() {
            let x_val = X[[i, j]];
            let y_val = y[i];
            let weight = weights[i];

            weighted_sum_xy += weight * x_val * y_val;
            weighted_sum_x += weight * x_val;
            weighted_sum_y += weight * y_val;
            weighted_sum_xx += weight * x_val * x_val;
            total_weight += weight;
        }

        if total_weight > 0.0 {
            let mean_x = weighted_sum_x / total_weight;
            let mean_y = weighted_sum_y / total_weight;

            let covariance = (weighted_sum_xy / total_weight) - (mean_x * mean_y);
            let variance = (weighted_sum_xx / total_weight) - (mean_x * mean_x);

            if variance > 1e-10 {
                let slope = covariance / variance;
                // Contribution is slope * (instance_value - baseline_value)
                feature_contributions[j] = slope * (instance[j] - baseline[j]);
            }
        }
    }

    Ok(feature_contributions)
}

/// Compute explanation confidence
fn compute_explanation_confidence(predictions: &[Float], weights: &[Float]) -> Float {
    if predictions.is_empty() {
        return 0.0;
    }

    // Confidence based on prediction stability
    let weighted_mean = predictions
        .iter()
        .zip(weights.iter())
        .map(|(&pred, &weight)| pred * weight)
        .sum::<Float>()
        / weights.iter().sum::<Float>();

    let weighted_variance = predictions
        .iter()
        .zip(weights.iter())
        .map(|(&pred, &weight)| weight * (pred - weighted_mean).powi(2))
        .sum::<Float>()
        / weights.iter().sum::<Float>();

    // Higher variance means lower confidence
    1.0 / (1.0 + weighted_variance)
}

/// Generate permutation-based explanation
fn generate_permutation_explanation<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    original_prediction: Float,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<LocalExplanation>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut feature_contributions = Array1::zeros(n_features);

    // For each feature, measure importance by permuting it
    for feature_idx in 0..n_features {
        let mut importance_scores = Vec::new();

        for _ in 0..config.n_perturbations / n_features {
            let mut perturbed_instance = instance.to_owned();

            // Sample a random value for this feature from training data
            let col_values: Vec<Float> = X_train.column(feature_idx).to_vec();
            if !col_values.is_empty() {
                perturbed_instance[feature_idx] = col_values[rng.gen_range(0..col_values.len())];
            }

            let pert_2d = perturbed_instance.insert_axis(Axis(0));
            let perturbed_prediction = predict_fn(&pert_2d.view())[0];

            // Importance is the change in prediction
            let importance = original_prediction - perturbed_prediction;
            importance_scores.push(importance);
        }

        // Average importance for this feature
        feature_contributions[feature_idx] =
            importance_scores.iter().sum::<Float>() / importance_scores.len() as Float;
    }

    // Get top features
    let mut indexed_contributions: Vec<(usize, Float)> = feature_contributions
        .iter()
        .enumerate()
        .map(|(idx, &contrib)| (idx, contrib.abs()))
        .collect();
    indexed_contributions
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_features = indexed_contributions
        .into_iter()
        .take(config.n_top_features)
        .collect();

    Ok(LocalExplanation {
        feature_contributions,
        baseline_prediction: 0.0, // Not applicable for permutation
        prediction: original_prediction,
        confidence: 0.8, // Fixed confidence for permutation method
        top_features,
    })
}

/// Generate Shapley value explanation (approximated)
fn generate_shapley_explanation<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
    baseline_prediction: Float,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<LocalExplanation>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut shapley_values = Array1::zeros(n_features);

    // Approximate Shapley values using sampling
    for _ in 0..config.n_perturbations {
        // Random permutation of features
        let mut feature_order: Vec<usize> = (0..n_features).collect();
        feature_order.shuffle(rng);

        let mut current_instance = baseline.to_owned();
        let current_2d = current_instance.view().insert_axis(Axis(0));
        let mut prev_prediction = predict_fn(&current_2d.view())[0];

        for &feature_idx in &feature_order {
            // Add this feature to the coalition
            current_instance[feature_idx] = instance[feature_idx];
            let current_2d = current_instance.view().insert_axis(Axis(0));
            let current_prediction = predict_fn(&current_2d.view())[0];

            // Marginal contribution of this feature
            let marginal_contribution = current_prediction - prev_prediction;
            shapley_values[feature_idx] += marginal_contribution;

            prev_prediction = current_prediction;
        }
    }

    // Average the Shapley values
    shapley_values /= config.n_perturbations as Float;

    let instance_2d = instance.insert_axis(Axis(0));
    let original_prediction = predict_fn(&instance_2d.view())[0];

    // Get top features
    let mut indexed_contributions: Vec<(usize, Float)> = shapley_values
        .iter()
        .enumerate()
        .map(|(idx, &contrib)| (idx, contrib.abs()))
        .collect();
    indexed_contributions
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_features = indexed_contributions
        .into_iter()
        .take(config.n_top_features)
        .collect();

    Ok(LocalExplanation {
        feature_contributions: shapley_values,
        baseline_prediction,
        prediction: original_prediction,
        confidence: 0.85, // Fixed confidence for Shapley method
        top_features,
    })
}

/// Generate occlusion-based explanation
fn generate_occlusion_explanation<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
    original_prediction: Float,
    baseline_prediction: Float,
    config: &ModelAgnosticConfig,
) -> SklResult<LocalExplanation>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut feature_contributions = Array1::zeros(n_features);

    // For each feature, replace it with baseline value and measure change
    for feature_idx in 0..n_features {
        let mut occluded_instance = instance.to_owned();
        occluded_instance[feature_idx] = baseline[feature_idx];

        let occluded_2d = occluded_instance.insert_axis(Axis(0));
        let occluded_prediction = predict_fn(&occluded_2d.view())[0];

        // Contribution is the change when removing this feature
        feature_contributions[feature_idx] = original_prediction - occluded_prediction;
    }

    // Get top features
    let mut indexed_contributions: Vec<(usize, Float)> = feature_contributions
        .iter()
        .enumerate()
        .map(|(idx, &contrib)| (idx, contrib.abs()))
        .collect();
    indexed_contributions
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_features = indexed_contributions
        .into_iter()
        .take(config.n_top_features)
        .collect();

    Ok(LocalExplanation {
        feature_contributions,
        baseline_prediction,
        prediction: original_prediction,
        confidence: 0.9, // High confidence for occlusion method
        top_features,
    })
}

/// Generate integrated gradients explanation (approximated with finite differences)
fn generate_integrated_gradients_explanation<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    baseline: &ArrayView1<Float>,
    config: &ModelAgnosticConfig,
) -> SklResult<LocalExplanation>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut integrated_gradients = Array1::zeros(n_features);
    let n_steps = 50; // Number of integration steps

    // Compute integrated gradients along the path from baseline to instance
    for step in 0..n_steps {
        let alpha = step as Float / (n_steps - 1) as Float;
        let interpolated = baseline + &(alpha * (instance - baseline));

        // Compute gradients using finite differences
        let gradients = compute_finite_difference_gradients(predict_fn, &interpolated.view());
        integrated_gradients += &gradients;
    }

    // Average the gradients and multiply by (instance - baseline)
    integrated_gradients /= n_steps as Float;
    integrated_gradients *= &(instance - baseline);

    let instance_2d = instance.insert_axis(Axis(0));
    let original_prediction = predict_fn(&instance_2d.view())[0];
    let baseline_2d = baseline.view().insert_axis(Axis(0));
    let baseline_prediction = predict_fn(&baseline_2d.view())[0];

    // Get top features
    let mut indexed_contributions: Vec<(usize, Float)> = integrated_gradients
        .iter()
        .enumerate()
        .map(|(idx, &contrib)| (idx, contrib.abs()))
        .collect();
    indexed_contributions
        .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let top_features = indexed_contributions
        .into_iter()
        .take(config.n_top_features)
        .collect();

    Ok(LocalExplanation {
        feature_contributions: integrated_gradients,
        baseline_prediction,
        prediction: original_prediction,
        confidence: 0.8, // Moderate confidence for approximated method
        top_features,
    })
}

/// Compute finite difference gradients
fn compute_finite_difference_gradients<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
) -> Array1<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let mut gradients = Array1::zeros(n_features);
    let epsilon = 1e-5;

    for i in 0..n_features {
        // Forward difference
        let mut instance_plus = instance.to_owned();
        instance_plus[i] += epsilon;
        let plus_2d = instance_plus.insert_axis(Axis(0));
        let pred_plus = predict_fn(&plus_2d.view())[0];

        // Backward difference
        let mut instance_minus = instance.to_owned();
        instance_minus[i] -= epsilon;
        let minus_2d = instance_minus.insert_axis(Axis(0));
        let pred_minus = predict_fn(&minus_2d.view())[0];

        gradients[i] = (pred_plus - pred_minus) / (2.0 * epsilon);
    }

    gradients
}

/// Compute global feature importance
fn compute_global_importance<F>(
    predict_fn: &F,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<Array1<Float>>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = X_train.ncols();
    let mut global_importance = Array1::zeros(n_features);

    // Sample a subset of training instances for efficiency
    let n_samples = (X_train.nrows()).min(100);
    let sample_size = n_samples.min(X_train.nrows());

    for _ in 0..sample_size {
        let sample_idx = rng.gen_range(0..X_train.nrows());
        let instance = X_train.row(sample_idx);

        // Compute local importance for this instance
        let local_config = ModelAgnosticConfig {
            n_perturbations: config.n_perturbations / 10, // Fewer perturbations for global analysis
            method: ExplanationMethod::Permutation,       // Use fast method
            ..config.clone()
        };

        if let Ok(local_explanation) = generate_permutation_explanation(
            predict_fn,
            &instance,
            predict_fn(&instance.insert_axis(Axis(0)).view())[0],
            X_train,
            rng,
            &local_config,
        ) {
            // Accumulate absolute contributions
            for (i, &contrib) in local_explanation.feature_contributions.iter().enumerate() {
                global_importance[i] += contrib.abs();
            }
        }
    }

    // Normalize by number of samples
    global_importance /= sample_size as Float;

    Ok(global_importance)
}

/// Analyze model behavior
fn analyze_model_behavior<F>(
    predict_fn: &F,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<ModelSummary>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_samples = 100.min(X_train.nrows());
    let mut predictions = Vec::new();
    let mut complexities = Vec::new();

    // Sample predictions for analysis
    for _ in 0..n_samples {
        let sample_idx = rng.gen_range(0..X_train.nrows());
        let instance = X_train.row(sample_idx);
        let instance_2d = instance.insert_axis(Axis(0));
        let prediction = predict_fn(&instance_2d.view())[0];
        predictions.push(prediction);

        // Estimate local complexity using perturbation variance
        let mut local_predictions = Vec::new();
        for _ in 0..10 {
            let perturbation = generate_perturbation(
                &instance,
                X_train,
                rng,
                PerturbationStrategy::Gaussian { std: 0.05 },
            );
            let pert_2d = perturbation.view().insert_axis(Axis(0));
            local_predictions.push(predict_fn(&pert_2d.view())[0]);
        }

        let local_variance = compute_variance(&local_predictions);
        complexities.push(local_variance);
    }

    let prediction_variance = compute_variance(&predictions);
    let complexity_score = complexities.iter().sum::<Float>() / complexities.len() as Float;

    Ok(ModelSummary {
        complexity_score,
        prediction_variance,
        interaction_strength: 0.0,                    // Simplified
        stability: 1.0 / (1.0 + prediction_variance), // Higher variance = lower stability
        nonlinearity: complexity_score,               // Proxy for non-linearity
    })
}

/// Compute explanation quality metrics
fn compute_explanation_quality<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    explanation: &LocalExplanation,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<QualityMetrics>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Compute fidelity: how well explanation approximates the model
    let fidelity = compute_explanation_fidelity(predict_fn, instance, explanation);

    // Compute stability: consistency across similar inputs
    let stability = compute_explanation_stability(predict_fn, instance, X_train, rng, config)?;

    // Compute sparsity: proportion of non-zero contributions
    let non_zero_count = explanation
        .feature_contributions
        .iter()
        .filter(|&&x| x.abs() > 1e-6)
        .count();
    let sparsity =
        1.0 - (non_zero_count as Float / explanation.feature_contributions.len() as Float);

    // Compute comprehensibility: based on number of top features
    let comprehensibility = if explanation.top_features.len() <= 5 {
        1.0
    } else if explanation.top_features.len() <= 10 {
        0.7
    } else {
        0.3
    };

    Ok(QualityMetrics {
        fidelity,
        stability,
        sparsity,
        comprehensibility,
    })
}

/// Compute explanation fidelity
fn compute_explanation_fidelity<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    explanation: &LocalExplanation,
) -> Float
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Simplified fidelity: correlation between explanation and actual importance
    let instance_2d = instance.insert_axis(Axis(0));
    let original_prediction = predict_fn(&instance_2d.view())[0];

    // Check if sum of contributions approximates the prediction difference
    let explained_prediction =
        explanation.baseline_prediction + explanation.feature_contributions.sum();

    let prediction_diff = (original_prediction - explanation.baseline_prediction).abs();

    if prediction_diff < 1e-6 {
        return 1.0; // Perfect fidelity for zero difference
    }

    let error = (explained_prediction - original_prediction).abs();
    1.0 / (1.0 + error / prediction_diff)
}

/// Compute explanation stability (simplified to avoid recursion)
fn compute_explanation_stability<F>(
    predict_fn: &F,
    instance: &ArrayView1<Float>,
    X_train: &ArrayView2<Float>,
    rng: &mut scirs2_core::random::rngs::StdRng,
    config: &ModelAgnosticConfig,
) -> SklResult<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Simple stability check: measure prediction variance under small perturbations
    let n_perturbations = 10;
    let mut predictions = Vec::new();

    // Get original prediction
    let instance_2d = instance.insert_axis(Axis(0));
    let original_pred = predict_fn(&instance_2d)[0];
    predictions.push(original_pred);

    // Generate small perturbations and measure prediction consistency
    for _ in 0..n_perturbations {
        let perturbation = generate_perturbation(
            instance,
            X_train,
            rng,
            PerturbationStrategy::Gaussian { std: 0.01 },
        );
        let pert_2d = perturbation.view().insert_axis(Axis(0));
        let pred = predict_fn(&pert_2d)[0];
        predictions.push(pred);
    }

    // Compute stability as inverse of prediction variance
    let variance = compute_variance(&predictions);
    let stability = 1.0 / (1.0 + variance);

    Ok(stability)
}

/// Compute correlation between two arrays
fn compute_correlation(a: &Array1<Float>, b: &Array1<Float>) -> Float {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }

    let mean_a = a.mean().unwrap_or(0.0);
    let mean_b = b.mean().unwrap_or(0.0);

    let numerator: Float = a
        .iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - mean_a) * (y - mean_b))
        .sum();

    let sum_sq_a: Float = a.iter().map(|&x| (x - mean_a).powi(2)).sum();
    let sum_sq_b: Float = b.iter().map(|&y| (y - mean_b).powi(2)).sum();

    let denominator = (sum_sq_a * sum_sq_b).sqrt();

    if denominator < 1e-10 {
        0.0
    } else {
        numerator / denominator
    }
}

/// Compute variance of a vector
fn compute_variance(values: &[Float]) -> Float {
    if values.is_empty() {
        return 0.0;
    }

    let mean = values.iter().sum::<Float>() / values.len() as Float;
    values.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / values.len() as Float
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    #[test]
    #[allow(non_snake_case)]
    fn test_model_agnostic_explanation() {
        // Simple linear model: prediction = sum of features
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let instance = array![1.0, 2.0, 3.0];
        let X_train = array![[0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [2.0, 3.0, 4.0]];

        let result = explain_model_agnostic(
            &predict_fn,
            &instance.view(),
            &X_train.view(),
            &ModelAgnosticConfig::default(),
        )
        .unwrap();

        assert_eq!(result.local_explanation.feature_contributions.len(), 3);
        assert!(result.local_explanation.confidence >= 0.0);
        assert!(!result.local_explanation.top_features.is_empty());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_baseline_generation() {
        let X_train = array![[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]];

        let baseline = generate_baseline(&X_train.view(), SamplingStrategy::Mean);
        assert_eq!(baseline[0], 2.0); // Mean of [1, 2, 3]
        assert_eq!(baseline[1], 20.0); // Mean of [10, 20, 30]

        let baseline = generate_baseline(&X_train.view(), SamplingStrategy::Zero);
        assert_eq!(baseline[0], 0.0);
        assert_eq!(baseline[1], 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_perturbation_strategies() {
        let instance = array![1.0, 2.0];
        let X_train = array![[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]];
        let mut rng = scirs2_core::random::rngs::StdRng::seed_from_u64(42);

        let perturbation = generate_perturbation(
            &instance.view(),
            &X_train.view(),
            &mut rng,
            PerturbationStrategy::Gaussian { std: 0.1 },
        );
        assert_eq!(perturbation.len(), 2);

        let perturbation = generate_perturbation(
            &instance.view(),
            &X_train.view(),
            &mut rng,
            PerturbationStrategy::Empirical,
        );
        assert_eq!(perturbation.len(), 2);
    }

    #[test]
    fn test_explanation_quality() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let instance = array![1.0, 2.0];
        let explanation = LocalExplanation {
            feature_contributions: array![0.5, 1.0],
            baseline_prediction: 0.0,
            prediction: 3.0,
            confidence: 0.8,
            top_features: vec![(1, 1.0), (0, 0.5)],
        };

        let fidelity = compute_explanation_fidelity(&predict_fn, &instance.view(), &explanation);
        assert!(fidelity >= 0.0 && fidelity <= 1.0);
    }
}
