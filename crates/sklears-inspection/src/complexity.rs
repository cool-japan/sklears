//! Model complexity analysis

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{ArrayView1, ArrayView2};
// ✅ SciRS2 Policy Compliant Import
use scirs2_core::random::SeedableRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// Model complexity analysis result
#[derive(Debug, Clone)]
pub struct ComplexityAnalysisResult {
    /// Effective degrees of freedom
    pub effective_degrees_freedom: Float,
    /// Model complexity score (higher = more complex)
    pub complexity_score: Float,
    /// Akaike Information Criterion (AIC)
    pub aic: Float,
    /// Bayesian Information Criterion (BIC)
    pub bic: Float,
    /// Minimum Description Length (MDL)
    pub mdl: Float,
    /// Cross-validation complexity estimate
    pub cv_complexity: Float,
    /// Feature interaction complexity
    pub interaction_complexity: Float,
    /// Number of effective parameters
    pub n_effective_params: usize,
}

/// Configuration for complexity analysis
#[derive(Debug, Clone)]
pub struct ComplexityConfig {
    /// Number of cross-validation folds
    pub cv_folds: usize,
    /// Whether to include interaction analysis
    pub include_interactions: bool,
    /// Penalty coefficient for complexity
    pub complexity_penalty: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl Default for ComplexityConfig {
    fn default() -> Self {
        Self {
            cv_folds: 5,
            include_interactions: true,
            complexity_penalty: 1.0,
            random_state: None,
        }
    }
}

/// Analyze model complexity using multiple metrics
///
/// This function computes various complexity measures for a given model:
/// - Information criteria (AIC, BIC, MDL)
/// - Cross-validation based complexity
/// - Feature interaction complexity
/// - Effective degrees of freedom
///
/// # Parameters
///
/// * `predict_fn` - Model prediction function
/// * `X` - Training features
/// * `y` - Training targets
/// * `n_params` - Number of model parameters
/// * `config` - Configuration for complexity analysis
///
/// # Examples
///
/// ```
/// use sklears_inspection::complexity::{analyze_model_complexity, ComplexityConfig};
/// use scirs2_core::ndarray::array;
///
/// let predict_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row.iter().sum())
///         .collect()
/// };
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let y = array![3.0, 7.0, 11.0];
/// let n_params = 3; // Number of model parameters
///
/// let result = analyze_model_complexity(
///     &predict_fn,
///     &X.view(),
///     &y.view(),
///     n_params,
///     &ComplexityConfig::default(),
/// ).unwrap();
///
/// assert!(result.complexity_score > 0.0);
/// ```
pub fn analyze_model_complexity<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    n_params: usize,
    config: &ComplexityConfig,
) -> SklResult<ComplexityAnalysisResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() {
        return Err(SklearsError::InvalidInput(
            "X and y must have the same number of samples".to_string(),
        ));
    }

    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "X and y must have non-zero samples and features".to_string(),
        ));
    }

    // Get model predictions
    let predictions = predict_fn(X);

    // Compute residual sum of squares
    let rss = compute_residual_sum_squares(y, &predictions);

    // Compute log-likelihood (assuming Gaussian errors)
    let log_likelihood = compute_log_likelihood(y, &predictions, rss);

    // Compute information criteria
    let aic = compute_aic(log_likelihood, n_params);
    let bic = compute_bic(log_likelihood, n_params, n_samples);
    let mdl = compute_mdl(log_likelihood, n_params, n_samples);

    // Estimate effective degrees of freedom using cross-validation
    let effective_df = estimate_effective_degrees_freedom(predict_fn, X, y, config)?;

    // Compute cross-validation complexity
    let cv_complexity = compute_cv_complexity(predict_fn, X, y, config)?;

    // Compute feature interaction complexity
    let interaction_complexity = if config.include_interactions {
        compute_interaction_complexity(predict_fn, X, y)?
    } else {
        0.0
    };

    // Overall complexity score (normalized)
    let complexity_score = compute_overall_complexity_score(
        effective_df,
        cv_complexity,
        interaction_complexity,
        n_params,
        n_features,
        config.complexity_penalty,
    );

    let n_effective_params = effective_df.round() as usize;

    Ok(ComplexityAnalysisResult {
        effective_degrees_freedom: effective_df,
        complexity_score,
        aic,
        bic,
        mdl,
        cv_complexity,
        interaction_complexity,
        n_effective_params,
    })
}

/// Compute residual sum of squares
fn compute_residual_sum_squares(y_true: &ArrayView1<Float>, y_pred: &[Float]) -> Float {
    y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).powi(2))
        .sum()
}

/// Compute log-likelihood assuming Gaussian errors
fn compute_log_likelihood(y_true: &ArrayView1<Float>, y_pred: &[Float], rss: Float) -> Float {
    let n = y_true.len() as Float;
    let sigma_squared = rss / n;

    if sigma_squared <= 0.0 {
        return 0.0; // Perfect fit
    }

    -0.5 * n * (2.0 * std::f64::consts::PI * sigma_squared).ln() - 0.5 * rss / sigma_squared
}

/// Compute Akaike Information Criterion
fn compute_aic(log_likelihood: Float, n_params: usize) -> Float {
    -2.0 * log_likelihood + 2.0 * n_params as Float
}

/// Compute Bayesian Information Criterion
fn compute_bic(log_likelihood: Float, n_params: usize, n_samples: usize) -> Float {
    -2.0 * log_likelihood + (n_samples as Float).ln() * n_params as Float
}

/// Compute Minimum Description Length
fn compute_mdl(log_likelihood: Float, n_params: usize, n_samples: usize) -> Float {
    // MDL = -log_likelihood + (k/2) * log(n)
    -log_likelihood + 0.5 * n_params as Float * (n_samples as Float).ln()
}

/// Estimate effective degrees of freedom using bootstrap
fn estimate_effective_degrees_freedom<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ComplexityConfig,
) -> SklResult<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    use scirs2_core::random::{seq::SliceRandom, SeedableRng};

    let n_samples = X.nrows();
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let mut df_estimates = Vec::new();

    // Bootstrap procedure to estimate degrees of freedom
    for _ in 0..50 {
        // Use 50 bootstrap samples
        // Create bootstrap sample
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);

        // For simplicity, estimate based on prediction variance
        let predictions = predict_fn(X);
        let pred_variance = compute_prediction_variance(&predictions);
        let noise_variance = estimate_noise_variance(y, &predictions);

        // Effective DF approximation: var(predictions) / noise_variance
        let df_est = if noise_variance > 0.0 {
            (pred_variance / noise_variance).min(n_samples as Float)
        } else {
            n_samples as Float
        };

        df_estimates.push(df_est);
    }

    // Return median estimate
    df_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok(df_estimates[df_estimates.len() / 2])
}

/// Compute prediction variance
fn compute_prediction_variance(predictions: &[Float]) -> Float {
    let mean = predictions.iter().sum::<Float>() / predictions.len() as Float;
    predictions
        .iter()
        .map(|&p| (p - mean).powi(2))
        .sum::<Float>()
        / predictions.len() as Float
}

/// Estimate noise variance from residuals
fn estimate_noise_variance(y_true: &ArrayView1<Float>, y_pred: &[Float]) -> Float {
    let residuals: Vec<Float> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| true_val - pred_val)
        .collect();

    let mean_residual = residuals.iter().sum::<Float>() / residuals.len() as Float;
    residuals
        .iter()
        .map(|&r| (r - mean_residual).powi(2))
        .sum::<Float>()
        / residuals.len() as Float
}

/// Compute cross-validation complexity
fn compute_cv_complexity<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
    config: &ComplexityConfig,
) -> SklResult<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_samples = X.nrows();
    let fold_size = n_samples / config.cv_folds;

    if fold_size == 0 {
        return Ok(1.0); // Default complexity for very small datasets
    }

    let mut cv_scores = Vec::new();

    // Simple k-fold CV (without actual retraining, just measuring prediction consistency)
    for fold in 0..config.cv_folds {
        let start_idx = fold * fold_size;
        let end_idx = if fold == config.cv_folds - 1 {
            n_samples
        } else {
            (fold + 1) * fold_size
        };

        // Create validation subset
        let val_indices: Vec<usize> = (start_idx..end_idx).collect();

        // For complexity estimation, measure prediction variability
        let predictions = predict_fn(X);
        let val_predictions: Vec<Float> = val_indices.iter().map(|&idx| predictions[idx]).collect();

        let val_y: Vec<Float> = val_indices.iter().map(|&idx| y[idx]).collect();

        // Compute score variance as complexity measure
        let score_variance = compute_score_variance(&val_y, &val_predictions);
        cv_scores.push(score_variance);
    }

    // Return mean CV complexity
    Ok(cv_scores.iter().sum::<Float>() / cv_scores.len() as Float)
}

/// Compute score variance as complexity measure
fn compute_score_variance(y_true: &[Float], y_pred: &[Float]) -> Float {
    if y_true.is_empty() {
        return 0.0;
    }

    let errors: Vec<Float> = y_true
        .iter()
        .zip(y_pred.iter())
        .map(|(&true_val, &pred_val)| (true_val - pred_val).abs())
        .collect();

    let mean_error = errors.iter().sum::<Float>() / errors.len() as Float;
    errors
        .iter()
        .map(|&e| (e - mean_error).powi(2))
        .sum::<Float>()
        / errors.len() as Float
}

/// Compute feature interaction complexity
fn compute_interaction_complexity<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    y: &ArrayView1<Float>,
) -> SklResult<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = X.ncols();

    if n_features < 2 {
        return Ok(0.0); // No interactions possible
    }

    // Measure interaction effects by comparing marginal vs joint effects
    let mut interaction_strength = 0.0;
    let baseline_predictions = predict_fn(X);

    // Sample a subset of feature pairs for efficiency
    let max_pairs = 10.min(n_features * (n_features - 1) / 2);
    let mut pair_count = 0;

    for i in 0..n_features {
        for j in (i + 1)..n_features {
            if pair_count >= max_pairs {
                break;
            }

            // Compute interaction effect between features i and j
            let interaction_effect =
                compute_pairwise_interaction(predict_fn, X, &baseline_predictions, i, j);

            interaction_strength += interaction_effect.abs();
            pair_count += 1;
        }
    }

    Ok(interaction_strength / pair_count as Float)
}

/// Compute pairwise interaction effect
fn compute_pairwise_interaction<F>(
    predict_fn: &F,
    X: &ArrayView2<Float>,
    baseline_predictions: &[Float],
    feature_i: usize,
    feature_j: usize,
) -> Float
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_samples = X.nrows();

    // Perturb both features and measure change
    let mut X_perturbed = X.to_owned();

    // Small perturbation
    let perturbation = 0.1;

    for sample_idx in 0..n_samples {
        X_perturbed[[sample_idx, feature_i]] += perturbation;
        X_perturbed[[sample_idx, feature_j]] += perturbation;
    }

    let perturbed_predictions = predict_fn(&X_perturbed.view());

    // Interaction effect is the difference from baseline
    let interaction_effect: Float = perturbed_predictions
        .iter()
        .zip(baseline_predictions.iter())
        .map(|(&perturbed, &baseline)| (perturbed - baseline).abs())
        .sum::<Float>()
        / n_samples as Float;

    interaction_effect
}

/// Compute overall complexity score
fn compute_overall_complexity_score(
    effective_df: Float,
    cv_complexity: Float,
    interaction_complexity: Float,
    n_params: usize,
    n_features: usize,
    penalty: Float,
) -> Float {
    // Normalize components
    let df_component = effective_df / n_features as Float;
    let cv_component = cv_complexity;
    let interaction_component = interaction_complexity;
    let param_component = n_params as Float / n_features as Float;

    // Weighted combination
    let complexity = 0.3 * df_component
        + 0.3 * cv_component
        + 0.2 * interaction_component
        + 0.2 * param_component;

    complexity * penalty
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::{array, ArrayView1, ArrayView2};

    #[test]
    #[allow(non_snake_case)]
    fn test_complexity_analysis() {
        // Simple linear model: y = x1 + x2
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![3.0, 7.0, 11.0, 15.0]; // Perfect linear relationship
        let n_params = 3; // 2 weights + bias

        let result = analyze_model_complexity(
            &predict_fn,
            &X.view(),
            &y.view(),
            n_params,
            &ComplexityConfig::default(),
        )
        .unwrap();

        assert!(result.complexity_score > 0.0);
        assert!(result.effective_degrees_freedom > 0.0);
        assert!(!result.aic.is_infinite());
        assert!(!result.bic.is_infinite());
        assert!(!result.mdl.is_infinite());
    }

    #[test]
    fn test_information_criteria() {
        let log_likelihood = -10.0;
        let n_params = 3;
        let n_samples = 100;

        let aic = compute_aic(log_likelihood, n_params);
        let bic = compute_bic(log_likelihood, n_params, n_samples);
        let mdl = compute_mdl(log_likelihood, n_params, n_samples);

        assert_eq!(aic, 26.0); // -2 * (-10) + 2 * 3
        assert!(bic > aic); // BIC typically penalizes complexity more
        assert!(mdl > 0.0);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_complexity_analysis_errors() {
        let predict_fn = |x: &ArrayView2<Float>| -> Vec<Float> {
            x.rows().into_iter().map(|row| row.iter().sum()).collect()
        };

        // Mismatched dimensions
        let X = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![3.0]; // Wrong length

        let result = analyze_model_complexity(
            &predict_fn,
            &X.view(),
            &y.view(),
            2,
            &ComplexityConfig::default(),
        );
        assert!(result.is_err());

        // Empty data
        let X_empty = array![[], []];
        let y_empty = array![];
        let result = analyze_model_complexity(
            &predict_fn,
            &X_empty.view(),
            &y_empty.view(),
            2,
            &ComplexityConfig::default(),
        );
        assert!(result.is_err());
    }
}
