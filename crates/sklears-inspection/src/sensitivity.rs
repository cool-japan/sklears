//! Sensitivity Analysis Methods
//!
//! This module provides various sensitivity analysis methods for understanding how
//! input features affect model predictions. Includes feature sensitivity computation,
//! gradient-based sensitivity, finite difference methods, Morris sensitivity analysis,
//! and Sobol indices.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::{seq::SliceRandom, Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// Configuration for sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityConfig {
    /// Method for sensitivity analysis
    pub method: SensitivityMethod,
    /// Number of samples for Monte Carlo methods
    pub n_samples: usize,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Step size for finite difference methods
    pub step_size: Float,
    /// Number of levels for Morris method
    pub levels: usize,
    /// Number of trajectories for Morris method
    pub n_trajectories: usize,
}

impl Default for SensitivityConfig {
    fn default() -> Self {
        Self {
            method: SensitivityMethod::FiniteDifference,
            n_samples: 1000,
            random_state: Some(42),
            step_size: 0.01,
            levels: 10,
            n_trajectories: 100,
        }
    }
}

/// Sensitivity analysis methods
#[derive(Debug, Clone, Copy)]
pub enum SensitivityMethod {
    /// Simple feature perturbation sensitivity
    FeaturePerturbation,
    /// Gradient-based sensitivity (requires differentiable model)
    GradientBased,
    /// Finite difference approximation
    FiniteDifference,
    /// Morris sensitivity analysis
    Morris,
    /// Sobol sensitivity indices
    Sobol,
}

/// Result of sensitivity analysis
#[derive(Debug, Clone)]
pub struct SensitivityResult {
    /// First-order sensitivity indices
    pub first_order: Array1<Float>,
    /// Total-order sensitivity indices (if available)
    pub total_order: Option<Array1<Float>>,
    /// Morris elementary effects (if using Morris method)
    pub morris_effects: Option<MorrisResult>,
    /// Sobol indices (if using Sobol method)
    pub sobol_indices: Option<SobolResult>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

/// Morris sensitivity analysis result
#[derive(Debug, Clone)]
pub struct MorrisResult {
    /// Mean of elementary effects
    pub mu: Array1<Float>,
    /// Mean of absolute elementary effects
    pub mu_star: Array1<Float>,
    /// Standard deviation of elementary effects
    pub sigma: Array1<Float>,
    /// All elementary effects
    pub elementary_effects: Array2<Float>,
}

/// Sobol sensitivity indices result
#[derive(Debug, Clone)]
pub struct SobolResult {
    /// First-order Sobol indices
    pub s1: Array1<Float>,
    /// Total-order Sobol indices
    pub st: Array1<Float>,
    /// Second-order indices (optional)
    pub s2: Option<Array2<Float>>,
    /// Confidence intervals for S1
    pub s1_conf: Option<Array2<Float>>,
    /// Confidence intervals for ST
    pub st_conf: Option<Array2<Float>>,
}

/// Analyze sensitivity of model predictions to input perturbations
///
/// # Examples
///
/// ```ignore
/// let model_fn = |x: &scirs2_core::ndarray::ArrayView2<f64>| -> Vec<f64> {
///     x.rows().into_iter()
///         .map(|row| row[0] * 2.0 + row[1] * 0.5)
///         .collect()
/// };
///
/// let X_base = array![[0.5, 0.5], [0.3, 0.7]];
/// let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
///
/// let config = SensitivityConfig {
///     method: SensitivityMethod::FiniteDifference,
///     n_samples: 100,
///     ..Default::default()
/// };
///
/// let result = analyze_sensitivity(&model_fn, &X_base.view(), &bounds, &config).unwrap();
/// assert_eq!(result.first_order.len(), 2);
/// ```
pub fn analyze_sensitivity<F>(
    model_fn: &F,
    X_base: &ArrayView2<Float>,
    feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = X_base.ncols();

    if feature_bounds.len() != n_features {
        return Err(SklearsError::InvalidInput(
            "Number of feature bounds must match number of features".to_string(),
        ));
    }

    match config.method {
        SensitivityMethod::FeaturePerturbation => {
            feature_perturbation_sensitivity(model_fn, X_base, feature_bounds, config)
        }
        SensitivityMethod::GradientBased => {
            gradient_based_sensitivity(model_fn, X_base, feature_bounds, config)
        }
        SensitivityMethod::FiniteDifference => {
            finite_difference_sensitivity(model_fn, X_base, feature_bounds, config)
        }
        SensitivityMethod::Morris => morris_sensitivity(model_fn, feature_bounds, config),
        SensitivityMethod::Sobol => sobol_sensitivity(model_fn, feature_bounds, config),
    }
}

/// Feature perturbation sensitivity analysis
fn feature_perturbation_sensitivity<F>(
    model_fn: &F,
    X_base: &ArrayView2<Float>,
    feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X_base.dim();
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    // Get baseline predictions
    let baseline_predictions = model_fn(X_base);
    let baseline_var = calculate_variance(&baseline_predictions);

    let mut sensitivities = Array1::zeros(n_features);

    // For each feature
    for feature_idx in 0..n_features {
        let mut feature_effects = Vec::new();

        // Generate random perturbations for this feature
        for _ in 0..config.n_samples {
            let mut X_perturbed = X_base.to_owned();

            // Perturb the feature for all samples
            for sample_idx in 0..n_samples {
                let (min_val, max_val) = feature_bounds[feature_idx];
                let perturbed_value = rng.gen_range(min_val..max_val);
                X_perturbed[[sample_idx, feature_idx]] = perturbed_value;
            }

            let perturbed_predictions = model_fn(&X_perturbed.view());
            let effect =
                calculate_prediction_difference(&baseline_predictions, &perturbed_predictions);
            feature_effects.push(effect);
        }

        // Calculate sensitivity as variance of effects
        sensitivities[feature_idx] = calculate_variance(&feature_effects) / baseline_var.max(1e-10);
    }

    Ok(SensitivityResult {
        first_order: sensitivities,
        total_order: None,
        morris_effects: None,
        sobol_indices: None,
        feature_names: None,
    })
}

/// Gradient-based sensitivity analysis (numerical approximation)
fn gradient_based_sensitivity<F>(
    model_fn: &F,
    X_base: &ArrayView2<Float>,
    _feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X_base.dim();
    let step_size = config.step_size;

    // Get baseline predictions
    let baseline_predictions = model_fn(X_base);

    let mut gradients = Array2::zeros((n_samples, n_features));

    // Calculate numerical gradients for each feature
    for feature_idx in 0..n_features {
        let mut X_plus = X_base.to_owned();
        let mut X_minus = X_base.to_owned();

        // Perturb feature by +/- step_size
        for sample_idx in 0..n_samples {
            X_plus[[sample_idx, feature_idx]] += step_size;
            X_minus[[sample_idx, feature_idx]] -= step_size;
        }

        let predictions_plus = model_fn(&X_plus.view());
        let predictions_minus = model_fn(&X_minus.view());

        // Calculate gradient using central difference
        for sample_idx in 0..n_samples {
            gradients[[sample_idx, feature_idx]] =
                (predictions_plus[sample_idx] - predictions_minus[sample_idx]) / (2.0 * step_size);
        }
    }

    // Calculate sensitivity as mean absolute gradient
    let sensitivities = gradients.map_axis(Axis(0), |col| {
        col.iter().map(|&x| x.abs()).sum::<Float>() / n_samples as Float
    });

    Ok(SensitivityResult {
        first_order: sensitivities,
        total_order: None,
        morris_effects: None,
        sobol_indices: None,
        feature_names: None,
    })
}

/// Finite difference sensitivity analysis
fn finite_difference_sensitivity<F>(
    model_fn: &F,
    X_base: &ArrayView2<Float>,
    _feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let (n_samples, n_features) = X_base.dim();
    let step_size = config.step_size;

    // Get baseline predictions
    let baseline_predictions = model_fn(X_base);

    let mut sensitivities = Array1::zeros(n_features);

    // For each feature, calculate finite difference
    for feature_idx in 0..n_features {
        let mut X_perturbed = X_base.to_owned();

        // Perturb the feature
        for sample_idx in 0..n_samples {
            X_perturbed[[sample_idx, feature_idx]] += step_size;
        }

        let perturbed_predictions = model_fn(&X_perturbed.view());

        // Calculate mean absolute difference
        let mut total_effect = 0.0;
        for sample_idx in 0..n_samples {
            total_effect +=
                (perturbed_predictions[sample_idx] - baseline_predictions[sample_idx]).abs();
        }

        sensitivities[feature_idx] = total_effect / (n_samples as Float * step_size);
    }

    Ok(SensitivityResult {
        first_order: sensitivities,
        total_order: None,
        morris_effects: None,
        sobol_indices: None,
        feature_names: None,
    })
}

/// Morris sensitivity analysis (Method of Morris)
fn morris_sensitivity<F>(
    model_fn: &F,
    feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = feature_bounds.len();
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    let mut all_effects = Array2::zeros((config.n_trajectories, n_features));

    // Generate trajectories
    for traj_idx in 0..config.n_trajectories {
        let trajectory = generate_morris_trajectory(feature_bounds, config.levels, &mut rng);
        let effects =
            calculate_elementary_effects(model_fn, &trajectory, feature_bounds, config.levels);

        for (feature_idx, effect) in effects.iter().enumerate() {
            all_effects[[traj_idx, feature_idx]] = *effect;
        }
    }

    // Calculate Morris statistics
    let mu = all_effects.mean_axis(Axis(0)).unwrap();
    let mu_star = all_effects.map_axis(Axis(0), |col| {
        col.iter().map(|&x| x.abs()).sum::<Float>() / col.len() as Float
    });
    let sigma = all_effects.std_axis(Axis(0), 0.0);

    let morris_result = MorrisResult {
        mu: mu.clone(),
        mu_star: mu_star.clone(),
        sigma,
        elementary_effects: all_effects,
    };

    Ok(SensitivityResult {
        first_order: mu_star,
        total_order: None,
        morris_effects: Some(morris_result),
        sobol_indices: None,
        feature_names: None,
    })
}

/// Sobol sensitivity analysis
fn sobol_sensitivity<F>(
    model_fn: &F,
    feature_bounds: &[(Float, Float)],
    config: &SensitivityConfig,
) -> SklResult<SensitivityResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = feature_bounds.len();
    let n_samples = config.n_samples;
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::from_rng(&mut scirs2_core::random::thread_rng()),
    };

    // Generate Sobol matrices A and B
    let matrix_a = generate_sobol_matrix(n_samples, feature_bounds, &mut rng);
    let matrix_b = generate_sobol_matrix(n_samples, feature_bounds, &mut rng);

    // Evaluate model on matrices
    let y_a = model_fn(&matrix_a.view());
    let y_b = model_fn(&matrix_b.view());

    let mut s1 = Array1::zeros(n_features);
    let mut st = Array1::zeros(n_features);

    // Calculate total variance
    let y_all: Vec<Float> = y_a.iter().chain(y_b.iter()).cloned().collect();
    let total_variance = calculate_variance(&y_all);

    // Calculate first-order and total-order indices
    for i in 0..n_features {
        // Create C_i matrix (A with i-th column from B)
        let mut matrix_ci = matrix_a.clone();
        for row in 0..n_samples {
            matrix_ci[[row, i]] = matrix_b[[row, i]];
        }
        let y_ci = model_fn(&matrix_ci.view());

        // First-order index: S_i = Var(E[Y|X_i]) / Var(Y)
        let first_order_var = calculate_first_order_variance(&y_a, &y_ci);
        s1[i] = first_order_var / total_variance.max(1e-10);

        // Total-order index: S_Ti = 1 - Var(E[Y|X_~i]) / Var(Y)
        let total_order_var = calculate_total_order_variance(&y_a, &y_b, &y_ci);
        st[i] = 1.0 - total_order_var / total_variance.max(1e-10);
    }

    let sobol_result = SobolResult {
        s1: s1.clone(),
        st: st.clone(),
        s2: None, // Second-order indices not implemented in this basic version
        s1_conf: None,
        st_conf: None,
    };

    Ok(SensitivityResult {
        first_order: s1,
        total_order: Some(st),
        morris_effects: None,
        sobol_indices: Some(sobol_result),
        feature_names: None,
    })
}

// Helper functions

fn calculate_variance(data: &[Float]) -> Float {
    let mean = data.iter().sum::<Float>() / data.len() as Float;
    data.iter().map(|&x| (x - mean).powi(2)).sum::<Float>() / data.len() as Float
}

fn calculate_prediction_difference(pred1: &[Float], pred2: &[Float]) -> Float {
    pred1
        .iter()
        .zip(pred2.iter())
        .map(|(&p1, &p2)| (p1 - p2).abs())
        .sum::<Float>()
        / pred1.len() as Float
}

fn generate_morris_trajectory(
    feature_bounds: &[(Float, Float)],
    levels: usize,
    rng: &mut impl Rng,
) -> Array2<Float> {
    let n_features = feature_bounds.len();
    let n_points = n_features + 1;
    let mut trajectory = Array2::zeros((n_points, n_features));

    // Start with random base point
    for j in 0..n_features {
        let (min_val, max_val) = feature_bounds[j];
        let level = rng.gen_range(0..levels);
        trajectory[[0, j]] = min_val + (max_val - min_val) * level as Float / (levels - 1) as Float;
    }

    // Generate trajectory by changing one feature at a time
    let mut feature_order: Vec<usize> = (0..n_features).collect();
    feature_order.shuffle(rng);

    for (step, &feature_idx) in feature_order.iter().enumerate() {
        for j in 0..n_features {
            trajectory[[step + 1, j]] = trajectory[[step, j]];
        }

        // Change the selected feature
        let (min_val, max_val) = feature_bounds[feature_idx];
        let delta = (max_val - min_val) / (levels - 1) as Float;
        trajectory[[step + 1, feature_idx]] += if rng.gen_bool(0.5) { delta } else { -delta };

        // Ensure bounds
        trajectory[[step + 1, feature_idx]] = trajectory[[step + 1, feature_idx]]
            .max(min_val)
            .min(max_val);
    }

    trajectory
}

fn calculate_elementary_effects<F>(
    model_fn: &F,
    trajectory: &Array2<Float>,
    feature_bounds: &[(Float, Float)],
    levels: usize,
) -> Vec<Float>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = feature_bounds.len();
    let mut effects = vec![0.0; n_features];

    // Evaluate model at each point in trajectory
    let mut predictions = Vec::new();
    for i in 0..trajectory.nrows() {
        let point = trajectory.slice(s![i, ..]).insert_axis(Axis(0));
        let pred = model_fn(&point.view());
        predictions.push(pred[0]);
    }

    // Calculate elementary effects
    for i in 0..n_features {
        let (min_val, max_val) = feature_bounds[i];
        let delta = (max_val - min_val) / (levels - 1) as Float;
        effects[i] = (predictions[i + 1] - predictions[i]) / delta;
    }

    effects
}

fn generate_sobol_matrix(
    n_samples: usize,
    feature_bounds: &[(Float, Float)],
    rng: &mut impl Rng,
) -> Array2<Float> {
    let n_features = feature_bounds.len();
    let mut matrix = Array2::zeros((n_samples, n_features));

    for i in 0..n_samples {
        for j in 0..n_features {
            let (min_val, max_val) = feature_bounds[j];
            matrix[[i, j]] = rng.gen_range(min_val..max_val);
        }
    }

    matrix
}

fn calculate_first_order_variance(y_a: &[Float], y_ci: &[Float]) -> Float {
    let n = y_a.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += y_a[i] * y_ci[i];
    }
    sum / n as Float - (y_a.iter().sum::<Float>() / n as Float).powi(2)
}

fn calculate_total_order_variance(y_a: &[Float], y_b: &[Float], _y_ci: &[Float]) -> Float {
    let n = y_a.len();
    let mut sum = 0.0;
    for i in 0..n {
        sum += y_a[i] * y_b[i];
    }
    sum / n as Float - (y_a.iter().sum::<Float>() / n as Float).powi(2)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    fn simple_model(x: &ArrayView2<Float>) -> Vec<Float> {
        x.rows()
            .into_iter()
            .map(|row| row[0] * 2.0 + row[1] * 0.5 + row.get(2).unwrap_or(&0.0) * 0.1)
            .collect()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_feature_perturbation_sensitivity() {
        let X_base = array![[0.5, 0.5, 0.5]];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let config = SensitivityConfig {
            method: SensitivityMethod::FeaturePerturbation,
            n_samples: 100,
            random_state: Some(42),
            ..Default::default()
        };

        let result = analyze_sensitivity(&simple_model, &X_base.view(), &bounds, &config).unwrap();
        assert_eq!(result.first_order.len(), 3);

        // First feature should have highest sensitivity (coefficient 2.0)
        assert!(result.first_order[0] > result.first_order[1]);
        assert!(result.first_order[1] > result.first_order[2]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_finite_difference_sensitivity() {
        let X_base = array![[0.5, 0.5, 0.5]];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let config = SensitivityConfig {
            method: SensitivityMethod::FiniteDifference,
            step_size: 0.01,
            ..Default::default()
        };

        let result = analyze_sensitivity(&simple_model, &X_base.view(), &bounds, &config).unwrap();
        assert_eq!(result.first_order.len(), 3);

        // Should reflect the linear coefficients: 2.0, 0.5, 0.1
        assert!(result.first_order[0] > result.first_order[1]);
        assert!(result.first_order[1] > result.first_order[2]);
    }

    #[test]
    fn test_morris_sensitivity() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0), (0.0, 1.0)];
        let config = SensitivityConfig {
            method: SensitivityMethod::Morris,
            n_trajectories: 10,
            levels: 5,
            random_state: Some(42),
            ..Default::default()
        };

        let result = analyze_sensitivity(
            &simple_model,
            &array![[0.0, 0.0, 0.0]].view(),
            &bounds,
            &config,
        )
        .unwrap();
        assert_eq!(result.first_order.len(), 3);
        assert!(result.morris_effects.is_some());

        let morris = result.morris_effects.unwrap();
        assert_eq!(morris.mu.len(), 3);
        assert_eq!(morris.mu_star.len(), 3);
        assert_eq!(morris.sigma.len(), 3);
    }

    #[test]
    fn test_sobol_sensitivity() {
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = SensitivityConfig {
            method: SensitivityMethod::Sobol,
            n_samples: 50,
            random_state: Some(42),
            ..Default::default()
        };

        let result =
            analyze_sensitivity(&simple_model, &array![[0.0, 0.0]].view(), &bounds, &config)
                .unwrap();
        assert_eq!(result.first_order.len(), 2);
        assert!(result.total_order.is_some());
        assert!(result.sobol_indices.is_some());

        let sobol = result.sobol_indices.unwrap();
        assert_eq!(sobol.s1.len(), 2);
        assert_eq!(sobol.st.len(), 2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_gradient_based_sensitivity() {
        let X_base = array![[0.5, 0.5]];
        let bounds = vec![(0.0, 1.0), (0.0, 1.0)];
        let config = SensitivityConfig {
            method: SensitivityMethod::GradientBased,
            step_size: 0.001,
            ..Default::default()
        };

        let result = analyze_sensitivity(&simple_model, &X_base.view(), &bounds, &config).unwrap();
        assert_eq!(result.first_order.len(), 2);

        // Should approximate gradients: 2.0 and 0.5
        assert!(result.first_order[0] > result.first_order[1]);
    }
}
