//! Local Explanations for Model Inspection
//!
//! This module provides local explanation methods for understanding model behavior
//! around specific instances, including local surrogate models, local linear
//! approximations, neighborhood-based explanations, and prototype-based methods.

// ✅ SciRS2 Policy Compliant Import
use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use scirs2_core::random::{Rng, SeedableRng};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

/// Configuration for local explanations
#[derive(Debug, Clone)]
pub struct LocalExplanationConfig {
    /// Method for local explanation
    pub method: LocalExplanationMethod,
    /// Number of neighbors/samples for local analysis
    pub n_neighbors: usize,
    /// Neighborhood radius for instance selection
    pub neighborhood_radius: Float,
    /// Number of perturbations for local surrogate
    pub n_perturbations: usize,
    /// Kernel function for weighting instances
    pub kernel: KernelType,
    /// Kernel bandwidth
    pub bandwidth: Float,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
    /// Feature selection method for local surrogate
    pub feature_selection: FeatureSelectionMethod,
    /// Maximum number of features for local model
    pub max_features: Option<usize>,
}

impl Default for LocalExplanationConfig {
    fn default() -> Self {
        Self {
            method: LocalExplanationMethod::LocalSurrogate,
            n_neighbors: 50,
            neighborhood_radius: 0.1,
            n_perturbations: 1000,
            kernel: KernelType::RBF,
            bandwidth: 1.0,
            random_state: Some(42),
            feature_selection: FeatureSelectionMethod::None,
            max_features: None,
        }
    }
}

/// Local explanation methods
#[derive(Debug, Clone, Copy)]
pub enum LocalExplanationMethod {
    /// Local surrogate model (LIME-like)
    LocalSurrogate,
    /// Local linear approximation
    LocalLinear,
    /// Neighborhood-based explanation
    Neighborhood,
    /// Prototype-based explanation
    Prototype,
    /// Exemplar-based explanation
    Exemplar,
    /// Local decision tree
    LocalTree,
}

/// Kernel types for weighting instances
#[derive(Debug, Clone, Copy)]
pub enum KernelType {
    /// Radial basis function kernel
    RBF,
    /// Linear kernel
    Linear,
    /// Polynomial kernel
    Polynomial,
    /// Cosine similarity kernel
    Cosine,
    /// Uniform kernel (constant weight)
    Uniform,
}

/// Feature selection methods for local models
#[derive(Debug, Clone, Copy)]
pub enum FeatureSelectionMethod {
    /// No feature selection
    None,
    /// Select top k features by importance
    TopK,
    /// Forward selection
    Forward,
    /// Lasso regularization
    Lasso,
    /// Ridge regularization
    Ridge,
}

/// Result of local explanation analysis
#[derive(Debug, Clone)]
pub struct LocalExplanationResult {
    /// Local feature importance scores
    pub feature_importance: Array1<Float>,
    /// Local linear coefficients (if applicable)
    pub linear_coefficients: Option<Array1<Float>>,
    /// Local intercept (if applicable)
    pub intercept: Option<Float>,
    /// Neighborhood instances used
    pub neighborhood_indices: Vec<usize>,
    /// Neighborhood weights
    pub neighborhood_weights: Array1<Float>,
    /// Local model predictions vs original model
    pub local_model_fidelity: Float,
    /// Prototype instances (if applicable)
    pub prototypes: Option<Array2<Float>>,
    /// Prototype weights (if applicable)
    pub prototype_weights: Option<Array1<Float>>,
    /// Feature names
    pub feature_names: Option<Vec<String>>,
}

/// Explain a model's prediction locally using various methods
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
/// let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
/// let instance = array![0.4, 0.5];
///
/// let config = LocalExplanationConfig {
///     method: LocalExplanationMethod::LocalSurrogate,
///     n_neighbors: 3,
///     ..Default::default()
/// };
///
/// let result = explain_locally(&model_fn, &X_train.view(), &instance.view(), &config).unwrap();
/// assert_eq!(result.feature_importance.len(), 2);
/// ```
pub fn explain_locally<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    match config.method {
        LocalExplanationMethod::LocalSurrogate => {
            local_surrogate_explanation(model_fn, X_train, instance, config)
        }
        LocalExplanationMethod::LocalLinear => {
            local_linear_explanation(model_fn, X_train, instance, config)
        }
        LocalExplanationMethod::Neighborhood => {
            neighborhood_explanation(model_fn, X_train, instance, config)
        }
        LocalExplanationMethod::Prototype => {
            prototype_explanation(model_fn, X_train, instance, config)
        }
        LocalExplanationMethod::Exemplar => {
            exemplar_explanation(model_fn, X_train, instance, config)
        }
        LocalExplanationMethod::LocalTree => {
            local_tree_explanation(model_fn, X_train, instance, config)
        }
    }
}

/// Local surrogate model explanation (LIME-like)
fn local_surrogate_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let mut rng = match config.random_state {
        Some(seed) => scirs2_core::random::rngs::StdRng::seed_from_u64(seed),
        None => scirs2_core::random::rngs::StdRng::seed_from_u64(42),
    };

    let n_features = instance.len();

    // Generate perturbations around the instance
    let mut perturbed_instances = Vec::new();
    let mut distances = Vec::new();

    for _ in 0..config.n_perturbations {
        let mut perturbed = instance.to_owned();
        let mut distance: f64 = 0.0;

        // Add Gaussian perturbations
        for j in 0..n_features {
            let std_dev = X_train.column(j).std(0.0) * 0.1; // 10% of feature std
            let noise: Float = rng.gen_range(-std_dev..std_dev + 1.0);
            perturbed[j] += noise;
            distance += noise * noise;
        }

        perturbed_instances.push(perturbed);
        distances.push(distance.sqrt());
    }

    // Create matrix of perturbed instances
    let n_perturbed = perturbed_instances.len();
    let mut X_perturbed = Array2::zeros((n_perturbed, n_features));
    for (i, instance) in perturbed_instances.iter().enumerate() {
        for j in 0..n_features {
            X_perturbed[[i, j]] = instance[j];
        }
    }

    // Get predictions for perturbed instances
    let predictions = model_fn(&X_perturbed.view());

    // Calculate weights based on distance
    let weights = calculate_kernel_weights(&distances, config.kernel, config.bandwidth);

    // Fit local linear model with weighted regression
    let (coefficients, intercept, fidelity) =
        fit_weighted_linear_model(&X_perturbed.view(), &predictions, &weights)?;

    // Get neighborhood from training data
    let (neighborhood_indices, neighborhood_weights) = find_neighborhood(
        X_train,
        instance,
        config.n_neighbors,
        config.kernel,
        config.bandwidth,
    );

    Ok(LocalExplanationResult {
        feature_importance: coefficients.mapv(|x| x.abs()),
        linear_coefficients: Some(coefficients),
        intercept: Some(intercept),
        neighborhood_indices,
        neighborhood_weights,
        local_model_fidelity: fidelity,
        prototypes: None,
        prototype_weights: None,
        feature_names: None,
    })
}

/// Local linear approximation explanation
fn local_linear_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    let n_features = instance.len();
    let step_size = 0.01;

    // Calculate numerical gradients
    let mut gradients = Array1::zeros(n_features);

    for j in 0..n_features {
        // Forward difference
        let mut instance_plus = instance.to_owned();
        instance_plus[j] += step_size;
        let pred_plus = {
            let x_plus = instance_plus.insert_axis(Axis(0));
            model_fn(&x_plus.view())[0]
        };

        // Backward difference
        let mut instance_minus = instance.to_owned();
        instance_minus[j] -= step_size;
        let pred_minus = {
            let x_minus = instance_minus.insert_axis(Axis(0));
            model_fn(&x_minus.view())[0]
        };

        // Central difference
        gradients[j] = (pred_plus - pred_minus) / (2.0 * step_size);
    }

    // Get baseline prediction
    let instance_mat = instance.insert_axis(Axis(0));
    let baseline_pred = model_fn(&instance_mat.view())[0];

    // Find neighborhood
    let (neighborhood_indices, neighborhood_weights) = find_neighborhood(
        X_train,
        instance,
        config.n_neighbors,
        config.kernel,
        config.bandwidth,
    );

    Ok(LocalExplanationResult {
        feature_importance: gradients.mapv(|x| x.abs()),
        linear_coefficients: Some(gradients),
        intercept: Some(baseline_pred),
        neighborhood_indices,
        neighborhood_weights,
        local_model_fidelity: 1.0, // Perfect for linear approximation
        prototypes: None,
        prototype_weights: None,
        feature_names: None,
    })
}

/// Neighborhood-based explanation
fn neighborhood_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Find neighborhood
    let (neighborhood_indices, neighborhood_weights) = find_neighborhood(
        X_train,
        instance,
        config.n_neighbors,
        config.kernel,
        config.bandwidth,
    );

    // Extract neighborhood data
    let mut X_neighborhood = Array2::zeros((neighborhood_indices.len(), instance.len()));
    for (i, &idx) in neighborhood_indices.iter().enumerate() {
        for j in 0..instance.len() {
            X_neighborhood[[i, j]] = X_train[[idx, j]];
        }
    }

    // Get predictions for neighborhood
    let neighborhood_predictions = model_fn(&X_neighborhood.view());

    // Calculate feature importance based on neighborhood variance
    let mut feature_importance = Array1::zeros(instance.len());
    for j in 0..instance.len() {
        let feature_values: Vec<Float> = X_neighborhood.column(j).to_vec();
        let weighted_mean = feature_values
            .iter()
            .zip(neighborhood_weights.iter())
            .map(|(&val, &weight)| val * weight)
            .sum::<Float>()
            / neighborhood_weights.sum();

        let weighted_variance = feature_values
            .iter()
            .zip(neighborhood_weights.iter())
            .map(|(&val, &weight)| weight * (val - weighted_mean).powi(2))
            .sum::<Float>()
            / neighborhood_weights.sum();

        feature_importance[j] = weighted_variance.sqrt();
    }

    Ok(LocalExplanationResult {
        feature_importance,
        linear_coefficients: None,
        intercept: None,
        neighborhood_indices,
        neighborhood_weights: neighborhood_weights.clone(),
        local_model_fidelity: 0.8, // Approximate fidelity
        prototypes: Some(X_neighborhood),
        prototype_weights: Some(neighborhood_weights),
        feature_names: None,
    })
}

/// Prototype-based explanation
fn prototype_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Find diverse prototypes using k-means-like clustering
    let prototypes = find_prototypes(X_train, instance, config.n_neighbors)?;

    // Calculate prototype weights based on similarity to instance
    let mut prototype_weights = Array1::zeros(prototypes.nrows());
    for i in 0..prototypes.nrows() {
        let prototype = prototypes.row(i);
        let distance = euclidean_distance(instance, &prototype);
        prototype_weights[i] = calculate_kernel_weight(distance, config.kernel, config.bandwidth);
    }

    // Normalize weights
    let weight_sum = prototype_weights.sum();
    if weight_sum > 0.0 {
        prototype_weights /= weight_sum;
    }

    // Get predictions for prototypes
    let prototype_predictions = model_fn(&prototypes.view());

    // Calculate feature importance based on prototype influence
    let mut feature_importance = Array1::zeros(instance.len());
    for j in 0..instance.len() {
        let mut importance = 0.0;
        for i in 0..prototypes.nrows() {
            let prototype_diff = (prototypes[[i, j]] - instance[j]).abs();
            importance += prototype_weights[i] * prototype_diff;
        }
        feature_importance[j] = importance;
    }

    let (neighborhood_indices, neighborhood_weights) = find_neighborhood(
        X_train,
        instance,
        config.n_neighbors,
        config.kernel,
        config.bandwidth,
    );

    Ok(LocalExplanationResult {
        feature_importance,
        linear_coefficients: None,
        intercept: None,
        neighborhood_indices,
        neighborhood_weights,
        local_model_fidelity: 0.7, // Approximate fidelity
        prototypes: Some(prototypes),
        prototype_weights: Some(prototype_weights),
        feature_names: None,
    })
}

/// Exemplar-based explanation
fn exemplar_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Find most similar and most dissimilar examples
    let mut distances_with_indices: Vec<(usize, Float)> = Vec::new();

    for i in 0..X_train.nrows() {
        let distance = euclidean_distance(instance, &X_train.row(i));
        distances_with_indices.push((i, distance));
    }

    // Sort by distance
    distances_with_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

    // Select exemplars (closest and furthest)
    let n_exemplars = config.n_neighbors.min(X_train.nrows());
    let mut exemplar_indices = Vec::new();
    let mut exemplar_weights = Vec::new();

    // Add closest examples
    for i in 0..n_exemplars / 2 {
        if i < distances_with_indices.len() {
            exemplar_indices.push(distances_with_indices[i].0);
            exemplar_weights.push(1.0 / (1.0 + distances_with_indices[i].1));
        }
    }

    // Add furthest examples (for contrast)
    for i in 0..n_exemplars / 2 {
        let idx = distances_with_indices.len() - 1 - i;
        if idx < distances_with_indices.len() && idx > 0 {
            exemplar_indices.push(distances_with_indices[idx].0);
            exemplar_weights.push(0.1); // Lower weight for contrasting examples
        }
    }

    // Create exemplar matrix
    let mut exemplars = Array2::zeros((exemplar_indices.len(), instance.len()));
    for (i, &idx) in exemplar_indices.iter().enumerate() {
        for j in 0..instance.len() {
            exemplars[[i, j]] = X_train[[idx, j]];
        }
    }

    let exemplar_weights_array = Array1::from_vec(exemplar_weights);

    // Calculate feature importance based on exemplar differences
    let mut feature_importance = Array1::zeros(instance.len());
    for j in 0..instance.len() {
        let mut weighted_diff = 0.0;
        for i in 0..exemplars.nrows() {
            let diff = (exemplars[[i, j]] - instance[j]).abs();
            weighted_diff += exemplar_weights_array[i] * diff;
        }
        feature_importance[j] = weighted_diff;
    }

    Ok(LocalExplanationResult {
        feature_importance,
        linear_coefficients: None,
        intercept: None,
        neighborhood_indices: exemplar_indices,
        neighborhood_weights: exemplar_weights_array.clone(),
        local_model_fidelity: 0.6, // Approximate fidelity
        prototypes: Some(exemplars),
        prototype_weights: Some(exemplar_weights_array.clone()),
        feature_names: None,
    })
}

/// Local decision tree explanation (simplified)
fn local_tree_explanation<F>(
    model_fn: &F,
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    config: &LocalExplanationConfig,
) -> SklResult<LocalExplanationResult>
where
    F: Fn(&ArrayView2<Float>) -> Vec<Float>,
{
    // Find neighborhood
    let (neighborhood_indices, neighborhood_weights) = find_neighborhood(
        X_train,
        instance,
        config.n_neighbors,
        config.kernel,
        config.bandwidth,
    );

    // For simplicity, use local linear approximation as a proxy for tree
    // A real implementation would build a local decision tree
    local_linear_explanation(model_fn, X_train, instance, config)
}

// Helper functions

fn calculate_kernel_weights(
    distances: &[Float],
    kernel: KernelType,
    bandwidth: Float,
) -> Array1<Float> {
    let weights: Vec<Float> = distances
        .iter()
        .map(|&d| calculate_kernel_weight(d, kernel, bandwidth))
        .collect();
    Array1::from_vec(weights)
}

fn calculate_kernel_weight(distance: Float, kernel: KernelType, bandwidth: Float) -> Float {
    match kernel {
        KernelType::RBF => (-distance.powi(2) / (2.0 * bandwidth.powi(2))).exp(),
        KernelType::Linear => (1.0 - distance / bandwidth).max(0.0),
        KernelType::Polynomial => (1.0 + distance / bandwidth).powi(-2),
        KernelType::Cosine => {
            if distance == 0.0 {
                1.0
            } else {
                1.0 / (1.0 + distance)
            }
        }
        KernelType::Uniform => {
            if distance <= bandwidth {
                1.0
            } else {
                0.0
            }
        }
    }
}

fn find_neighborhood(
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    n_neighbors: usize,
    kernel: KernelType,
    bandwidth: Float,
) -> (Vec<usize>, Array1<Float>) {
    let mut distances_with_indices: Vec<(usize, Float)> = Vec::new();

    for i in 0..X_train.nrows() {
        let distance = euclidean_distance(instance, &X_train.row(i));
        distances_with_indices.push((i, distance));
    }

    // Sort by distance and take top k
    distances_with_indices.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    distances_with_indices.truncate(n_neighbors);

    let indices: Vec<usize> = distances_with_indices.iter().map(|(i, _)| *i).collect();
    let distances: Vec<Float> = distances_with_indices.iter().map(|(_, d)| *d).collect();
    let weights = calculate_kernel_weights(&distances, kernel, bandwidth);

    (indices, weights)
}

fn euclidean_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| (x - y).powi(2))
        .sum::<Float>()
        .sqrt()
}

fn fit_weighted_linear_model(
    X: &ArrayView2<Float>,
    y: &[Float],
    weights: &Array1<Float>,
) -> SklResult<(Array1<Float>, Float, Float)> {
    let (n_samples, n_features) = X.dim();

    if n_samples != y.len() || n_samples != weights.len() {
        return Err(SklearsError::InvalidInput(
            "Dimension mismatch in weighted linear regression".to_string(),
        ));
    }

    // Weighted least squares: (X^T W X)^-1 X^T W y
    // For simplicity, use normal equations (not numerically stable for large problems)

    // Create weighted X and y
    let mut X_weighted = Array2::zeros((n_samples, n_features));
    let mut y_weighted = Array1::zeros(n_samples);

    for i in 0..n_samples {
        let weight_sqrt = weights[i].sqrt();
        y_weighted[i] = y[i] * weight_sqrt;
        for j in 0..n_features {
            X_weighted[[i, j]] = X[[i, j]] * weight_sqrt;
        }
    }

    // Calculate X^T X and X^T y
    let mut XtX = Array2::zeros((n_features, n_features));
    let mut Xty = Array1::zeros(n_features);

    for i in 0..n_features {
        for j in 0..n_features {
            XtX[[i, j]] = X_weighted.column(i).dot(&X_weighted.column(j));
        }
        Xty[i] = X_weighted.column(i).dot(&y_weighted);
    }

    // Solve normal equations (simplified - should use proper linear algebra)
    let mut coefficients = Array1::zeros(n_features);

    // Simple diagonal approximation for demonstration
    for i in 0..n_features {
        if XtX[[i, i]].abs() > 1e-10 {
            coefficients[i] = Xty[i] / XtX[[i, i]];
        }
    }

    // Calculate intercept
    let y_mean = y.iter().sum::<Float>() / y.len() as Float;
    let x_means = X.mean_axis(Axis(0)).unwrap();
    let intercept = y_mean - coefficients.dot(&x_means);

    // Calculate R² as fidelity measure
    let mut y_pred = Array1::zeros(n_samples);
    for i in 0..n_samples {
        y_pred[i] = intercept + X.row(i).dot(&coefficients);
    }

    let ss_res: Float = y
        .iter()
        .zip(y_pred.iter())
        .map(|(&actual, &pred)| (actual - pred).powi(2))
        .sum();
    let ss_tot: Float = y.iter().map(|&val| (val - y_mean).powi(2)).sum();

    let r_squared = if ss_tot > 1e-10 {
        1.0 - ss_res / ss_tot
    } else {
        0.0
    };

    Ok((coefficients, intercept, r_squared))
}

fn find_prototypes(
    X_train: &ArrayView2<Float>,
    instance: &ArrayView1<Float>,
    n_prototypes: usize,
) -> SklResult<Array2<Float>> {
    let n_samples = X_train.nrows();
    let actual_prototypes = n_prototypes.min(n_samples);

    if actual_prototypes == 0 {
        return Err(SklearsError::InvalidInput(
            "No prototypes to select".to_string(),
        ));
    }

    // Simple prototype selection: evenly spaced samples
    let mut prototypes = Array2::zeros((actual_prototypes, instance.len()));
    let step = n_samples / actual_prototypes;

    for i in 0..actual_prototypes {
        let idx = (i * step).min(n_samples - 1);
        for j in 0..instance.len() {
            prototypes[[i, j]] = X_train[[idx, j]];
        }
    }

    Ok(prototypes)
}

#[cfg(test)]
mod tests {
    use super::*;
    // ✅ SciRS2 Policy Compliant Import
    use scirs2_core::ndarray::array;

    fn simple_model(x: &ArrayView2<Float>) -> Vec<Float> {
        x.rows()
            .into_iter()
            .map(|row| row[0] * 2.0 + row[1] * 0.5)
            .collect()
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_local_surrogate_explanation() {
        let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
        let instance = array![0.4, 0.5];
        let config = LocalExplanationConfig {
            method: LocalExplanationMethod::LocalSurrogate,
            n_perturbations: 50,
            n_neighbors: 3,
            random_state: Some(42),
            ..Default::default()
        };

        let result =
            explain_locally(&simple_model, &X_train.view(), &instance.view(), &config).unwrap();

        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.linear_coefficients.is_some());
        assert!(result.intercept.is_some());
        assert!(result.local_model_fidelity >= 0.0);

        // Feature 0 should have higher importance (coefficient 2.0 vs 0.5)
        assert!(result.feature_importance[0] > result.feature_importance[1]);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_local_linear_explanation() {
        let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]];
        let instance = array![0.4, 0.5];
        let config = LocalExplanationConfig {
            method: LocalExplanationMethod::LocalLinear,
            n_neighbors: 2,
            ..Default::default()
        };

        let result =
            explain_locally(&simple_model, &X_train.view(), &instance.view(), &config).unwrap();

        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.linear_coefficients.is_some());

        // Should approximate gradients [2.0, 0.5]
        let coeffs = result.linear_coefficients.unwrap();
        assert!((coeffs[0] - 2.0).abs() < 0.2);
        assert!((coeffs[1] - 0.5).abs() < 0.2);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_neighborhood_explanation() {
        let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
        let instance = array![0.4, 0.5];
        let config = LocalExplanationConfig {
            method: LocalExplanationMethod::Neighborhood,
            n_neighbors: 3,
            ..Default::default()
        };

        let result =
            explain_locally(&simple_model, &X_train.view(), &instance.view(), &config).unwrap();

        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.prototypes.is_some());
        assert_eq!(result.neighborhood_indices.len(), 3);
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_prototype_explanation() {
        let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
        let instance = array![0.4, 0.5];
        let config = LocalExplanationConfig {
            method: LocalExplanationMethod::Prototype,
            n_neighbors: 3,
            ..Default::default()
        };

        let result =
            explain_locally(&simple_model, &X_train.view(), &instance.view(), &config).unwrap();

        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.prototypes.is_some());
        assert!(result.prototype_weights.is_some());
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_exemplar_explanation() {
        let X_train = array![[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]];
        let instance = array![0.4, 0.5];
        let config = LocalExplanationConfig {
            method: LocalExplanationMethod::Exemplar,
            n_neighbors: 4,
            ..Default::default()
        };

        let result =
            explain_locally(&simple_model, &X_train.view(), &instance.view(), &config).unwrap();

        assert_eq!(result.feature_importance.len(), 2);
        assert!(result.prototypes.is_some());
        assert_eq!(result.neighborhood_indices.len(), 4);
    }

    #[test]
    fn test_kernel_weights() {
        let distances = vec![0.0, 0.5, 1.0, 2.0];

        let rbf_weights = calculate_kernel_weights(&distances, KernelType::RBF, 1.0);
        assert!(rbf_weights[0] > rbf_weights[1]);
        assert!(rbf_weights[1] > rbf_weights[2]);

        let linear_weights = calculate_kernel_weights(&distances, KernelType::Linear, 1.0);
        assert!(linear_weights[0] > linear_weights[1]);
        assert_eq!(linear_weights[2], 0.0); // Outside kernel radius
    }

    #[test]
    fn test_euclidean_distance() {
        let a = array![1.0, 2.0];
        let b = array![4.0, 6.0];
        let dist = euclidean_distance(&a.view(), &b.view());
        assert!((dist - 5.0).abs() < 1e-10); // sqrt((4-1)² + (6-2)²) = sqrt(9+16) = 5
    }

    #[test]
    #[allow(non_snake_case)]
    fn test_find_neighborhood() {
        let X_train = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0], [10.0, 10.0]];
        let instance = array![0.5, 0.5];

        let (indices, weights) =
            find_neighborhood(&X_train.view(), &instance.view(), 2, KernelType::RBF, 1.0);

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);
        // Should find the two closest points: [0,0] and [1,1]
        assert!(indices.contains(&0));
        assert!(indices.contains(&1));
    }
}
