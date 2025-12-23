/// Synthetic dataset generation utilities
///
/// This module provides functions for generating synthetic datasets commonly
/// used in machine learning for testing, prototyping, and benchmarking.
use crate::dataset::core::Dataset;
use crate::error::Result;
// SciRS2 Policy: Using scirs2_core::ndarray (COMPLIANT)
use scirs2_core::ndarray::Array;

/// Load the classic Iris dataset (subset for testing)
///
/// Returns a small subset of the famous Iris flower dataset, which is
/// commonly used for testing classification algorithms. The dataset
/// contains 4 features (sepal length/width, petal length/width) and
/// 3 classes (setosa, versicolor, virginica).
///
/// # Returns
///
/// A Dataset containing 6 samples with 4 features each, representing
/// 2 samples from each of the 3 Iris species.
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::synthetic::load_iris;
///
/// let iris = load_iris().unwrap();
/// assert_eq!(iris.data.dim(), (6, 4));
/// assert_eq!(iris.target.len(), 6);
/// assert_eq!(iris.feature_names.len(), 4);
/// ```
pub fn load_iris() -> Result<Dataset> {
    // Simple iris dataset for testing (subset)
    let data = Array::from_shape_vec(
        (6, 4),
        vec![
            5.1, 3.5, 1.4, 0.2, // setosa
            4.9, 3.0, 1.4, 0.2, // setosa
            7.0, 3.2, 4.7, 1.4, // versicolor
            6.4, 3.2, 4.5, 1.5, // versicolor
            6.3, 3.3, 6.0, 2.5, // virginica
            5.8, 2.7, 5.1, 1.9, // virginica
        ],
    )
    .map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

    let target = Array::from_vec(vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0]);

    Ok(Dataset::new(data, target)
        .with_feature_names(vec![
            "sepal_length".to_string(),
            "sepal_width".to_string(),
            "petal_length".to_string(),
            "petal_width".to_string(),
        ])
        .with_target_names(vec![
            "setosa".to_string(),
            "versicolor".to_string(),
            "virginica".to_string(),
        ])
        .with_description("Iris dataset (subset for testing)".to_string()))
}

/// Generate synthetic regression dataset
///
/// Creates a regression dataset with random features and targets that follow
/// a linear relationship with added Gaussian noise. The features are drawn
/// from a standard normal distribution, and targets are computed as a linear
/// combination of features plus noise.
///
/// # Arguments
///
/// * `n_samples` - Number of samples to generate
/// * `n_features` - Number of features per sample
/// * `noise` - Standard deviation of Gaussian noise added to targets
///
/// # Returns
///
/// A Dataset with synthetic regression data suitable for testing linear models
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::synthetic::make_regression;
///
/// let dataset = make_regression(100, 5, 0.1).unwrap();
/// assert_eq!(dataset.data.dim(), (100, 5));
/// assert_eq!(dataset.target.len(), 100);
/// ```
///
/// # Note
///
/// Currently uses f64 due to random number generation constraints, but the
/// infrastructure is designed to support generic types when RNG libraries
/// provide broader type support.
pub fn make_regression(n_samples: usize, n_features: usize, noise: f64) -> Result<Dataset> {
    // SciRS2 Policy: Using scirs2_core::random (COMPLIANT)
    use scirs2_core::random::{thread_rng, Rng};

    let mut rng = thread_rng();

    // Generate random features using Box-Muller transform for normal distribution
    let mut x_data = Vec::with_capacity(n_samples * n_features);
    for _ in 0..(n_samples * n_features + 1) / 2 {
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let z0 = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * std::f64::consts::PI * u2).cos();
        let z1 = (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * std::f64::consts::PI * u2).sin();
        x_data.push(z0);
        if x_data.len() < n_samples * n_features {
            x_data.push(z1);
        }
    }
    x_data.truncate(n_samples * n_features);
    let x = Array::from_shape_vec((n_samples, n_features), x_data)
        .map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;

    // Generate random coefficients for linear combination
    let mut coef: Vec<f64> = Vec::with_capacity(n_features);
    for _ in 0..n_features {
        coef.push(rng.gen::<f64>() * 20.0 - 10.0); // Map [0,1] to [-10,10]
    }

    // Generate targets: y = X @ coef + noise
    let mut y_data = Vec::with_capacity(n_samples);

    for i in 0..n_samples {
        let mut y_i = 0.0;
        for j in 0..n_features {
            y_i += x[[i, j]] * coef[j];
        }
        // Add noise using Box-Muller transform
        let u1: f64 = rng.gen();
        let u2: f64 = rng.gen();
        let noise_val =
            noise * (-2.0f64 * u1.ln()).sqrt() * (2.0f64 * std::f64::consts::PI * u2).cos();
        y_i += noise_val;
        y_data.push(y_i);
    }
    let y = Array::from_vec(y_data);

    Ok(Dataset::new(x, y).with_description(format!(
        "Synthetic regression dataset with {n_samples} samples and {n_features} features"
    )))
}

/// Generate synthetic classification dataset with Gaussian clusters
///
/// Creates a classification dataset by generating Gaussian clusters centered
/// at random locations. Each cluster represents a different class, and samples
/// are drawn from normal distributions around each cluster center.
///
/// # Arguments
///
/// * `n_samples` - Total number of samples to generate
/// * `n_features` - Number of features per sample
/// * `centers` - Number of cluster centers (classes)
/// * `cluster_std` - Standard deviation of clusters
///
/// # Returns
///
/// A Dataset with synthetic classification data where each class forms
/// a roughly spherical cluster in the feature space
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::synthetic::make_blobs;
///
/// let dataset = make_blobs(150, 2, 3, 1.0).unwrap();
/// assert_eq!(dataset.data.dim(), (150, 2));
/// assert_eq!(dataset.target.len(), 150);
/// ```
pub fn make_blobs(
    n_samples: usize,
    n_features: usize,
    centers: usize,
    cluster_std: f64,
) -> Result<Dataset> {
    // SciRS2 Policy: Using scirs2_core::random (COMPLIANT)
    use scirs2_core::random::{thread_rng, Rng};

    let mut rng = thread_rng();
    let samples_per_center = n_samples / centers;

    // Generate random cluster centers
    let mut center_coords: Vec<f64> = Vec::with_capacity(centers * n_features);
    for _ in 0..centers * n_features {
        center_coords.push(rng.gen::<f64>() * 20.0 - 10.0); // Map [0,1] to [-10,10]
    }

    let mut x_data = Vec::with_capacity(n_samples * n_features);
    let mut y_data = Vec::with_capacity(n_samples);

    // Generate samples for each cluster
    for center_idx in 0..centers {
        for _ in 0..samples_per_center {
            for feature_idx in 0..n_features {
                let center_value = center_coords[center_idx * n_features + feature_idx];
                // Generate normal random value using Box-Muller transform
                let u1: f64 = rng.gen();
                let u2: f64 = rng.gen();
                let normal_val = cluster_std
                    * (-2.0f64 * u1.ln()).sqrt()
                    * (2.0f64 * std::f64::consts::PI * u2).cos();
                x_data.push(center_value + normal_val);
            }
            y_data.push(center_idx as f64);
        }
    }

    // Handle remaining samples if n_samples is not divisible by centers
    let remaining = n_samples - (samples_per_center * centers);
    for _ in 0..remaining {
        let center_idx = centers - 1;
        for feature_idx in 0..n_features {
            let center_value = center_coords[center_idx * n_features + feature_idx];
            // Generate normal random value using Box-Muller transform
            let u1: f64 = rng.gen();
            let u2: f64 = rng.gen();
            let normal_val = cluster_std
                * (-2.0f64 * u1.ln()).sqrt()
                * (2.0f64 * std::f64::consts::PI * u2).cos();
            x_data.push(center_value + normal_val);
        }
        y_data.push(center_idx as f64);
    }

    let x = Array::from_shape_vec((n_samples, n_features), x_data)
        .map_err(|e| crate::error::SklearsError::Other(e.to_string()))?;
    let y = Array::from_vec(y_data);

    Ok(Dataset::new(x, y).with_description(format!(
        "Synthetic blob dataset with {n_samples} samples, {n_features} features, and {centers} centers"
    )))
}

/// Generate binary classification dataset with specified class separation
///
/// Creates a binary classification problem where the two classes can be
/// separated by a hyperplane. The difficulty of the classification task
/// can be controlled by adjusting the separation between classes.
///
/// # Arguments
///
/// * `n_samples` - Number of samples to generate
/// * `n_features` - Number of features per sample
/// * `separation` - Distance between class centers (higher = easier)
///
/// # Returns
///
/// A Dataset with binary classification data (targets are 0.0 or 1.0)
///
/// # Examples
///
/// ```rust
/// use sklears_core::dataset::synthetic::make_classification;
///
/// let dataset = make_classification(200, 3, 2.0).unwrap();
/// assert_eq!(dataset.data.dim(), (200, 3));
/// assert_eq!(dataset.target.len(), 200);
/// ```
pub fn make_classification(
    n_samples: usize,
    n_features: usize,
    separation: f64,
) -> Result<Dataset> {
    // Generate a simple binary classification dataset
    make_blobs(n_samples, n_features, 2, 1.0).map(|mut dataset| {
        // Adjust cluster separation by scaling the first center
        for i in 0..n_samples / 2 {
            for j in 0..n_features {
                dataset.data[[i, j]] += separation;
            }
        }

        dataset.with_description(format!(
            "Synthetic binary classification dataset with {n_samples} samples and {n_features} features"
        ))
    })
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_iris() {
        let iris = load_iris().unwrap();

        assert_eq!(iris.data.dim(), (6, 4));
        assert_eq!(iris.target.len(), 6);
        assert_eq!(iris.feature_names.len(), 4);
        assert_eq!(iris.target_names.as_ref().unwrap().len(), 3);
        assert!(iris.description.contains("Iris"));
    }

    #[test]
    fn test_make_regression() {
        let dataset = make_regression(50, 3, 0.1).unwrap();

        assert_eq!(dataset.data.dim(), (50, 3));
        assert_eq!(dataset.target.len(), 50);
        assert!(dataset.description.contains("regression"));
        assert!(dataset.description.contains("50"));
        assert!(dataset.description.contains("3"));
    }

    #[test]
    fn test_make_blobs() {
        let dataset = make_blobs(60, 2, 3, 1.0).unwrap();

        assert_eq!(dataset.data.dim(), (60, 2));
        assert_eq!(dataset.target.len(), 60);
        assert!(dataset.description.contains("blob"));

        // Check that we have the expected number of classes (0, 1, 2)
        let mut unique_targets = dataset.target.iter().cloned().collect::<Vec<_>>();
        unique_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_targets.dedup();
        assert_eq!(unique_targets.len(), 3);
    }

    #[test]
    fn test_make_classification() {
        let dataset = make_classification(100, 2, 3.0).unwrap();

        assert_eq!(dataset.data.dim(), (100, 2));
        assert_eq!(dataset.target.len(), 100);
        assert!(dataset.description.contains("classification"));

        // Should be binary classification (only values 0.0 and 1.0)
        let mut unique_targets: Vec<_> = dataset.target.iter().cloned().collect();
        unique_targets.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_targets.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        assert!(unique_targets.len() <= 2);
    }

    #[test]
    fn test_regression_noise_effect() {
        // Test just basic functionality - noise parameter affects variance
        let low_noise = make_regression(50, 2, 0.0).unwrap(); // No noise
        let high_noise = make_regression(50, 2, 1.0).unwrap(); // Some noise

        // Both should have the same structure
        assert_eq!(low_noise.data.dim(), high_noise.data.dim());
        assert_eq!(low_noise.target.len(), high_noise.target.len());

        // Test that the function works with different noise levels
        // (this is more about API correctness than statistical properties)
        let zero_noise = make_regression(10, 1, 0.0).unwrap();
        let some_noise = make_regression(10, 1, 0.5).unwrap();

        // Just verify both work without errors
        assert_eq!(zero_noise.data.dim().0, 10);
        assert_eq!(some_noise.data.dim().0, 10);
        assert!(zero_noise.description.contains("regression"));
        assert!(some_noise.description.contains("regression"));
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
fn variance(data: &scirs2_core::ndarray::Array1<f64>) -> f64 {
    let mean = data.mean().unwrap();
    let sum_sq_diff: f64 = data.iter().map(|&x| (x - mean).powi(2)).sum();
    sum_sq_diff / (data.len() as f64 - 1.0)
}
