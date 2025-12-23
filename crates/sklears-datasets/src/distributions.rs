//! Statistical distribution dataset generators
//!
//! This module provides generators for various statistical distributions including
//! Gaussian mixtures, heavy-tailed distributions, and multivariate distributions.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal, Gamma};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Generate samples from a Gaussian mixture model
///
/// Creates samples from a mixture of multivariate Gaussian distributions with
/// specified means, covariances, and mixing weights. This is useful for testing
/// clustering algorithms and studying multi-modal data.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `means`: Matrix where each row is a component mean vector
/// - `covariances`: Covariance matrices for each component (simplified to diagonal)
/// - `weights`: Mixing weights for each component (must sum to 1.0)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (data matrix, component labels)
pub fn make_gaussian_mixture(
    n_samples: usize,
    means: &Array2<f64>,
    covariances: &Array2<f64>, // Each row contains diagonal covariance values
    weights: &Array1<f64>,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let n_components = means.nrows();
    let n_features = means.ncols();

    if n_components == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "means matrix cannot have zero dimensions".to_string(),
        ));
    }

    if covariances.shape() != means.shape() {
        return Err(SklearsError::InvalidInput(
            "covariances must have same shape as means".to_string(),
        ));
    }

    if weights.len() != n_components {
        return Err(SklearsError::InvalidInput(
            "weights must have same length as number of components".to_string(),
        ));
    }

    // Check if weights sum to approximately 1.0
    let weight_sum = weights.sum();
    if (weight_sum - 1.0).abs() > 1e-10 {
        return Err(SklearsError::InvalidInput(
            "weights must sum to 1.0".to_string(),
        ));
    }

    // Check if all weights are non-negative
    if weights.iter().any(|&w| w < 0.0) {
        return Err(SklearsError::InvalidInput(
            "all weights must be non-negative".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut data = Array2::zeros((n_samples, n_features));
    let mut labels = Array1::zeros(n_samples);

    // Create cumulative distribution for component selection
    let mut cumulative_weights = Array1::zeros(n_components);
    cumulative_weights[0] = weights[0];
    for i in 1..n_components {
        cumulative_weights[i] = cumulative_weights[i - 1] + weights[i];
    }

    for sample_idx in 0..n_samples {
        // Select component based on weights
        let rand_val = rng.gen();
        let component = cumulative_weights
            .iter()
            .position(|&cum_weight| rand_val <= cum_weight)
            .unwrap_or(n_components - 1);

        labels[sample_idx] = component as i32;

        // Generate sample from selected component
        for feature_idx in 0..n_features {
            let mean_val = means[[component, feature_idx]];
            let std_val = covariances[[component, feature_idx]].sqrt();

            let normal = Normal::new(mean_val, std_val).unwrap();
            data[[sample_idx, feature_idx]] = rng.sample(normal);
        }
    }

    Ok((data, labels))
}

/// Generate samples from a mixture of different distribution types
///
/// Creates samples from a mixture where each component follows a different
/// type of distribution (normal, uniform, exponential, gamma). This is useful
/// for testing robust statistical methods.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `distribution_types`: Vector of distribution names
/// - `parameters`: Matrix where each row contains parameters for corresponding distribution
/// - `weights`: Mixing weights for each distribution
/// - `random_state`: Random seed for reproducibility
///
/// # Distribution types and parameters
/// - `"normal"`: [mean, std]
/// - `"uniform"`: [low, high]
/// - `"exponential"`: [lambda]
/// - `"gamma"`: [shape, scale]
///
/// # Returns
/// Tuple of (1D array of samples, distribution labels)
pub fn make_distribution_mixture(
    n_samples: usize,
    distribution_types: &[&str],
    parameters: &Array2<f64>, // Each row: parameters for one distribution
    weights: &Array1<f64>,
    random_state: Option<u64>,
) -> Result<(Array1<f64>, Array1<i32>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let n_distributions = distribution_types.len();
    if n_distributions == 0 {
        return Err(SklearsError::InvalidInput(
            "distribution_types cannot be empty".to_string(),
        ));
    }

    if parameters.nrows() != n_distributions {
        return Err(SklearsError::InvalidInput(
            "parameters must have same number of rows as distributions".to_string(),
        ));
    }

    if weights.len() != n_distributions {
        return Err(SklearsError::InvalidInput(
            "weights must have same length as number of distributions".to_string(),
        ));
    }

    // Check if weights sum to approximately 1.0
    let weight_sum = weights.sum();
    if (weight_sum - 1.0).abs() > 1e-10 {
        return Err(SklearsError::InvalidInput(
            "weights must sum to 1.0".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut data = Array1::zeros(n_samples);
    let mut labels = Array1::zeros(n_samples);

    // Create cumulative distribution for component selection
    let mut cumulative_weights = Array1::zeros(n_distributions);
    cumulative_weights[0] = weights[0];
    for i in 1..n_distributions {
        cumulative_weights[i] = cumulative_weights[i - 1] + weights[i];
    }

    for sample_idx in 0..n_samples {
        // Select distribution based on weights
        let rand_val = rng.gen();
        let dist_idx = cumulative_weights
            .iter()
            .position(|&cum_weight| rand_val <= cum_weight)
            .unwrap_or(n_distributions - 1);

        labels[sample_idx] = dist_idx as i32;

        // Generate sample from selected distribution
        let dist_type = distribution_types[dist_idx];
        let params = parameters.row(dist_idx);

        data[sample_idx] = match dist_type {
            "normal" => {
                if params.len() < 2 {
                    return Err(SklearsError::InvalidInput(
                        "normal distribution requires 2 parameters (mean, std)".to_string(),
                    ));
                }
                let normal = Normal::new(params[0], params[1]).unwrap();
                rng.sample(normal)
            }
            "uniform" => {
                if params.len() < 2 {
                    return Err(SklearsError::InvalidInput(
                        "uniform distribution requires 2 parameters (low, high)".to_string(),
                    ));
                }
                rng.gen_range(params[0]..params[1])
            }
            "exponential" => {
                if params.len() < 1 {
                    return Err(SklearsError::InvalidInput(
                        "exponential distribution requires 1 parameter (lambda)".to_string(),
                    ));
                }
                // Exponential using inverse transform sampling
                let u: f64 = rng.gen();
                -u.ln() / params[0]
            }
            "gamma" => {
                if params.len() < 2 {
                    return Err(SklearsError::InvalidInput(
                        "gamma distribution requires 2 parameters (shape, scale)".to_string(),
                    ));
                }
                let gamma_dist = Gamma::new(params[0], params[1]).unwrap();
                rng.sample(gamma_dist)
            }
            _ => {
                return Err(SklearsError::InvalidInput(format!(
                    "Unknown distribution type: {}",
                    dist_type
                )));
            }
        };
    }

    Ok((data, labels))
}

/// Generate samples from a multivariate mixture with different covariance structures
///
/// Creates a multivariate Gaussian mixture with automatically generated component
/// centers and configurable covariance structure types.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_features`: Number of features
/// - `n_components`: Number of mixture components
/// - `covariance_type`: Type of covariance ("diagonal", "spherical", "tied")
/// - `cluster_std`: Standard deviation for component separation
/// - `random_state`: Random seed for reproducibility
///
/// # Covariance types
/// - `"diagonal"`: Different variance for each feature in each component
/// - `"spherical"`: Same variance for all features in each component
/// - `"tied"`: Same covariance for all components
///
/// # Returns
/// Tuple of (data matrix, component labels)
pub fn make_multivariate_mixture(
    n_samples: usize,
    n_features: usize,
    n_components: usize,
    covariance_type: &str,
    cluster_std: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<i32>)> {
    if n_samples == 0 || n_features == 0 || n_components == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_features, and n_components must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate random component centers
    let mut centers = Array2::zeros((n_components, n_features));
    for i in 0..n_components {
        for j in 0..n_features {
            centers[[i, j]] = rng.random_range(-10.0, 10.0);
        }
    }

    // Equal weights for simplicity
    let weight = 1.0 / n_components as f64;
    let weights = Array1::from_elem(n_components, weight);

    // Create covariance matrices based on type
    let covariances = match covariance_type {
        "diagonal" => {
            // Diagonal covariance: different variance for each feature in each component
            let mut covs = Array2::zeros((n_components, n_features));
            for i in 0..n_components {
                for j in 0..n_features {
                    covs[[i, j]] = cluster_std * cluster_std * rng.random_range(0.5, 2.0);
                }
            }
            covs
        }
        "spherical" => {
            // Spherical covariance: same variance for all features in each component
            let mut covs = Array2::zeros((n_components, n_features));
            for i in 0..n_components {
                let variance = cluster_std * cluster_std * rng.random_range(0.5, 2.0);
                for j in 0..n_features {
                    covs[[i, j]] = variance;
                }
            }
            covs
        }
        "tied" => {
            // Tied covariance: same covariance for all components
            let mut covs = Array2::zeros((n_components, n_features));
            let base_variance = cluster_std * cluster_std;
            for i in 0..n_components {
                for j in 0..n_features {
                    covs[[i, j]] = base_variance;
                }
            }
            covs
        }
        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown covariance_type: {}. Use 'diagonal', 'spherical', or 'tied'",
                covariance_type
            )));
        }
    };

    // Generate the mixture
    make_gaussian_mixture(n_samples, &centers, &covariances, &weights, random_state)
}

/// Generate samples from heavy-tailed distributions
///
/// Creates samples from various heavy-tailed distributions that are useful for
/// testing robust statistical methods and studying extreme events.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `distribution_name`: Name of the distribution
/// - `parameters`: Distribution parameters (varies by distribution)
/// - `random_state`: Random seed for reproducibility
///
/// # Distribution types and parameters
/// - `"student_t"`: [degrees_of_freedom, location, scale]
/// - `"pareto"`: [shape, scale]
/// - `"cauchy"`: [location, scale]
/// - `"levy"`: [location, scale]
/// - `"log_normal"`: [mu, sigma] (parameters of underlying normal)
/// - `"weibull"`: [shape, scale]
///
/// # Returns
/// Array of generated samples
pub fn make_heavy_tailed_distribution(
    n_samples: usize,
    distribution_name: &str,
    parameters: &[f64],
    random_state: Option<u64>,
) -> Result<Array1<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut samples = Array1::zeros(n_samples);

    match distribution_name {
        "student_t" => {
            if parameters.len() < 3 {
                return Err(SklearsError::InvalidInput(
                    "student_t requires 3 parameters: [degrees_of_freedom, location, scale]"
                        .to_string(),
                ));
            }
            let nu = parameters[0];
            let location = parameters[1];
            let scale = parameters[2];

            if nu <= 0.0 || scale <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "degrees_of_freedom and scale must be positive".to_string(),
                ));
            }

            // Use transformation method for t-distribution
            for i in 0..n_samples {
                // Generate chi-squared sample with nu degrees of freedom
                let chi_sq = (0..nu as usize)
                    .map(|_| {
                        let normal: f64 = rng.sample(StandardNormal);
                        normal * normal
                    })
                    .sum::<f64>();

                let normal: f64 = rng.sample(StandardNormal);
                let t_sample = normal / (chi_sq / nu).sqrt();
                samples[i] = location + scale * t_sample;
            }
        }

        "pareto" => {
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "pareto requires 2 parameters: [shape, scale]".to_string(),
                ));
            }
            let shape = parameters[0];
            let scale = parameters[1];

            if shape <= 0.0 || scale <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "shape and scale must be positive".to_string(),
                ));
            }

            for i in 0..n_samples {
                let u: f64 = rng.gen();
                samples[i] = scale * u.powf(-1.0 / shape);
            }
        }

        "cauchy" => {
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "cauchy requires 2 parameters: [location, scale]".to_string(),
                ));
            }
            let location = parameters[0];
            let scale = parameters[1];

            if scale <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "scale must be positive".to_string(),
                ));
            }

            for i in 0..n_samples {
                let u: f64 = rng.random_range(-PI / 2.0, PI / 2.0);
                samples[i] = location + scale * u.tan();
            }
        }

        "levy" => {
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "levy requires 2 parameters: [location, scale]".to_string(),
                ));
            }
            let location = parameters[0];
            let scale = parameters[1];

            if scale <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "scale must be positive".to_string(),
                ));
            }

            for i in 0..n_samples {
                let normal: f64 = rng.sample(StandardNormal);
                samples[i] = location + scale / (normal * normal);
            }
        }

        "log_normal" => {
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "log_normal requires 2 parameters: [mu, sigma]".to_string(),
                ));
            }
            let mu = parameters[0];
            let sigma = parameters[1];

            if sigma <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "sigma must be positive".to_string(),
                ));
            }

            let normal_dist = Normal::new(mu, sigma).unwrap();
            for i in 0..n_samples {
                let normal_sample: f64 = rng.sample(normal_dist);
                samples[i] = normal_sample.exp();
            }
        }

        "weibull" => {
            if parameters.len() < 2 {
                return Err(SklearsError::InvalidInput(
                    "weibull requires 2 parameters: [shape, scale]".to_string(),
                ));
            }
            let shape = parameters[0];
            let scale = parameters[1];

            if shape <= 0.0 || scale <= 0.0 {
                return Err(SklearsError::InvalidInput(
                    "shape and scale must be positive".to_string(),
                ));
            }

            for i in 0..n_samples {
                let u: f64 = rng.gen();
                samples[i] = scale * (-u.ln()).powf(1.0 / shape);
            }
        }

        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown heavy-tailed distribution: {}",
                distribution_name
            )));
        }
    }

    Ok(samples)
}