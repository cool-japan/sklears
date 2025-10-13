//! Manifold and spatial pattern dataset generators
//!
//! This module provides generators for geometric manifolds, spatial point patterns,
//! and spatial statistical data commonly used in spatial analysis and manifold learning.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::{Random, rng};
use scirs2_core::random::distributions::{Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

/// Trait for manifold generators
pub trait ManifoldGenerator {
    fn generate_point(&self, parameters: &[f64]) -> Array1<f64>;
    fn parameter_bounds(&self) -> Vec<(f64, f64)>;
}

/// Generate samples from a custom manifold
///
/// Creates samples lying on a manifold defined by a custom generator function.
/// This is useful for testing manifold learning algorithms on known structures.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `generator`: Custom manifold generator implementing ManifoldGenerator trait
/// - `noise_level`: Amount of noise to add to manifold samples
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Array of samples on the specified manifold
pub fn make_custom_manifold<T: ManifoldGenerator>(
    n_samples: usize,
    generator: &T,
    noise_level: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let bounds = generator.parameter_bounds();
    let n_params = bounds.len();

    if n_params == 0 {
        return Err(SklearsError::InvalidInput(
            "Manifold generator must have at least one parameter".to_string(),
        ));
    }

    // Generate first sample to determine embedding dimension
    let mut params = vec![0.0; n_params];
    for (i, &(min_val, max_val)) in bounds.iter().enumerate() {
        params[i] = rng.gen_range(min_val..max_val);
    }
    let first_point = generator.generate_point(&params);
    let embedding_dim = first_point.len();

    let mut samples = Array2::zeros((n_samples, embedding_dim));

    // Generate all samples
    for i in 0..n_samples {
        // Generate random parameters within bounds
        for (j, &(min_val, max_val)) in bounds.iter().enumerate() {
            params[j] = rng.gen_range(min_val..max_val);
        }

        let mut point = generator.generate_point(&params);

        // Add noise if specified
        if noise_level > 0.0 {
            for j in 0..embedding_dim {
                let noise = rng.sample(Normal::new(0.0, noise_level).unwrap());
                point[j] += noise;
            }
        }

        for j in 0..embedding_dim {
            samples[[i, j]] = point[j];
        }
    }

    Ok(samples)
}

/// Generate samples on an n-dimensional sphere
///
/// Creates samples uniformly distributed on the surface of an n-dimensional sphere.
/// This is a fundamental manifold used in many manifold learning benchmarks.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_dim`: Dimension of the sphere (n_dim=2 gives unit circle, n_dim=3 gives unit sphere)
/// - `radius`: Radius of the sphere
/// - `noise_level`: Amount of noise to add
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Array of samples on the n-sphere
pub fn make_n_sphere(
    n_samples: usize,
    n_dim: usize,
    radius: f64,
    noise_level: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 || n_dim == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_dim must be positive".to_string(),
        ));
    }

    if radius <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "radius must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut samples = Array2::zeros((n_samples, n_dim));

    for i in 0..n_samples {
        // Generate point from normal distribution
        let mut point = Array1::zeros(n_dim);
        for j in 0..n_dim {
            point[j] = rng.sample(StandardNormal);
        }

        // Normalize to unit sphere
        let norm = point.iter().map(|&x| x * x).sum::<f64>().sqrt();
        for j in 0..n_dim {
            point[j] = radius * point[j] / norm;

            // Add noise if specified
            if noise_level > 0.0 {
                let noise = rng.sample(Normal::new(0.0, noise_level).unwrap());
                point[j] += noise;
            }

            samples[[i, j]] = point[j];
        }
    }

    Ok(samples)
}

/// Generate samples on an n-dimensional torus
///
/// Creates samples on the surface of an n-dimensional torus embedded in higher-dimensional space.
/// This is useful for testing manifold learning on curved manifolds with interesting topology.
///
/// # Parameters
/// - `n_samples`: Number of samples to generate
/// - `n_dim`: Intrinsic dimension of the torus
/// - `major_radius`: Major radius of the torus
/// - `minor_radius`: Minor radius of the torus
/// - `noise_level`: Amount of noise to add
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Array of samples on the n-torus
pub fn make_n_torus(
    n_samples: usize,
    n_dim: usize,
    major_radius: f64,
    minor_radius: f64,
    noise_level: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 || n_dim == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_dim must be positive".to_string(),
        ));
    }

    if major_radius <= 0.0 || minor_radius <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "radii must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Embedding dimension is 2 * n_dim for n-torus
    let embedding_dim = 2 * n_dim;
    let mut samples = Array2::zeros((n_samples, embedding_dim));

    for i in 0..n_samples {
        // Generate angles for each dimension
        let mut angles = vec![0.0; n_dim];
        for j in 0..n_dim {
            angles[j] = rng.gen_range(0.0..2.0 * PI);
        }

        // Map to embedding space (simplified n-torus embedding)
        for j in 0..n_dim {
            let cos_angle = angles[j].cos();
            let sin_angle = angles[j].sin();

            samples[[i, j * 2]] = (major_radius + minor_radius * cos_angle) * angles[j].cos();
            samples[[i, j * 2 + 1]] = (major_radius + minor_radius * cos_angle) * sin_angle;

            // Add noise if specified
            if noise_level > 0.0 {
                let noise1 = rng.sample(Normal::new(0.0, noise_level).unwrap());
                let noise2 = rng.sample(Normal::new(0.0, noise_level).unwrap());
                samples[[i, j * 2]] += noise1;
                samples[[i, j * 2 + 1]] += noise2;
            }
        }
    }

    Ok(samples)
}

/// Generate spatial point pattern with specified characteristics
///
/// Creates spatial point patterns following various statistical processes
/// commonly used in spatial statistics and ecology.
///
/// # Parameters
/// - `n_points`: Number of points to generate
/// - `pattern_type`: Type of spatial pattern ("random", "clustered", "regular", "inhibited")
/// - `intensity`: Average intensity (points per unit area)
/// - `clustering_parameter`: Parameter controlling clustering/regularity
/// - `domain_size`: Size of the spatial domain (square domain of this size)
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Array of 2D spatial coordinates
pub fn make_spatial_point_pattern(
    n_points: usize,
    pattern_type: &str,
    intensity: f64,
    clustering_parameter: f64,
    domain_size: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_points == 0 {
        return Err(SklearsError::InvalidInput(
            "n_points must be positive".to_string(),
        ));
    }

    if intensity <= 0.0 || domain_size <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "intensity and domain_size must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    let mut points = Array2::zeros((n_points, 2));

    match pattern_type {
        "random" => {
            // Complete spatial randomness (Poisson process)
            for i in 0..n_points {
                points[[i, 0]] = rng.gen_range(0.0..domain_size);
                points[[i, 1]] = rng.gen_range(0.0..domain_size);
            }
        }

        "clustered" => {
            // Clustered pattern using cluster centers
            let n_clusters = (n_points as f64 / clustering_parameter).max(1.0) as usize;
            let points_per_cluster = n_points / n_clusters;

            // Generate cluster centers
            let mut cluster_centers = Array2::zeros((n_clusters, 2));
            for i in 0..n_clusters {
                cluster_centers[[i, 0]] = rng.gen_range(0.0..domain_size);
                cluster_centers[[i, 1]] = rng.gen_range(0.0..domain_size);
            }

            // Generate points around cluster centers
            for i in 0..n_points {
                let cluster_idx = i / points_per_cluster;
                let cluster_idx = cluster_idx.min(n_clusters - 1);

                let center_x = cluster_centers[[cluster_idx, 0]];
                let center_y = cluster_centers[[cluster_idx, 1]];

                // Add points with Gaussian dispersion around center
                let cluster_std = domain_size / (10.0 * clustering_parameter.sqrt());
                let dx = rng.sample(Normal::new(0.0, cluster_std).unwrap());
                let dy = rng.sample(Normal::new(0.0, cluster_std).unwrap());

                points[[i, 0]] = (center_x + dx).clamp(0.0, domain_size);
                points[[i, 1]] = (center_y + dy).clamp(0.0, domain_size);
            }
        }

        "regular" => {
            // Regular pattern using grid with perturbation
            let grid_size = (n_points as f64).sqrt() as usize;
            let cell_size = domain_size / grid_size as f64;
            let perturbation = cell_size * (1.0 - clustering_parameter) / 4.0;

            for i in 0..n_points {
                let row = i / grid_size;
                let col = i % grid_size;

                let base_x = (col as f64 + 0.5) * cell_size;
                let base_y = (row as f64 + 0.5) * cell_size;

                let dx = rng.gen_range(-perturbation..perturbation);
                let dy = rng.gen_range(-perturbation..perturbation);

                points[[i, 0]] = (base_x + dx).clamp(0.0, domain_size);
                points[[i, 1]] = (base_y + dy).clamp(0.0, domain_size);
            }
        }

        "inhibited" => {
            // Inhibited pattern using simple sequential inhibition
            let min_distance = clustering_parameter * domain_size / (n_points as f64).sqrt();
            let mut placed_points = Vec::new();

            for i in 0..n_points {
                let mut attempts = 0;
                let max_attempts = 1000;

                loop {
                    let x = rng.gen_range(0.0..domain_size);
                    let y = rng.gen_range(0.0..domain_size);

                    // Check minimum distance to existing points
                    let mut valid = true;
                    for &(px, py) in &placed_points {
                        let dist = ((x - px).powi(2) + (y - py).powi(2)).sqrt();
                        if dist < min_distance {
                            valid = false;
                            break;
                        }
                    }

                    if valid || attempts > max_attempts {
                        points[[i, 0]] = x;
                        points[[i, 1]] = y;
                        placed_points.push((x, y));
                        break;
                    }

                    attempts += 1;
                }
            }
        }

        _ => {
            return Err(SklearsError::InvalidInput(format!(
                "Unknown pattern_type: {}. Use 'random', 'clustered', 'regular', or 'inhibited'",
                pattern_type
            )));
        }
    }

    Ok(points)
}

/// Generate geostatistical data with spatial correlation
///
/// Creates spatially correlated data following a specified covariance structure.
/// This is useful for testing spatial interpolation and geostatistical methods.
///
/// # Parameters
/// - `n_points`: Number of spatial locations
/// - `domain_size`: Size of the spatial domain
/// - `correlation_range`: Range parameter for spatial correlation
/// - `nugget`: Nugget effect (measurement error variance)
/// - `sill`: Sill parameter (total variance)
/// - `correlation_function`: Type of correlation function ("exponential", "gaussian", "spherical")
/// - `random_state`: Random seed for reproducibility
///
/// # Returns
/// Tuple of (spatial_coordinates, correlated_values)
pub fn make_geostatistical_data(
    n_points: usize,
    domain_size: f64,
    correlation_range: f64,
    nugget: f64,
    sill: f64,
    correlation_function: &str,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_points == 0 {
        return Err(SklearsError::InvalidInput(
            "n_points must be positive".to_string(),
        ));
    }

    if domain_size <= 0.0 || correlation_range <= 0.0 || sill <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "domain_size, correlation_range, and sill must be positive".to_string(),
        ));
    }

    let mut rng = Random::from_seed(random_state.unwrap_or_else(|| rng().gen()));

    // Generate spatial coordinates
    let mut coordinates = Array2::zeros((n_points, 2));
    for i in 0..n_points {
        coordinates[[i, 0]] = rng.gen_range(0.0..domain_size);
        coordinates[[i, 1]] = rng.gen_range(0.0..domain_size);
    }

    // Calculate correlation matrix
    let mut correlation_matrix = Array2::zeros((n_points, n_points));
    for i in 0..n_points {
        for j in 0..n_points {
            let dx = coordinates[[i, 0]] - coordinates[[j, 0]];
            let dy = coordinates[[i, 1]] - coordinates[[j, 1]];
            let distance = (dx * dx + dy * dy).sqrt();

            let correlation = match correlation_function {
                "exponential" => {
                    if i == j {
                        sill
                    } else {
                        (sill - nugget) * (-distance / correlation_range).exp() +
                        if i == j { nugget } else { 0.0 }
                    }
                }
                "gaussian" => {
                    if i == j {
                        sill
                    } else {
                        (sill - nugget) * (-(distance / correlation_range).powi(2)).exp() +
                        if i == j { nugget } else { 0.0 }
                    }
                }
                "spherical" => {
                    if i == j {
                        sill
                    } else if distance <= correlation_range {
                        let h_over_a = distance / correlation_range;
                        (sill - nugget) * (1.0 - 1.5 * h_over_a + 0.5 * h_over_a.powi(3))
                    } else {
                        nugget
                    }
                }
                _ => {
                    return Err(SklearsError::InvalidInput(format!(
                        "Unknown correlation_function: {}. Use 'exponential', 'gaussian', or 'spherical'",
                        correlation_function
                    )));
                }
            };

            correlation_matrix[[i, j]] = correlation;
        }
    }

    // Generate correlated values using Cholesky decomposition (simplified)
    let mut values = Array1::zeros(n_points);
    for i in 0..n_points {
        values[i] = rng.sample(StandardNormal);
    }

    // Apply correlation structure (simplified approach)
    let mut correlated_values = Array1::zeros(n_points);
    for i in 0..n_points {
        let mut sum = 0.0;
        for j in 0..=i {
            sum += correlation_matrix[[i, j]].sqrt() * values[j];
        }
        correlated_values[i] = sum / (i + 1) as f64; // Simplified normalization
    }

    Ok((coordinates, correlated_values))
}