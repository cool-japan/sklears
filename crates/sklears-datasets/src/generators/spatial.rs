//! Spatial statistics and geostatistical data generators
//!
//! This module contains generators for spatial data including spatial point processes,
//! geostatistical data with spatial correlation, and geographic information datasets.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::Normal;
use sklears_core::error::{Result, SklearsError};

/// Generate spatial point process data
pub fn make_spatial_point_process(
    n_points: usize,
    region_bounds: (f64, f64, f64, f64), // (x_min, x_max, y_min, y_max)
    process_type: &str,                  // "poisson", "cluster", "regular"
    intensity: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_points == 0 {
        return Err(SklearsError::InvalidInput(
            "n_points must be positive".to_string(),
        ));
    }

    if intensity <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "intensity must be positive".to_string(),
        ));
    }

    let (x_min, x_max, y_min, y_max) = region_bounds;
    if x_min >= x_max || y_min >= y_max {
        return Err(SklearsError::InvalidInput(
            "Invalid region bounds".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let mut points = Array2::zeros((n_points, 2));
    let mut intensities = Array1::zeros(n_points);

    match process_type {
        "poisson" => {
            // Homogeneous Poisson process
            for i in 0..n_points {
                points[[i, 0]] = rng.gen_range(x_min..x_max);
                points[[i, 1]] = rng.gen_range(y_min..y_max);
                intensities[i] = intensity;
            }
        }
        "cluster" => {
            // Cluster process with random cluster centers
            let n_clusters = (n_points as f64 / 10.0).ceil() as usize;
            let cluster_std = ((x_max - x_min) + (y_max - y_min)) / 20.0;

            let mut cluster_centers = Vec::new();
            for _ in 0..n_clusters {
                let center_x = rng.gen_range(x_min..x_max);
                let center_y = rng.gen_range(y_min..y_max);
                cluster_centers.push((center_x, center_y));
            }

            let normal = Normal::new(0.0, cluster_std).unwrap();

            for i in 0..n_points {
                let cluster_idx = rng.gen_range(0..n_clusters);
                let (center_x, center_y) = cluster_centers[cluster_idx];

                let offset_x: f64 = rng.sample(normal);
                let offset_y: f64 = rng.sample(normal);

                points[[i, 0]] = (center_x + offset_x).max(x_min).min(x_max);
                points[[i, 1]] = (center_y + offset_y).max(y_min).min(y_max);
                intensities[i] = intensity * 2.0; // Higher intensity in clusters
            }
        }
        "regular" => {
            // Regular grid with small perturbations
            let grid_size = (n_points as f64).sqrt().ceil() as usize;
            let x_step = (x_max - x_min) / grid_size as f64;
            let y_step = (y_max - y_min) / grid_size as f64;

            let perturbation_std = (x_step + y_step) / 10.0;
            let normal = Normal::new(0.0, perturbation_std).unwrap();

            for i in 0..n_points {
                let grid_x = i % grid_size;
                let grid_y = i / grid_size;

                let base_x = x_min + grid_x as f64 * x_step;
                let base_y = y_min + grid_y as f64 * y_step;

                let perturb_x: f64 = rng.sample(normal);
                let perturb_y: f64 = rng.sample(normal);

                points[[i, 0]] = (base_x + perturb_x).max(x_min).min(x_max);
                points[[i, 1]] = (base_y + perturb_y).max(y_min).min(y_max);
                intensities[i] = intensity * 0.8; // Lower intensity for regular patterns
            }
        }
        _ => {
            return Err(SklearsError::InvalidInput(
                "process_type must be 'poisson', 'cluster', or 'regular'".to_string(),
            ));
        }
    }

    Ok((points, intensities))
}

/// Generate geostatistical data with spatial correlation
pub fn make_geostatistical_data(
    n_points: usize,
    region_bounds: (f64, f64, f64, f64),
    correlation_range: f64,
    nugget: f64,
    sill: f64,
    variogram_model: &str, // "exponential", "gaussian", "spherical"
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_points == 0 {
        return Err(SklearsError::InvalidInput(
            "n_points must be positive".to_string(),
        ));
    }

    if correlation_range <= 0.0 || sill <= 0.0 || nugget < 0.0 {
        return Err(SklearsError::InvalidInput(
            "correlation_range and sill must be positive, nugget must be non-negative".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let (x_min, x_max, y_min, y_max) = region_bounds;

    // Generate random locations
    let mut locations = Array2::zeros((n_points, 2));
    for i in 0..n_points {
        locations[[i, 0]] = rng.gen_range(x_min..x_max);
        locations[[i, 1]] = rng.gen_range(y_min..y_max);
    }

    // Create covariance matrix based on distances
    let mut covariance = Array2::zeros((n_points, n_points));

    for i in 0..n_points {
        for j in 0..n_points {
            let dx = locations[[i, 0]] - locations[[j, 0]];
            let dy = locations[[i, 1]] - locations[[j, 1]];
            let distance = (dx * dx + dy * dy).sqrt();

            let covariance_value = if i == j {
                sill + nugget
            } else {
                match variogram_model {
                    "exponential" => sill * (-distance / correlation_range).exp(),
                    "gaussian" => {
                        sill * (-(distance * distance) / (correlation_range * correlation_range))
                            .exp()
                    }
                    "spherical" => {
                        if distance >= correlation_range {
                            0.0
                        } else {
                            sill * (1.0 - 1.5 * distance / correlation_range
                                + 0.5 * (distance / correlation_range).powi(3))
                        }
                    }
                    _ => {
                        return Err(SklearsError::InvalidInput(
                            "variogram_model must be 'exponential', 'gaussian', or 'spherical'"
                                .to_string(),
                        ))
                    }
                }
            };

            covariance[[i, j]] = covariance_value;
        }
    }

    // Generate correlated random field using Cholesky decomposition
    let mut values = Array1::zeros(n_points);
    let normal = Normal::new(0.0, 1.0).unwrap();

    // Simple approach: approximate multivariate normal by weighted sum
    let mut random_values = Array1::zeros(n_points);
    for i in 0..n_points {
        random_values[i] = rng.sample(normal);
    }

    for i in 0..n_points {
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for j in 0..n_points {
            let weight = covariance[[i, j]];
            weighted_sum += weight * random_values[j];
            total_weight += weight;
        }

        values[i] = if total_weight > 0.0 {
            weighted_sum / total_weight.sqrt()
        } else {
            random_values[i]
        };
    }

    Ok((locations, values))
}

/// Geographic information configuration
#[derive(Debug, Clone)]
pub struct GeographicConfig {
    pub region_name: String,
    pub coordinate_system: String,   // "WGS84", "UTM", "Mercator"
    pub elevation_model: String,     // "flat", "hillside", "mountain", "valley"
    pub land_use_types: Vec<String>, // "urban", "forest", "water", "agriculture", etc.
}

/// Generate geographic information datasets
pub fn make_geographic_information_dataset(
    n_locations: usize,
    region_bounds: (f64, f64, f64, f64), // (lat_min, lat_max, lon_min, lon_max)
    config: GeographicConfig,
    include_demographics: bool,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<usize>, Array1<f64>)> {
    if n_locations == 0 {
        return Err(SklearsError::InvalidInput(
            "n_locations must be positive".to_string(),
        ));
    }

    let (lat_min, lat_max, lon_min, lon_max) = region_bounds;
    if lat_min >= lat_max || lon_min >= lon_max {
        return Err(SklearsError::InvalidInput(
            "Invalid region bounds".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    // Generate coordinates
    let mut coordinates = Array2::zeros((n_locations, 2));
    for i in 0..n_locations {
        coordinates[[i, 0]] = rng.gen_range(lat_min..lat_max); // latitude
        coordinates[[i, 1]] = rng.gen_range(lon_min..lon_max); // longitude
    }

    // Generate geographic features (elevation, slope, aspect, distance to water, etc.)
    let n_features = 8; // elevation, slope, aspect, distance_to_water, population_density, road_density, vegetation_index, temperature
    let mut geographic_features = Array2::zeros((n_locations, n_features));

    // Generate land use classification
    let mut land_use = Array1::zeros(n_locations);

    // Generate elevation data based on elevation model
    let mut elevations = Array1::zeros(n_locations);

    let normal = Normal::new(0.0, 1.0).unwrap();

    for i in 0..n_locations {
        let lat = coordinates[[i, 0]];
        let lon = coordinates[[i, 1]];

        // Generate elevation based on model
        let base_elevation = match config.elevation_model.as_str() {
            "flat" => 100.0 + 20.0 * rng.sample(normal),
            "hillside" => {
                let slope_factor = (lat - lat_min) / (lat_max - lat_min);
                200.0 + slope_factor * 300.0 + 50.0 * rng.sample(normal)
            }
            "mountain" => {
                // Gaussian mountain peaks
                let center_lat = (lat_min + lat_max) / 2.0;
                let center_lon = (lon_min + lon_max) / 2.0;
                let dist_from_center =
                    ((lat - center_lat).powi(2) + (lon - center_lon).powi(2)).sqrt();
                let max_dist =
                    ((lat_max - lat_min).powi(2) + (lon_max - lon_min).powi(2)).sqrt() / 2.0;
                let height_factor = (-2.0 * (dist_from_center / max_dist).powi(2)).exp();
                500.0 + height_factor * 2000.0 + 100.0 * rng.sample(normal)
            }
            "valley" => {
                // U-shaped valley
                let center_lat = (lat_min + lat_max) / 2.0;
                let dist_from_center = (lat - center_lat).abs();
                let valley_depth = (dist_from_center / (lat_max - lat_min)).powi(2);
                200.0 + valley_depth * 500.0 + 30.0 * rng.sample(normal)
            }
            _ => 300.0 + 200.0 * rng.sample(normal),
        };

        elevations[i] = base_elevation.max(0.0);
        geographic_features[[i, 0]] = elevations[i];

        // Generate slope (degrees)
        let slope = if i > 0 {
            let prev_elevation = elevations[i - 1];
            let elevation_diff = (elevations[i] - prev_elevation).abs();
            let distance = 0.001; // Approximate distance between points
            (elevation_diff / distance).atan().to_degrees().min(45.0)
        } else {
            rng.gen_range(0.0..15.0)
        };
        geographic_features[[i, 1]] = slope;

        // Generate aspect (compass direction of slope)
        let aspect = rng.gen_range(0.0..360.0);
        geographic_features[[i, 2]] = aspect;

        // Generate distance to water (km)
        let distance_to_water = rng.gen_range(0.1..50.0);
        geographic_features[[i, 3]] = distance_to_water;

        // Generate population density (people per km²) - correlated with distance to water
        let population_density = if distance_to_water < 5.0 {
            rng.gen_range(100.0..2000.0) // High density near water
        } else if distance_to_water < 20.0 {
            rng.gen_range(10.0..500.0) // Medium density
        } else {
            rng.gen_range(1.0..50.0) // Low density far from water
        };
        geographic_features[[i, 4]] = population_density;

        // Generate road density (km/km²) - correlated with population
        let road_density = population_density.ln() * 0.5 + rng.gen_range(0.0..2.0);
        geographic_features[[i, 5]] = road_density.max(0.0);

        // Generate vegetation index (NDVI-like, 0-1)
        let vegetation_index = if elevations[i] > 1500.0 {
            rng.gen_range(0.1..0.4) // Low vegetation at high elevation
        } else if distance_to_water < 10.0 {
            rng.gen_range(0.6..0.9) // High vegetation near water
        } else {
            rng.gen_range(0.3..0.7) // Medium vegetation
        };
        geographic_features[[i, 6]] = vegetation_index;

        // Generate temperature (°C) - correlated with elevation and latitude
        let base_temp = 20.0 - (elevations[i] / 1000.0) * 6.5; // Temperature lapse rate
        let latitude_effect = (lat - lat_min) / (lat_max - lat_min) * -10.0; // Cooler towards poles
        let temperature = base_temp + latitude_effect + 5.0 * rng.sample(normal);
        geographic_features[[i, 7]] = temperature;

        // Determine land use based on geographic features
        let land_use_type = if distance_to_water < 2.0 && vegetation_index > 0.7 {
            3 // water/wetland
        } else if population_density > 500.0 {
            0 // urban
        } else if vegetation_index > 0.6 && elevations[i] < 1000.0 {
            1 // forest
        } else if slope < 10.0 && vegetation_index > 0.4 {
            2 // agriculture
        } else if elevations[i] > 1500.0 {
            4 // bare/rock
        } else {
            5 // grassland
        };

        land_use[i] = land_use_type.min(config.land_use_types.len() - 1);
    }

    // Generate demographic features if requested
    let _demographics_features = if include_demographics {
        let mut demographics = Array2::zeros((n_locations, 5)); // age_median, income_median, education_level, employment_rate, housing_density

        for i in 0..n_locations {
            let pop_density = geographic_features[[i, 4]];

            // Median age (years) - tends to be higher in suburban areas
            let median_age = if pop_density > 1000.0 {
                rng.gen_range(28.0..42.0) // Urban: younger
            } else if pop_density > 100.0 {
                rng.gen_range(35.0..50.0) // Suburban: older
            } else {
                rng.gen_range(32.0..55.0) // Rural: varied
            };
            demographics[[i, 0]] = median_age;

            // Median income (thousands) - correlated with population density and road access
            let road_access_factor = geographic_features[[i, 5]] / 10.0;
            let income_base = if pop_density > 1000.0 {
                rng.gen_range(40.0..120.0) // Urban income range
            } else {
                rng.gen_range(25.0..80.0) // Rural income range
            };
            let median_income = income_base * (1.0 + road_access_factor * 0.5);
            demographics[[i, 1]] = median_income;

            // Education level (years of schooling)
            let education_level = median_income / 10.0 + rng.gen_range(8.0..18.0);
            demographics[[i, 2]] = education_level.min(20.0);

            // Employment rate (percentage)
            let employment_rate = if pop_density > 500.0 {
                rng.gen_range(85.0..96.0) // Higher employment in cities
            } else {
                rng.gen_range(70.0..90.0) // Lower in rural areas
            };
            demographics[[i, 3]] = employment_rate;

            // Housing density (units per km²)
            let housing_density = pop_density / rng.gen_range(2.0..4.0); // 2-4 people per housing unit
            demographics[[i, 4]] = housing_density;
        }

        demographics
    } else {
        Array2::zeros((n_locations, 0))
    };

    Ok((coordinates, geographic_features, land_use, elevations))
}

/// Generate spatial clustering datasets
pub fn make_spatial_clustering_dataset(
    n_points: usize,
    n_clusters: usize,
    region_bounds: (f64, f64, f64, f64),
    cluster_shape: String,   // "circular", "elliptical", "irregular", "linear"
    cluster_separation: f64, // Minimum distance between cluster centers
    noise_ratio: f64,        // Proportion of noise points (0.0 to 1.0)
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<usize>, Array1<f64>)> {
    if n_points == 0 || n_clusters == 0 {
        return Err(SklearsError::InvalidInput(
            "n_points and n_clusters must be positive".to_string(),
        ));
    }

    if cluster_separation <= 0.0 {
        return Err(SklearsError::InvalidInput(
            "cluster_separation must be positive".to_string(),
        ));
    }

    if !(0.0..=1.0).contains(&noise_ratio) {
        return Err(SklearsError::InvalidInput(
            "noise_ratio must be in [0, 1]".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let (x_min, x_max, y_min, y_max) = region_bounds;
    if x_min >= x_max || y_min >= y_max {
        return Err(SklearsError::InvalidInput(
            "Invalid region bounds".to_string(),
        ));
    }

    // Calculate number of noise points and cluster points
    let n_noise_points = (n_points as f64 * noise_ratio) as usize;
    let n_cluster_points = n_points - n_noise_points;
    let points_per_cluster = n_cluster_points / n_clusters;
    let extra_points = n_cluster_points % n_clusters;

    let mut points = Array2::zeros((n_points, 2));
    let mut labels = Array1::zeros(n_points);
    let mut cluster_densities = Array1::zeros(n_points);

    // Generate cluster centers with minimum separation
    let mut cluster_centers: Vec<(f64, f64)> = Vec::new();
    for _i in 0..n_clusters {
        let mut attempts = 0;
        loop {
            let center_x = rng.gen_range(x_min..x_max);
            let center_y = rng.gen_range(y_min..y_max);

            // Check minimum separation from existing centers
            let mut valid = true;
            for existing_center in &cluster_centers {
                let dist = ((center_x - existing_center.0).powi(2)
                    + (center_y - existing_center.1).powi(2))
                .sqrt();
                if dist < cluster_separation {
                    valid = false;
                    break;
                }
            }

            if valid || attempts > 100 {
                cluster_centers.push((center_x, center_y));
                break;
            }
            attempts += 1;
        }
    }

    let _normal = Normal::new(0.0, 1.0).unwrap();
    let mut point_idx = 0;

    // Generate points for each cluster
    for cluster_id in 0..n_clusters {
        let n_points_in_cluster =
            points_per_cluster + if cluster_id < extra_points { 1 } else { 0 };
        let (center_x, center_y) = cluster_centers[cluster_id];

        // Cluster size based on separation
        let cluster_size = cluster_separation / 4.0;

        for _ in 0..n_points_in_cluster {
            let (x, y) = match cluster_shape.as_str() {
                "circular" => {
                    // Circular cluster
                    let radius = rng.gen_range(0.0..cluster_size);
                    let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                    (
                        center_x + radius * angle.cos(),
                        center_y + radius * angle.sin(),
                    )
                }
                "elliptical" => {
                    // Elliptical cluster
                    let semi_major = cluster_size;
                    let semi_minor = cluster_size * 0.5;
                    let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                    let radius = rng.gen_range(0.0..1.0);

                    let x_offset = semi_major * radius * angle.cos();
                    let y_offset = semi_minor * radius * angle.sin();

                    // Random rotation
                    let rotation = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                    let cos_rot = rotation.cos();
                    let sin_rot = rotation.sin();

                    (
                        center_x + x_offset * cos_rot - y_offset * sin_rot,
                        center_y + x_offset * sin_rot + y_offset * cos_rot,
                    )
                }
                "irregular" => {
                    // Irregular cluster using multiple Gaussian components
                    let n_components = 3;
                    let component = rng.gen_range(0..n_components);
                    let offset_angle =
                        (component as f64) * 2.0 * std::f64::consts::PI / (n_components as f64);
                    let offset_radius = cluster_size * 0.3;

                    let component_center_x = center_x + offset_radius * offset_angle.cos();
                    let component_center_y = center_y + offset_radius * offset_angle.sin();

                    let std_dev = cluster_size * 0.4;
                    let gaussian = Normal::new(0.0, std_dev).unwrap();

                    (
                        component_center_x + rng.sample(gaussian),
                        component_center_y + rng.sample(gaussian),
                    )
                }
                "linear" => {
                    // Linear cluster
                    let length = cluster_size * 2.0;
                    let width = cluster_size * 0.3;

                    let t = rng.gen_range(-length / 2.0..length / 2.0);
                    let perpendicular_offset = rng.sample(Normal::new(0.0, width).unwrap());

                    // Random orientation
                    let angle = rng.gen_range(0.0..2.0 * std::f64::consts::PI);
                    let cos_angle = angle.cos();
                    let sin_angle = angle.sin();

                    (
                        center_x + t * cos_angle - perpendicular_offset * sin_angle,
                        center_y + t * sin_angle + perpendicular_offset * cos_angle,
                    )
                }
                _ => {
                    return Err(SklearsError::InvalidInput(
                        "cluster_shape must be 'circular', 'elliptical', 'irregular', or 'linear'"
                            .to_string(),
                    ));
                }
            };

            // Clamp to region bounds
            let clamped_x = x.max(x_min).min(x_max);
            let clamped_y = y.max(y_min).min(y_max);

            points[[point_idx, 0]] = clamped_x;
            points[[point_idx, 1]] = clamped_y;
            labels[point_idx] = cluster_id;

            // Calculate cluster density (inverse of distance from center)
            let dist_from_center =
                ((clamped_x - center_x).powi(2) + (clamped_y - center_y).powi(2)).sqrt();
            cluster_densities[point_idx] = 1.0 / (1.0 + dist_from_center);

            point_idx += 1;
        }
    }

    // Generate noise points
    for _ in 0..n_noise_points {
        points[[point_idx, 0]] = rng.gen_range(x_min..x_max);
        points[[point_idx, 1]] = rng.gen_range(y_min..y_max);
        labels[point_idx] = n_clusters; // Noise label
        cluster_densities[point_idx] = 0.1; // Low density for noise
        point_idx += 1;
    }

    Ok((points, labels, cluster_densities))
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_make_spatial_point_process() {
        let (points, intensities) =
            make_spatial_point_process(100, (0.0, 10.0, 0.0, 10.0), "poisson", 1.0, Some(42))
                .unwrap();

        assert_eq!(points.shape(), &[100, 2]);
        assert_eq!(intensities.len(), 100);

        // Check points are within bounds
        for i in 0..100 {
            assert!(
                points[[i, 0]] >= 0.0 && points[[i, 0]] <= 10.0,
                "X coordinates should be in bounds"
            );
            assert!(
                points[[i, 1]] >= 0.0 && points[[i, 1]] <= 10.0,
                "Y coordinates should be in bounds"
            );
        }

        // Check intensities are positive
        for &intensity in intensities.iter() {
            assert!(intensity > 0.0, "Intensities should be positive");
        }
    }

    #[test]
    fn test_spatial_point_process_cluster() {
        let (points, intensities) =
            make_spatial_point_process(50, (0.0, 5.0, 0.0, 5.0), "cluster", 2.0, Some(42)).unwrap();

        assert_eq!(points.shape(), &[50, 2]);
        assert_eq!(intensities.len(), 50);

        // Cluster process should have higher intensities
        let mean_intensity = intensities.mean().unwrap();
        assert!(
            mean_intensity > 2.0,
            "Cluster process should have higher mean intensity"
        );
    }

    #[test]
    fn test_make_geostatistical_data() {
        let (locations, values) = make_geostatistical_data(
            30,
            (0.0, 1.0, 0.0, 1.0),
            0.3,
            0.1,
            1.0,
            "exponential",
            Some(42),
        )
        .unwrap();

        assert_eq!(locations.shape(), &[30, 2]);
        assert_eq!(values.len(), 30);

        // Check locations are within bounds
        for i in 0..30 {
            assert!(
                locations[[i, 0]] >= 0.0 && locations[[i, 0]] <= 1.0,
                "X coordinates should be in bounds"
            );
            assert!(
                locations[[i, 1]] >= 0.0 && locations[[i, 1]] <= 1.0,
                "Y coordinates should be in bounds"
            );
        }

        // Check values have some variation
        let variance = values.var(0.0);
        assert!(variance > 0.0, "Values should have some variation");
    }

    #[test]
    fn test_geostatistical_data_invalid_input() {
        // Invalid correlation_range
        assert!(make_geostatistical_data(
            30,
            (0.0, 1.0, 0.0, 1.0),
            0.0,
            0.1,
            1.0,
            "exponential",
            Some(42)
        )
        .is_err());

        // Invalid variogram model
        assert!(make_geostatistical_data(
            30,
            (0.0, 1.0, 0.0, 1.0),
            0.3,
            0.1,
            1.0,
            "invalid",
            Some(42)
        )
        .is_err());

        // Invalid n_points
        assert!(make_geostatistical_data(
            0,
            (0.0, 1.0, 0.0, 1.0),
            0.3,
            0.1,
            1.0,
            "exponential",
            Some(42)
        )
        .is_err());
    }

    #[test]
    fn test_make_geographic_information_dataset() {
        let config = GeographicConfig {
            region_name: "Test Region".to_string(),
            coordinate_system: "WGS84".to_string(),
            elevation_model: "mountain".to_string(),
            land_use_types: vec![
                "urban".to_string(),
                "forest".to_string(),
                "agriculture".to_string(),
                "water".to_string(),
                "bare".to_string(),
                "grassland".to_string(),
            ],
        };

        let (coordinates, geographic_features, land_use, elevations) =
            make_geographic_information_dataset(
                50,
                (37.0, 38.0, -122.5, -121.5),
                config,
                false,
                Some(42),
            )
            .unwrap();

        assert_eq!(coordinates.shape(), &[50, 2]);
        assert_eq!(geographic_features.shape(), &[50, 8]);
        assert_eq!(land_use.len(), 50);
        assert_eq!(elevations.len(), 50);

        // Check coordinates are within bounds
        for i in 0..50 {
            assert!(
                coordinates[[i, 0]] >= 37.0 && coordinates[[i, 0]] <= 38.0,
                "Latitude should be in bounds"
            );
            assert!(
                coordinates[[i, 1]] >= -122.5 && coordinates[[i, 1]] <= -121.5,
                "Longitude should be in bounds"
            );
        }

        // Check elevations are positive
        for &elevation in elevations.iter() {
            assert!(elevation >= 0.0, "Elevations should be non-negative");
        }

        // Check land use labels are valid
        for &land_use_type in land_use.iter() {
            assert!(land_use_type < 6, "Land use types should be valid");
        }

        // Check geographic features have reasonable ranges
        for i in 0..50 {
            let slope = geographic_features[[i, 1]];
            let aspect = geographic_features[[i, 2]];
            let distance_to_water = geographic_features[[i, 3]];
            let vegetation_index = geographic_features[[i, 6]];

            assert!(
                slope >= 0.0 && slope <= 45.0,
                "Slope should be in [0, 45] degrees"
            );
            assert!(
                aspect >= 0.0 && aspect <= 360.0,
                "Aspect should be in [0, 360] degrees"
            );
            assert!(
                distance_to_water > 0.0,
                "Distance to water should be positive"
            );
            assert!(
                vegetation_index >= 0.0 && vegetation_index <= 1.0,
                "Vegetation index should be in [0, 1]"
            );
        }
    }

    #[test]
    fn test_make_spatial_clustering_dataset() {
        let (points, labels, cluster_densities) = make_spatial_clustering_dataset(
            100,
            4,
            (0.0, 10.0, 0.0, 10.0),
            "circular".to_string(),
            2.0,
            0.1,
            Some(42),
        )
        .unwrap();

        assert_eq!(points.shape(), &[100, 2]);
        assert_eq!(labels.len(), 100);
        assert_eq!(cluster_densities.len(), 100);

        // Check points are within bounds
        for i in 0..100 {
            assert!(
                points[[i, 0]] >= 0.0 && points[[i, 0]] <= 10.0,
                "X coordinates should be in bounds"
            );
            assert!(
                points[[i, 1]] >= 0.0 && points[[i, 1]] <= 10.0,
                "Y coordinates should be in bounds"
            );
        }

        // Check cluster labels are valid (0-3 for clusters, 4 for noise)
        for &label in labels.iter() {
            assert!(label <= 4, "Cluster labels should be valid");
        }

        // Check cluster densities are positive
        for &density in cluster_densities.iter() {
            assert!(density > 0.0, "Cluster densities should be positive");
        }

        // Check that we have approximately the right number of noise points
        let noise_count = labels.iter().filter(|&&l| l == 4).count();
        assert!(
            noise_count >= 8 && noise_count <= 12,
            "Should have ~10% noise points"
        );
    }

    #[test]
    fn test_spatial_clustering_different_shapes() {
        let shapes = vec!["circular", "elliptical", "irregular", "linear"];

        for shape in shapes {
            let result = make_spatial_clustering_dataset(
                50,
                3,
                (0.0, 5.0, 0.0, 5.0),
                shape.to_string(),
                1.5,
                0.0,
                Some(42),
            );

            assert!(result.is_ok(), "Shape {} should be valid", shape);

            let (points, labels, _) = result.unwrap();
            assert_eq!(points.shape(), &[50, 2]);
            assert_eq!(labels.len(), 50);

            // All labels should be 0, 1, or 2 (no noise)
            for &label in labels.iter() {
                assert!(label < 3, "Labels should be valid cluster IDs");
            }
        }
    }

    #[test]
    fn test_spatial_clustering_invalid_inputs() {
        // Invalid cluster shape
        assert!(make_spatial_clustering_dataset(
            50,
            3,
            (0.0, 5.0, 0.0, 5.0),
            "invalid_shape".to_string(),
            1.5,
            0.0,
            Some(42),
        )
        .is_err());

        // Invalid noise ratio
        assert!(make_spatial_clustering_dataset(
            50,
            3,
            (0.0, 5.0, 0.0, 5.0),
            "circular".to_string(),
            1.5,
            1.5,
            Some(42),
        )
        .is_err());

        // Invalid cluster separation
        assert!(make_spatial_clustering_dataset(
            50,
            3,
            (0.0, 5.0, 0.0, 5.0),
            "circular".to_string(),
            0.0,
            0.1,
            Some(42),
        )
        .is_err());
    }
}
