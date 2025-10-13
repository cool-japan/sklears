//! Manifold and geometric data generators
//!
//! This module provides functions for generating data on various manifolds
//! and geometric structures useful for testing manifold learning algorithms.

use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Gamma, Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

pub fn make_swiss_roll(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate 3D Swiss roll manifold
    let mut x = Array2::zeros((n_samples, 3));
    let mut t = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Parameter t controls the angle of the roll
        let t_val = 1.5 * PI * (1.0 + 2.0 * rng.gen::<f64>());
        t[i] = t_val;

        // Swiss roll equations
        x[[i, 0]] = t_val * t_val.cos(); // x = t * cos(t)
        x[[i, 1]] = 21.0 * rng.gen::<f64>(); // y = uniform [0, 21]
        x[[i, 2]] = t_val * t_val.sin(); // z = t * sin(t)

        // Add noise if specified
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            for j in 0..3 {
                x[[i, j]] += rng.sample(noise_dist);
            }
        }
    }

    Ok((x, t))
}

pub fn make_s_curve(
    n_samples: usize,
    noise: f64,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<f64>)> {
    if n_samples == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    // Generate 3D S-curve manifold
    let mut x = Array2::zeros((n_samples, 3));
    let mut t = Array1::zeros(n_samples);

    for i in 0..n_samples {
        // Parameters for S-curve
        let t_val = 3.0 * PI * (rng.gen::<f64>() - 0.5); // t in [-1.5π, 1.5π]
        t[i] = t_val;

        // S-curve equations
        x[[i, 0]] = t_val.sin(); // x = sin(t)
        x[[i, 1]] = 2.0 * rng.gen::<f64>(); // y = uniform [0, 2]
        x[[i, 2]] = t_val.cos().signum() * (t_val.cos().abs().powf(0.5)); // z = sign(cos(t)) * sqrt(|cos(t)|)

        // Add noise if specified
        if noise > 0.0 {
            let noise_dist = Normal::new(0.0, noise).unwrap();
            for j in 0..3 {
                x[[i, j]] += rng.sample(noise_dist);
            }
        }
    }

    Ok((x, t))
}

pub fn make_biclusters(
    shape: (usize, usize),
    n_clusters: usize,
    noise: f64,
    minval: f64,
    maxval: f64,
    shuffle: bool,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<usize>, Array1<usize>)> {
    if shape.0 == 0 || shape.1 == 0 {
        return Err(SklearsError::InvalidInput(
            "shape dimensions must be positive".to_string(),
        ));
    }

    if n_clusters == 0 {
        return Err(SklearsError::InvalidInput(
            "n_clusters must be positive".to_string(),
        ));
    }

    if minval >= maxval {
        return Err(SklearsError::InvalidInput(
            "minval must be < maxval".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let (n_rows, n_cols) = shape;

    // Initialize data matrix with noise
    let mut data = Array2::zeros((n_rows, n_cols));
    if noise > 0.0 {
        let noise_dist = Normal::new(0.0, noise).unwrap();
        for i in 0..n_rows {
            for j in 0..n_cols {
                data[[i, j]] = rng.sample(noise_dist);
            }
        }
    }

    // Create bicluster assignments
    let mut row_labels = Array1::zeros(n_rows);
    let mut col_labels = Array1::zeros(n_cols);

    // Assign rows and columns to clusters
    let rows_per_cluster = n_rows / n_clusters;
    let cols_per_cluster = n_cols / n_clusters;

    for cluster in 0..n_clusters {
        let row_start = cluster * rows_per_cluster;
        let row_end = if cluster == n_clusters - 1 {
            n_rows
        } else {
            (cluster + 1) * rows_per_cluster
        };
        let col_start = cluster * cols_per_cluster;
        let col_end = if cluster == n_clusters - 1 {
            n_cols
        } else {
            (cluster + 1) * cols_per_cluster
        };

        // Assign cluster labels
        for i in row_start..row_end {
            row_labels[i] = cluster;
        }
        for j in col_start..col_end {
            col_labels[j] = cluster;
        }

        // Generate bicluster values
        let cluster_value = rng.gen_range(minval..maxval);
        for i in row_start..row_end {
            for j in col_start..col_end {
                data[[i, j]] += cluster_value;
            }
        }
    }

    // Shuffle if requested
    if shuffle {
        // Shuffle rows
        let mut row_indices: Vec<usize> = (0..n_rows).collect();
        for i in (1..row_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            row_indices.swap(i, j);
        }

        // Shuffle columns
        let mut col_indices: Vec<usize> = (0..n_cols).collect();
        for i in (1..col_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            col_indices.swap(i, j);
        }

        // Apply shuffling to data and labels
        let data_copy = data.clone();
        let row_labels_copy = row_labels.clone();
        let col_labels_copy = col_labels.clone();

        for (new_i, &old_i) in row_indices.iter().enumerate() {
            row_labels[new_i] = row_labels_copy[old_i];
            for (new_j, &old_j) in col_indices.iter().enumerate() {
                data[[new_i, new_j]] = data_copy[[old_i, old_j]];
            }
        }

        for (new_j, &old_j) in col_indices.iter().enumerate() {
            col_labels[new_j] = col_labels_copy[old_j];
        }
    }

    Ok((data, row_labels, col_labels))
}

pub fn make_checkerboard(
    shape: (usize, usize),
    n_clusters: (usize, usize),
    noise: f64,
    minval: f64,
    maxval: f64,
    shuffle: bool,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array1<usize>, Array1<usize>)> {
    if shape.0 == 0 || shape.1 == 0 {
        return Err(SklearsError::InvalidInput(
            "shape dimensions must be positive".to_string(),
        ));
    }

    if n_clusters.0 == 0 || n_clusters.1 == 0 {
        return Err(SklearsError::InvalidInput(
            "n_clusters dimensions must be positive".to_string(),
        ));
    }

    if minval >= maxval {
        return Err(SklearsError::InvalidInput(
            "minval must be < maxval".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut rand::thread_rng())
    };

    let (n_rows, n_cols) = shape;
    let (n_row_clusters, n_col_clusters) = n_clusters;

    // Initialize data matrix with noise
    let mut data = Array2::zeros((n_rows, n_cols));
    if noise > 0.0 {
        let noise_dist = Normal::new(0.0, noise).unwrap();
        for i in 0..n_rows {
            for j in 0..n_cols {
                data[[i, j]] = rng.sample(noise_dist);
            }
        }
    }

    // Create cluster assignments
    let mut row_labels = Array1::zeros(n_rows);
    let mut col_labels = Array1::zeros(n_cols);

    // Calculate cluster sizes
    let rows_per_cluster = n_rows / n_row_clusters;
    let cols_per_cluster = n_cols / n_col_clusters;

    // Assign row and column labels
    for i in 0..n_rows {
        row_labels[i] = std::cmp::min(i / rows_per_cluster, n_row_clusters - 1);
    }

    for j in 0..n_cols {
        col_labels[j] = std::cmp::min(j / cols_per_cluster, n_col_clusters - 1);
    }

    // Create checkerboard pattern
    for i in 0..n_rows {
        for j in 0..n_cols {
            let row_cluster = row_labels[i];
            let col_cluster = col_labels[j];

            // Checkerboard pattern: alternate values based on sum of cluster indices
            let cluster_value = if (row_cluster + col_cluster) % 2 == 0 {
                rng.gen_range(minval..maxval)
            } else {
                rng.gen_range(minval..maxval) * -1.0 // Opposite pattern
            };

            data[[i, j]] += cluster_value;
        }
    }

    // Shuffle if requested
    if shuffle {
        // Shuffle rows
        let mut row_indices: Vec<usize> = (0..n_rows).collect();
        for i in (1..row_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            row_indices.swap(i, j);
        }

        // Shuffle columns
        let mut col_indices: Vec<usize> = (0..n_cols).collect();
        for i in (1..col_indices.len()).rev() {
            let j = rng.gen_range(0..=i);
            col_indices.swap(i, j);
        }

        // Apply shuffling to data and labels
        let data_copy = data.clone();
        let row_labels_copy = row_labels.clone();
        let col_labels_copy = col_labels.clone();

        for (new_i, &old_i) in row_indices.iter().enumerate() {
            row_labels[new_i] = row_labels_copy[old_i];
            for (new_j, &old_j) in col_indices.iter().enumerate() {
                data[[new_i, new_j]] = data_copy[[old_i, old_j]];
            }
        }

        for (new_j, &old_j) in col_indices.iter().enumerate() {
            col_labels[new_j] = col_labels_copy[old_j];
        }
    }

    Ok((data, row_labels, col_labels))
}

