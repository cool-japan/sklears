//! Matrix and linear algebra data generators
//!
//! This module provides functions for generating matrices with specific
//! properties useful for testing linear algebra algorithms.

use scirs2_core::ndarray::{s, Array1, Array2, Array3};
use scirs2_core::random::prelude::*;
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::{Gamma, Normal, StandardNormal};
use sklears_core::error::{Result, SklearsError};
use std::f64::consts::PI;

pub fn make_low_rank_matrix(
    n_samples: usize,
    n_features: usize,
    effective_rank: usize,
    tail_strength: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_samples == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples and n_features must be positive".to_string(),
        ));
    }

    if effective_rank == 0 || effective_rank > std::cmp::min(n_samples, n_features) {
        return Err(SklearsError::InvalidInput(
            "effective_rank must be positive and <= min(n_samples, n_features)".to_string(),
        ));
    }

    if tail_strength < 0.0 || tail_strength > 1.0 {
        return Err(SklearsError::InvalidInput(
            "tail_strength must be between 0 and 1".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = StandardNormal;
    let min_dim = std::cmp::min(n_samples, n_features);

    // Generate left and right matrices for SVD-like decomposition
    let mut u = Array2::zeros((n_samples, effective_rank));
    let mut v = Array2::zeros((effective_rank, n_features));

    // Fill U matrix (left singular vectors)
    for i in 0..n_samples {
        for j in 0..effective_rank {
            u[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Fill V matrix (right singular vectors)
    for i in 0..effective_rank {
        for j in 0..n_features {
            v[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Generate singular values with exponential decay
    let mut singular_values = Array1::zeros(effective_rank);
    for i in 0..effective_rank {
        let decay_factor = (-tail_strength * i as f64).exp();
        singular_values[i] = 10.0 * decay_factor + 0.1; // Ensure positive values
    }

    // Construct the low-rank matrix as U * diag(singular_values) * V
    let mut result = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..effective_rank {
                sum += u[[i, k]] * singular_values[k] * v[[k, j]];
            }
            result[[i, j]] = sum;
        }
    }

    Ok(result)
}

pub fn make_sparse_coded_signal(
    n_samples: usize,
    n_components: usize,
    n_features: usize,
    n_nonzero_coefs: usize,
    random_state: Option<u64>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    if n_samples == 0 || n_components == 0 || n_features == 0 {
        return Err(SklearsError::InvalidInput(
            "n_samples, n_components, and n_features must be positive".to_string(),
        ));
    }

    if n_nonzero_coefs == 0 || n_nonzero_coefs > n_components {
        return Err(SklearsError::InvalidInput(
            "n_nonzero_coefs must be positive and <= n_components".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = StandardNormal;

    // Generate dictionary (n_features x n_components)
    let mut dictionary = Array2::zeros((n_features, n_components));
    for i in 0..n_features {
        for j in 0..n_components {
            dictionary[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Normalize dictionary columns
    for j in 0..n_components {
        let mut col_norm = 0.0;
        for i in 0..n_features {
            col_norm += dictionary[[i, j]] * dictionary[[i, j]];
        }
        col_norm = col_norm.sqrt();
        if col_norm > 1e-12 {
            for i in 0..n_features {
                dictionary[[i, j]] /= col_norm;
            }
        }
    }

    // Generate sparse code (n_samples x n_components)
    let mut code = Array2::zeros((n_samples, n_components));
    for i in 0..n_samples {
        // Select n_nonzero_coefs random components
        let mut available_components: Vec<usize> = (0..n_components).collect();
        for j in (n_nonzero_coefs..available_components.len()).rev() {
            let k = rng.gen_range(0..j + 1);
            available_components.swap(j, k);
        }

        // Set non-zero coefficients
        for &comp_idx in available_components.iter().take(n_nonzero_coefs) {
            code[[i, comp_idx]] = rng.sample::<f64, _>(normal);
        }
    }

    // Generate signal: X = code * dictionary^T
    let mut signal = Array2::zeros((n_samples, n_features));
    for i in 0..n_samples {
        for j in 0..n_features {
            let mut sum = 0.0;
            for k in 0..n_components {
                sum += code[[i, k]] * dictionary[[j, k]];
            }
            signal[[i, j]] = sum;
        }
    }

    Ok((signal, code, dictionary))
}

pub fn make_sparse_spd_matrix(
    n_dim: usize,
    alpha: f64,
    norm_diag: bool,
    smallest_coef: f64,
    largest_coef: f64,
    random_state: Option<u64>,
) -> Result<Array2<f64>> {
    if n_dim == 0 {
        return Err(SklearsError::InvalidInput(
            "n_dim must be positive".to_string(),
        ));
    }

    if alpha < 0.0 || alpha > 1.0 {
        return Err(SklearsError::InvalidInput(
            "alpha must be between 0 and 1".to_string(),
        ));
    }

    if smallest_coef >= largest_coef {
        return Err(SklearsError::InvalidInput(
            "smallest_coef must be < largest_coef".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    // Start with identity matrix
    let mut matrix = Array2::eye(n_dim);

    // Determine number of off-diagonal non-zero elements
    let n_off_diag = ((n_dim * (n_dim - 1)) as f64 * alpha / 2.0) as usize;

    // Generate random off-diagonal positions
    let mut positions = Vec::new();
    for i in 0..n_dim {
        for j in (i + 1)..n_dim {
            positions.push((i, j));
        }
    }

    // Randomly select positions for non-zero elements
    for _ in 0..n_off_diag {
        if !positions.is_empty() {
            let idx = rng.gen_range(0..positions.len());
            let (i, j) = positions.swap_remove(idx);

            // Generate random coefficient in the specified range
            let coef = rng.gen_range(smallest_coef..largest_coef);
            matrix[[i, j]] = coef;
            matrix[[j, i]] = coef; // Ensure symmetry
        }
    }

    // Make matrix positive definite by ensuring dominant diagonal
    let mut row_sums = Array1::zeros(n_dim);
    for i in 0..n_dim {
        let mut sum = 0.0;
        for j in 0..n_dim {
            if i != j {
                sum += matrix[[i, j]].abs();
            }
        }
        row_sums[i] = sum;
    }

    // Set diagonal elements to ensure positive definiteness
    for i in 0..n_dim {
        matrix[[i, i]] = row_sums[i] + (largest_coef - smallest_coef).abs() + 0.1;
    }

    // Normalize diagonal if requested
    if norm_diag {
        for i in 0..n_dim {
            matrix[[i, i]] = 1.0;
        }

        // Add small positive value to ensure positive definiteness after normalization
        let min_eigenvalue = 0.01;
        for i in 0..n_dim {
            matrix[[i, i]] += min_eigenvalue;
        }
    }

    Ok(matrix)
}

pub fn make_spd_matrix(n_dim: usize, random_state: Option<u64>) -> Result<Array2<f64>> {
    if n_dim == 0 {
        return Err(SklearsError::InvalidInput(
            "n_dim must be positive".to_string(),
        ));
    }

    let mut rng = if let Some(seed) = random_state {
        StdRng::seed_from_u64(seed)
    } else {
        StdRng::from_rng(&mut scirs2_core::random::thread_rng())
    };

    let normal = StandardNormal;

    // Generate a random matrix A
    let mut a = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        for j in 0..n_dim {
            a[[i, j]] = rng.sample::<f64, _>(normal);
        }
    }

    // Compute A^T * A to get a positive semidefinite matrix
    let mut spd = Array2::zeros((n_dim, n_dim));
    for i in 0..n_dim {
        for j in 0..n_dim {
            let mut sum = 0.0;
            for k in 0..n_dim {
                sum += a[[k, i]] * a[[k, j]];
            }
            spd[[i, j]] = sum;
        }
    }

    // Add a small positive value to the diagonal to ensure positive definiteness
    let min_eigenvalue = 0.01;
    for i in 0..n_dim {
        spd[[i, i]] += min_eigenvalue;
    }

    Ok(spd)
}

