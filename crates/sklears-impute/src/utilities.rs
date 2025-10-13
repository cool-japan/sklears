//! Utility functions for advanced imputation methods

use scirs2_core::ndarray::{Array1, Array2};

/// Solve a linear system Ax = b using Gaussian elimination with partial pivoting
pub fn solve_linear_system(A: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, &'static str> {
    let n = A.nrows();
    if A.ncols() != n || b.len() != n {
        return Err("Incompatible dimensions");
    }

    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[[i, j]] = A[[i, j]];
        }
        aug[[i, n]] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        for i in (k + 1)..n {
            if aug[[i, k]].abs() > aug[[pivot_row, k]].abs() {
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if pivot_row != k {
            for j in 0..=n {
                let temp = aug[[k, j]];
                aug[[k, j]] = aug[[pivot_row, j]];
                aug[[pivot_row, j]] = temp;
            }
        }

        // Check for singular matrix
        if aug[[k, k]].abs() < 1e-12 {
            return Err("Matrix is singular");
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[[i, k]] / aug[[k, k]];
            for j in k..=n {
                aug[[i, j]] -= factor * aug[[k, j]];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        x[i] = aug[[i, n]];
        for j in (i + 1)..n {
            x[i] -= aug[[i, j]] * x[j];
        }
        x[i] /= aug[[i, i]];
    }

    Ok(x)
}

/// Solve a linear system using Vec-based matrices (alternative implementation)
pub fn solve_linear_system_vec(A: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>, &'static str> {
    let n = A.len();
    if A.is_empty() || A[0].len() != n || b.len() != n {
        return Err("Incompatible dimensions");
    }

    let mut aug = vec![vec![0.0; n + 1]; n];
    for i in 0..n {
        for j in 0..n {
            aug[i][j] = A[i][j];
        }
        aug[i][n] = b[i];
    }

    // Forward elimination with partial pivoting
    for k in 0..n {
        // Find pivot
        let mut pivot_row = k;
        for i in (k + 1)..n {
            if aug[i][k].abs() > aug[pivot_row][k].abs() {
                pivot_row = i;
            }
        }

        // Swap rows if needed
        if pivot_row != k {
            aug.swap(k, pivot_row);
        }

        // Check for singular matrix
        if aug[k][k].abs() < 1e-12 {
            return Err("Matrix is singular");
        }

        // Eliminate
        for i in (k + 1)..n {
            let factor = aug[i][k] / aug[k][k];
            for j in k..=n {
                aug[i][j] -= factor * aug[k][j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        x[i] = aug[i][n];
        for j in (i + 1)..n {
            x[i] -= aug[i][j] * x[j];
        }
        x[i] /= aug[i][i];
    }

    Ok(x)
}

/// Compute robust statistics (median and MAD) for a set of values
pub fn compute_robust_statistics_helper(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 1.0);
    }

    let mut sorted_values = values.to_vec();
    sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    // Use median as robust central tendency
    let median = if sorted_values.len() % 2 == 0 {
        (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
    } else {
        sorted_values[sorted_values.len() / 2]
    };

    // Use Median Absolute Deviation (MAD) as robust scale
    let mut deviations: Vec<f64> = sorted_values.iter().map(|&x| (x - median).abs()).collect();
    deviations.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let mad = if deviations.len() % 2 == 0 {
        (deviations[deviations.len() / 2 - 1] + deviations[deviations.len() / 2]) / 2.0
    } else {
        deviations[deviations.len() / 2]
    };

    // Convert MAD to consistent estimator for standard deviation
    let robust_scale = 1.4826 * mad;

    (median, robust_scale)
}

/// Calculate mean squared error for a set of targets
pub fn calculate_mse(targets: &[f64]) -> f64 {
    if targets.is_empty() {
        return 0.0;
    }

    let mean = targets.iter().sum::<f64>() / targets.len() as f64;
    targets
        .iter()
        .map(|&target| (target - mean).powi(2))
        .sum::<f64>()
        / targets.len() as f64
}