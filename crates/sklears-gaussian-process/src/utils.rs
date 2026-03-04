//! Utility functions for Gaussian Process computations

// SciRS2 Policy - Use scirs2-autograd for ndarray types and operations
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use sklears_core::error::{Result as SklResult, SklearsError};
use std::f64::consts::PI;
use std::hash::Hash;

/// Robust Cholesky decomposition with automatic jitter addition
pub fn robust_cholesky(K: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = K.nrows();
    let mut K_jittered = K.clone();
    let mut jitter = 1e-12;
    let max_jitter = 1e-3;

    while jitter <= max_jitter {
        // Try Cholesky decomposition
        match cholesky_decomposition(&K_jittered) {
            Ok(L) => return Ok(L),
            Err(_) => {
                // Add jitter to diagonal
                for i in 0..n {
                    K_jittered[[i, i]] = K[[i, i]] + jitter;
                }
                jitter *= 10.0;
            }
        }
    }

    Err(SklearsError::NumericalError(
        "Cholesky decomposition failed even with maximum jitter".to_string(),
    ))
}

/// Standard Cholesky decomposition
pub fn cholesky_decomposition(A: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = A.nrows();
    if n != A.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for Cholesky decomposition".to_string(),
        ));
    }

    let mut L = Array2::<f64>::zeros((n, n));

    for i in 0..n {
        for j in 0..=i {
            if i == j {
                // Diagonal elements
                let sum: f64 = (0..j).map(|k| L[[i, k]].powi(2)).sum();
                let val = A[[i, i]] - sum;
                if val <= 0.0 {
                    return Err(SklearsError::NumericalError(format!(
                        "Matrix is not positive definite at diagonal element {}",
                        i
                    )));
                }
                L[[i, j]] = val.sqrt();
            } else {
                // Off-diagonal elements
                let sum: f64 = (0..j).map(|k| L[[i, k]] * L[[j, k]]).sum();
                if L[[j, j]] == 0.0 {
                    return Err(SklearsError::NumericalError(
                        "Division by zero in Cholesky decomposition".to_string(),
                    ));
                }
                L[[i, j]] = (A[[i, j]] - sum) / L[[j, j]];
            }
        }
    }

    Ok(L)
}

/// Solve triangular system L * x = b using forward/backward substitution
pub fn triangular_solve(L: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
    let n = L.nrows();
    if n != b.len() || n != L.ncols() {
        return Err(SklearsError::InvalidInput(
            "Dimension mismatch in triangular solve".to_string(),
        ));
    }

    // Forward substitution: L * y = b
    let mut y = Array1::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = 0.0;
        for j in 0..i {
            sum += L[[i, j]] * y[j];
        }

        if L[[i, i]].abs() < 1e-14 {
            return Err(SklearsError::NumericalError(format!(
                "Near-singular matrix in triangular solve at row {}",
                i
            )));
        }

        y[i] = (b[i] - sum) / L[[i, i]];
    }

    // Backward substitution: L^T * x = y
    let mut x = Array1::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = 0.0;
        for j in (i + 1)..n {
            sum += L[[j, i]] * x[j]; // L^T[i, j] = L[j, i]
        }

        if L[[i, i]].abs() < 1e-14 {
            return Err(SklearsError::NumericalError(format!(
                "Near-singular matrix in backward substitution at row {}",
                i
            )));
        }

        x[i] = (y[i] - sum) / L[[i, i]];
    }

    Ok(x)
}

/// Solve lower-triangular system L * X = B (forward substitution only, no back-solve).
/// This is the batch version needed for predictive variance: solve once for all columns.
pub fn forward_solve_matrix(L: &Array2<f64>, B: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = L.nrows();
    let m = B.ncols();

    if n != B.nrows() || n != L.ncols() {
        return Err(SklearsError::InvalidInput(
            "Dimension mismatch in forward_solve_matrix".to_string(),
        ));
    }

    let mut X = Array2::<f64>::zeros((n, m));

    // Forward substitution for all columns at once (better cache locality than column-by-column)
    for i in 0..n {
        if L[[i, i]].abs() < 1e-14 {
            return Err(SklearsError::NumericalError(format!(
                "Near-singular matrix in forward_solve_matrix at row {}", i
            )));
        }
        let inv_diag = 1.0 / L[[i, i]];
        for j in 0..m {
            let mut sum = B[[i, j]];
            for k in 0..i {
                sum -= L[[i, k]] * X[[k, j]];
            }
            X[[i, j]] = sum * inv_diag;
        }
    }

    Ok(X)
}

/// Compute log marginal likelihood for Gaussian process
pub fn log_marginal_likelihood(L: &Array2<f64>, alpha: &Array1<f64>, y: &Array1<f64>) -> f64 {
    let n = y.len() as f64;

    // Log determinant of K = 2 * sum(log(diag(L)))
    let log_det_K = 2.0 * L.diag().mapv(|x| x.ln()).sum();

    // Quadratic term: y^T * K^{-1} * y = alpha^T * y
    let quadratic_term = alpha.dot(y);

    // Log marginal likelihood
    -0.5 * quadratic_term - 0.5 * log_det_K - 0.5 * n * (2.0 * PI).ln()
}

/// Estimate condition number of a matrix
pub fn estimate_condition_number(K: &Array2<f64>) -> f64 {
    // Simple estimate using diagonal elements
    let diag = K.diag();
    let max_diag = diag.fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_diag = diag.fold(f64::INFINITY, |a, &b| a.min(b));

    if min_diag <= 0.0 {
        f64::INFINITY
    } else {
        max_diag / min_diag
    }
}

/// Add jitter to matrix diagonal for numerical stability
pub fn add_jitter(K: &mut Array2<f64>, jitter: f64) {
    for i in 0..K.nrows() {
        K[[i, i]] += jitter;
    }
}

/// Initialize inducing points using k-means clustering
pub fn kmeans_inducing_points(
    X: &ArrayView2<f64>,
    n_inducing: usize,
    random_state: Option<u64>,
) -> SklResult<Array2<f64>> {
    let n_samples = X.nrows();
    let n_features = X.ncols();

    if n_inducing >= n_samples {
        return Ok(X.to_owned());
    }

    // Simple k-means implementation
    let rng = match random_state {
        Some(seed) => {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::Hasher;
            let mut hasher = DefaultHasher::new();
            seed.hash(&mut hasher);
            hasher.finish()
        }
        None => 42, // Default seed
    };

    // Initialize centroids randomly
    let mut centroids = Array2::<f64>::zeros((n_inducing, n_features));
    for i in 0..n_inducing {
        let idx = (rng as usize + i * 1337) % n_samples;
        centroids.row_mut(i).assign(&X.row(idx));
    }

    // Simple k-means iterations
    for _iter in 0..10 {
        // Assign points to closest centroids
        let mut assignments = vec![0; n_samples];
        for i in 0..n_samples {
            let mut min_dist = f64::INFINITY;
            for j in 0..n_inducing {
                let dist = (&X.row(i) - &centroids.row(j)).mapv(|x| x.powi(2)).sum();
                if dist < min_dist {
                    min_dist = dist;
                    assignments[i] = j;
                }
            }
        }

        // Update centroids
        for j in 0..n_inducing {
            let mut count = 0;
            let mut sum = Array1::<f64>::zeros(n_features);
            for i in 0..n_samples {
                if assignments[i] == j {
                    sum = sum + X.row(i);
                    count += 1;
                }
            }
            if count > 0 {
                centroids.row_mut(j).assign(&(sum / count as f64));
            }
        }
    }

    Ok(centroids)
}

/// Initialize inducing points uniformly from the data
pub fn uniform_inducing_points(
    X: &ArrayView2<f64>,
    n_inducing: usize,
    random_state: Option<u64>,
) -> SklResult<Array2<f64>> {
    let n_samples = X.nrows();

    if n_inducing >= n_samples {
        return Ok(X.to_owned());
    }

    let mut rng = random_state.unwrap_or(42);

    let mut inducing_points = Array2::<f64>::zeros((n_inducing, X.ncols()));
    let mut used_indices = std::collections::HashSet::new();

    for i in 0..n_inducing {
        let mut idx;
        loop {
            idx = (rng as usize + i * 1337 + i * i * 7919) % n_samples;
            if !used_indices.contains(&idx) {
                used_indices.insert(idx);
                break;
            }
            rng = rng.wrapping_mul(1103515245).wrapping_add(12345); // Simple LCG
        }
        inducing_points.row_mut(i).assign(&X.row(idx));
    }

    Ok(inducing_points)
}

/// Initialize inducing points randomly
pub fn random_inducing_points(
    X: &ArrayView2<f64>,
    n_inducing: usize,
    random_state: Option<u64>,
) -> SklResult<Array2<f64>> {
    uniform_inducing_points(X, n_inducing, random_state)
}

/// Solve triangular system L * X = B where B is a matrix
pub fn triangular_solve_matrix(L: &Array2<f64>, B: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = L.nrows();
    let m = B.ncols();

    if n != B.nrows() || n != L.ncols() {
        return Err(SklearsError::InvalidInput(
            "Dimension mismatch in triangular solve matrix".to_string(),
        ));
    }

    let mut X = Array2::<f64>::zeros((n, m));

    // Solve each column separately
    for j in 0..m {
        let b_col = B.column(j).to_owned();
        let x_col = triangular_solve(L, &b_col)?;
        X.column_mut(j).assign(&x_col);
    }

    Ok(X)
}

/// Compute matrix inverse using Cholesky decomposition
#[allow(non_snake_case)]
pub fn matrix_inverse(A: &Array2<f64>) -> SklResult<Array2<f64>> {
    let n = A.nrows();
    if n != A.ncols() {
        return Err(SklearsError::InvalidInput(
            "Matrix must be square for inversion".to_string(),
        ));
    }

    // Cholesky decomposition
    let L = robust_cholesky(A)?;

    // Solve L * X = I
    let I = Array2::<f64>::eye(n);
    let Y = triangular_solve_matrix(&L, &I)?;

    // Solve L^T * A_inv = Y
    let L_T = L.t().to_owned();
    let A_inv = triangular_solve_matrix(&L_T, &Y)?;

    Ok(A_inv.t().to_owned())
}

/// Alias for cholesky_decomposition for compatibility
pub fn cholesky(A: &Array2<f64>) -> SklResult<Array2<f64>> {
    cholesky_decomposition(A)
}

/// Compute the log determinant of a matrix
#[allow(non_snake_case)]
pub fn log_determinant(A: &Array2<f64>) -> SklResult<f64> {
    let L = robust_cholesky(A)?;
    let log_det = 2.0 * L.diag().mapv(|x| x.ln()).sum();
    Ok(log_det)
}

/// Compute KL divergence between two multivariate normal distributions
pub fn kl_divergence_gaussian(
    mu1: &Array1<f64>,
    sigma1: &Array2<f64>,
    mu2: &Array1<f64>,
    sigma2: &Array2<f64>,
) -> SklResult<f64> {
    let k = mu1.len() as f64;

    // Inverse of sigma2
    let sigma2_inv = matrix_inverse(sigma2)?;

    // Trace term: tr(sigma2_inv @ sigma1)
    let trace_term = (&sigma2_inv * sigma1).sum();

    // Quadratic term: (mu2 - mu1)^T @ sigma2_inv @ (mu2 - mu1)
    let mu_diff = mu2 - mu1;
    let quad_term = mu_diff.dot(&sigma2_inv.dot(&mu_diff));

    // Log determinant terms
    let log_det_sigma2 = log_determinant(sigma2)?;
    let log_det_sigma1 = log_determinant(sigma1)?;

    let kl_div = 0.5 * (trace_term + quad_term - k + log_det_sigma2 - log_det_sigma1);

    Ok(kl_div)
}

/// Ensure a matrix is positive definite by adding regularization if needed
pub fn ensure_positive_definite(A: &Array2<f64>) -> SklResult<Array2<f64>> {
    let mut A_reg = A.clone();
    let n = A.nrows();

    // Check if matrix is already positive definite
    match cholesky_decomposition(&A_reg) {
        Ok(_) => return Ok(A_reg),
        Err(_) => {
            // Add regularization to diagonal
            let min_eigenvalue = estimate_min_eigenvalue(&A_reg);
            let regularization = if min_eigenvalue <= 0.0 {
                1e-6 - min_eigenvalue
            } else {
                1e-6
            };

            for i in 0..n {
                A_reg[[i, i]] += regularization;
            }
        }
    }

    Ok(A_reg)
}

/// Estimate the minimum eigenvalue using power iteration (simplified)
fn estimate_min_eigenvalue(A: &Array2<f64>) -> f64 {
    let n = A.nrows();
    if n == 0 {
        return 0.0;
    }

    // Simple approximation: minimum diagonal element
    A.diag().iter().cloned().fold(f64::INFINITY, f64::min)
}
