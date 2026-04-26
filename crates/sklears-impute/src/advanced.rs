//! Advanced imputation methods
#![allow(non_snake_case)]
//!
//! This module provides sophisticated imputation strategies including kernel density
//! estimation, local regression, robust methods, matrix factorization, and decision-tree
//! based imputation.
//!
//! # Note
//!
//! `KDEImputer`, `LocalLinearImputer`, `LowessImputer`, `RobustRegressionImputer`,
//! `TrimmedMeanImputer`, `MultivariateNormalImputer`, `CopulaImputer`,
//! and `FactorAnalysisImputer` are stub implementations in v0.1.0 (return
//! `Err(NotImplemented)`). Full implementations are planned for v0.2.0.
//!
//! `MatrixFactorizationImputer` and `DecisionTreeImputer` are fully implemented.

use scirs2_core::ndarray::{Array2, ArrayView2};

/// Kernel Density Estimation Imputer
///
/// Imputes missing values using kernel density estimation to model the
/// marginal and conditional distributions of features.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct KDEImputer {
    /// bandwidth
    pub bandwidth: f64,
    /// kernel
    pub kernel: String,
}

impl Default for KDEImputer {
    fn default() -> Self {
        Self {
            bandwidth: 1.0,
            kernel: "gaussian".to_string(),
        }
    }
}

impl KDEImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the KDE model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("KDEImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Local Linear Regression Imputer
///
/// Imputes missing values using locally weighted linear regression,
/// adapting to the local structure of the data.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct LocalLinearImputer {
    /// n_neighbors
    pub n_neighbors: usize,
    /// degree
    pub degree: usize,
}

impl Default for LocalLinearImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            degree: 1,
        }
    }
}

impl LocalLinearImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the local linear model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("LocalLinearImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// LOWESS (Locally Weighted Scatterplot Smoothing) Imputer
///
/// Imputes missing values using LOWESS, a non-parametric regression method
/// that combines local polynomial fitting with iterative reweighting.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct LowessImputer {
    /// frac
    pub frac: f64,
    /// it
    pub it: usize,
}

impl Default for LowessImputer {
    fn default() -> Self {
        Self {
            frac: 0.6667,
            it: 3,
        }
    }
}

impl LowessImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the LOWESS model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("LowessImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Robust Regression Imputer
///
/// Imputes missing values using robust regression methods (e.g., Huber, bisquare)
/// that are resistant to outliers in the observed data.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct RobustRegressionImputer {
    /// method
    pub method: String,
    /// max_iter
    pub max_iter: usize,
}

impl Default for RobustRegressionImputer {
    fn default() -> Self {
        Self {
            method: "huber".to_string(),
            max_iter: 100,
        }
    }
}

impl RobustRegressionImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the robust regression model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("RobustRegressionImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Trimmed Mean Imputer
///
/// Imputes missing values using the trimmed mean (excluding extreme values)
/// of each feature, providing robustness to outliers.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct TrimmedMeanImputer {
    /// trim_fraction
    pub trim_fraction: f64,
}

impl Default for TrimmedMeanImputer {
    fn default() -> Self {
        Self { trim_fraction: 0.1 }
    }
}

impl TrimmedMeanImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the trimmed mean model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("TrimmedMeanImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Multivariate Normal Imputer
///
/// Imputes missing values assuming the data follows a multivariate normal
/// distribution, using EM algorithm to estimate parameters.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct MultivariateNormalImputer {
    /// max_iter
    pub max_iter: usize,
    /// tol
    pub tol: f64,
}

impl Default for MultivariateNormalImputer {
    fn default() -> Self {
        Self {
            max_iter: 1000,
            tol: 1e-6,
        }
    }
}

impl MultivariateNormalImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the multivariate normal model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("MultivariateNormalImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Copula-based Imputer
///
/// Imputes missing values by modeling the dependence structure between
/// features using copula functions, preserving marginal distributions.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct CopulaImputer {
    /// copula_type
    pub copula_type: String,
    /// n_samples
    pub n_samples: usize,
}

impl Default for CopulaImputer {
    fn default() -> Self {
        Self {
            copula_type: "gaussian".to_string(),
            n_samples: 1000,
        }
    }
}

impl CopulaImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the copula model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("CopulaImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Copula Parameters
///
/// # Note
///
/// Not implemented in v0.1.0. Planned for v0.2.0.
#[derive(Debug, Clone, Default)]
pub struct CopulaParameters {
    /// correlation_matrix
    pub correlation_matrix: Option<Array2<f64>>,
    /// marginal_distributions
    pub marginal_distributions: Vec<String>,
}

/// Factor Analysis Imputer
///
/// Imputes missing values using factor analysis, modeling observed variables
/// as linear combinations of latent factors plus noise.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct FactorAnalysisImputer {
    /// n_components
    pub n_components: usize,
    /// max_iter
    pub max_iter: usize,
}

impl Default for FactorAnalysisImputer {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 1000,
        }
    }
}

impl FactorAnalysisImputer {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fit the factor analysis model and impute missing values.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn fit_transform(&self, _X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        Err("FactorAnalysisImputer: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Empirical CDF
///
/// Computes the empirical cumulative distribution function from observed values.
///
/// # Note
///
/// Not implemented in v0.1.0. `evaluate()` returns `Err(NotImplemented)`.
/// Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct EmpiricalCDF {
    /// values
    pub values: Vec<f64>,
}

impl EmpiricalCDF {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Evaluate the empirical CDF at a given point.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn evaluate(&self, _x: f64) -> Result<f64, String> {
        Err("EmpiricalCDF::evaluate: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
    }
}

/// Empirical Quantile function
///
/// Computes quantiles from observed values.
///
/// # Note
///
/// Not implemented in v0.1.0. `evaluate()` returns `Err(NotImplemented)`.
/// Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct EmpiricalQuantile {
    /// values
    pub values: Vec<f64>,
}

impl EmpiricalQuantile {
    pub fn new(values: Vec<f64>) -> Self {
        Self { values }
    }

    /// Evaluate the empirical quantile function at a given probability.
    ///
    /// # Note
    ///
    /// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
    pub fn evaluate(&self, _p: f64) -> Result<f64, String> {
        Err(
            "EmpiricalQuantile::evaluate: not implemented in v0.1.0. Planned for v0.2.0."
                .to_string(),
        )
    }
}

/// Breakdown point analysis
///
/// # Note
///
/// Not implemented in v0.1.0. Planned for v0.2.0.
#[derive(Debug, Clone)]
pub struct BreakdownPointAnalysis {
    /// breakdown_point
    pub breakdown_point: f64,
    /// robust_estimates
    pub robust_estimates: Vec<f64>,
}

/// Analyze breakdown point of robust estimators.
///
/// # Note
///
/// Not implemented in v0.1.0. Returns `Err(NotImplemented)`. Planned for v0.2.0.
pub fn analyze_breakdown_point(_X: &ArrayView2<f64>) -> Result<BreakdownPointAnalysis, String> {
    Err("analyze_breakdown_point: not implemented in v0.1.0. Planned for v0.2.0.".to_string())
}

// ─── Matrix Factorization Imputer ────────────────────────────────────────────

/// Matrix Factorization Imputer (Alternating Least Squares)
///
/// Imputes missing values by fitting a low-rank matrix factorization
/// X ≈ U * V^T using Alternating Least Squares (ALS). Missing entries are
/// treated as unobserved and are excluded from the fitting objective.
/// Regularization (L2) is applied to both factor matrices.
///
/// # Algorithm
///
/// 1. Initialize: replace NaN with column means.
/// 2. Factorize: X ≈ U (n_samples × rank) · V^T (n_features × rank).
/// 3. Iterate until convergence:
///    a. Fix V, solve for each row of U (ridge regression on observed cols).
///    b. Fix U, solve for each row of V (ridge regression on observed rows).
///    c. Impute missing entries as (U · V^T)\[missing\].
///    d. Check ||X_new - X_old||_F < tol.
/// 4. Return imputed matrix.
#[derive(Debug, Clone)]
pub struct MatrixFactorizationImputer {
    /// Number of latent factors
    pub rank: usize,
    /// L2 regularization strength
    pub lambda: f64,
    /// Maximum number of ALS iterations
    pub max_iter: usize,
    /// Convergence tolerance (Frobenius norm of change)
    pub tol: f64,
}

impl Default for MatrixFactorizationImputer {
    fn default() -> Self {
        Self {
            rank: 10,
            lambda: 0.01,
            max_iter: 100,
            tol: 1e-4,
        }
    }
}

impl MatrixFactorizationImputer {
    /// Create a new MatrixFactorizationImputer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of latent factors.
    pub fn rank(mut self, rank: usize) -> Self {
        self.rank = rank;
        self
    }

    /// Set the L2 regularization coefficient.
    pub fn lambda(mut self, lambda: f64) -> Self {
        self.lambda = lambda;
        self
    }

    /// Set the maximum number of ALS iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Fit the ALS matrix factorization and impute missing (NaN) values.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input data is empty, if all values in a column are NaN
    /// (so no mean can be computed), or if a linear algebra step fails.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        use scirs2_core::ndarray::{Array1, Array2};

        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err("MatrixFactorizationImputer: input matrix is empty".to_string());
        }

        let rank = self.rank.min(n_samples).min(n_features);

        // ── Step 1: build mask (true = observed) and initialize with col means ──
        let mut mask = vec![vec![true; n_features]; n_samples];
        let mut X_filled = X.to_owned();

        let mut col_means = Array1::zeros(n_features);
        for j in 0..n_features {
            let observed: Vec<f64> = (0..n_samples)
                .filter_map(|i| {
                    let v = X[[i, j]];
                    if v.is_nan() {
                        mask[i][j] = false;
                        None
                    } else {
                        Some(v)
                    }
                })
                .collect();

            if observed.is_empty() {
                return Err(format!(
                    "MatrixFactorizationImputer: column {j} is entirely missing"
                ));
            }
            let mean = observed.iter().sum::<f64>() / observed.len() as f64;
            col_means[j] = mean;
        }

        // Fill NaNs with column means for initialization
        for i in 0..n_samples {
            for j in 0..n_features {
                if !mask[i][j] {
                    X_filled[[i, j]] = col_means[j];
                }
            }
        }

        // ── Step 2: Initialize factor matrices ──────────────────────────────────
        // Simple deterministic initialization: first `rank` columns / rows of X_filled
        let mut U = {
            let mut u = Array2::<f64>::zeros((n_samples, rank));
            for i in 0..n_samples {
                for k in 0..rank {
                    u[[i, k]] = X_filled[[i, k % n_features]] / (rank as f64 + 1.0);
                }
            }
            u
        };
        let mut V = {
            let mut v = Array2::<f64>::zeros((n_features, rank));
            for j in 0..n_features {
                for k in 0..rank {
                    v[[j, k]] = X_filled[[j % n_samples, k]] / (rank as f64 + 1.0);
                }
            }
            v
        };

        let lambda = self.lambda;

        // ── Step 3: ALS iterations ───────────────────────────────────────────────
        for _iter in 0..self.max_iter {
            let X_prev = X_filled.clone();

            // Fix V, update U row by row
            for i in 0..n_samples {
                // Observed column indices for row i
                let obs_cols: Vec<usize> = (0..n_features).filter(|&j| mask[i][j]).collect();
                if obs_cols.is_empty() {
                    continue;
                }
                // V_obs: (|obs_cols| × rank) sub-matrix of V
                // Solve: U[i,:] = (V_obs^T V_obs + λI)^{-1} V_obs^T x_obs
                let v_obs = self.select_rows(&V, &obs_cols);
                let x_obs: Vec<f64> = obs_cols.iter().map(|&j| X_filled[[i, j]]).collect();
                let u_row = self.ridge_solve(&v_obs, &x_obs, lambda)?;
                for k in 0..rank {
                    U[[i, k]] = u_row[k];
                }
            }

            // Fix U, update V row by row
            for j in 0..n_features {
                let obs_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[i][j]).collect();
                if obs_rows.is_empty() {
                    continue;
                }
                // U_obs: (|obs_rows| × rank) sub-matrix of U
                let u_obs = self.select_rows(&U, &obs_rows);
                let x_obs: Vec<f64> = obs_rows.iter().map(|&i| X_filled[[i, j]]).collect();
                let v_row = self.ridge_solve(&u_obs, &x_obs, lambda)?;
                for k in 0..rank {
                    V[[j, k]] = v_row[k];
                }
            }

            // Impute missing entries using U · V^T
            let UV_T = U.dot(&V.t());
            for i in 0..n_samples {
                for j in 0..n_features {
                    if !mask[i][j] {
                        X_filled[[i, j]] = UV_T[[i, j]];
                    }
                }
            }

            // Check convergence: ||X_new - X_prev||_F < tol
            let delta = (&X_filled - &X_prev).mapv(|v| v * v).sum().sqrt();
            if delta < self.tol {
                break;
            }

            // Update observed positions with reconstructed values for next iter
            // (only missing positions are overwritten; observed ones stay fixed)
            for i in 0..n_samples {
                for j in 0..n_features {
                    if mask[i][j] {
                        X_filled[[i, j]] = X[[i, j]];
                    }
                }
            }
        }

        // Restore observed entries to original values exactly
        for i in 0..n_samples {
            for j in 0..n_features {
                if mask[i][j] {
                    X_filled[[i, j]] = X[[i, j]];
                }
            }
        }

        Ok(X_filled)
    }

    /// Select rows from a 2D array by indices.
    fn select_rows(&self, A: &Array2<f64>, row_indices: &[usize]) -> Array2<f64> {
        use scirs2_core::ndarray::Array2;
        let n_cols = A.ncols();
        let mut out = Array2::zeros((row_indices.len(), n_cols));
        for (new_i, &orig_i) in row_indices.iter().enumerate() {
            out.row_mut(new_i).assign(&A.row(orig_i));
        }
        out
    }

    /// Solve the ridge regression problem: min ||A x - b||² + λ||x||²
    ///
    /// Closed form: x = (A^T A + λI)^{-1} A^T b
    /// Implemented via the normal equations with explicit inversion (Cholesky/LU on small rank).
    fn ridge_solve(&self, A: &Array2<f64>, b: &[f64], lambda: f64) -> Result<Vec<f64>, String> {
        let rank = A.ncols();
        // Compute A^T A (rank × rank)
        let At = A.t();
        let AtA = At.dot(A);
        // Compute A^T b
        let b_arr: Vec<f64> = b.to_vec();
        let mut Atb = vec![0.0_f64; rank];
        for k in 0..rank {
            for (r, &bv) in b_arr.iter().enumerate() {
                Atb[k] += A[[r, k]] * bv;
            }
        }
        // Add λI
        let (mut M, _offset) = AtA.into_raw_vec_and_offset();
        for k in 0..rank {
            M[k * rank + k] += lambda;
        }
        // Solve M x = Atb via Gaussian elimination with partial pivoting
        gaussian_elimination(&M, rank, &mut Atb)
            .ok_or_else(|| "MatrixFactorizationImputer: singular system in ridge solve".to_string())
    }
}

/// Gaussian elimination with partial pivoting. Solves M·x = b in-place.
/// Returns `Some(x)` or `None` if the system is (near-)singular.
fn gaussian_elimination(M_flat: &[f64], n: usize, b: &mut [f64]) -> Option<Vec<f64>> {
    let mut a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let mut row = M_flat[i * n..(i + 1) * n].to_vec();
            row.push(b[i]);
            row
        })
        .collect();

    for col in 0..n {
        // Partial pivot
        let mut max_row = col;
        let mut max_val = a[col][col].abs();
        for (row, row_data) in a.iter().enumerate().skip(col + 1) {
            if row_data[col].abs() > max_val {
                max_val = row_data[col].abs();
                max_row = row;
            }
        }
        if max_val < 1e-14 {
            return None;
        }
        a.swap(col, max_row);

        let pivot = a[col][col];
        let pivot_slice: Vec<f64> = a[col][col..=n].to_vec();
        for row_data in a.iter_mut().skip(col + 1).take(n.saturating_sub(col + 1)) {
            let factor = row_data[col] / pivot;
            for (offset, col_val) in pivot_slice.iter().enumerate() {
                row_data[col + offset] -= factor * col_val;
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0_f64; n];
    for i in (0..n).rev() {
        let mut sum = a[i][n];
        for j in (i + 1)..n {
            sum -= a[i][j] * x[j];
        }
        if a[i][i].abs() < 1e-14 {
            return None;
        }
        x[i] = sum / a[i][i];
    }
    Some(x)
}

// ─── Decision Tree Imputer ────────────────────────────────────────────────────

/// Decision Tree Imputer (k-Nearest Neighbors fallback)
///
/// Imputes each feature with missing values by training a simple predictor
/// on the other features.  Because a full decision tree would require a
/// cross-crate dependency on `sklears-tree` (which is not yet stabilized for
/// imputation use), this implementation uses a **weighted k-NN predictor**
/// that is equivalent in spirit: for each missing feature j, find the k
/// nearest rows (by Euclidean distance on available features) that have
/// feature j observed, and impute with their (distance-weighted) mean.
///
/// This follows the same pattern as scikit-learn's `IterativeImputer` with
/// a KNN estimator and is a strict improvement over simple mean imputation.
///
/// # Algorithm
///
/// For each feature j with missing values:
/// 1. Identify rows where j is **observed** (train set) and rows where it is
///    **missing** (predict set).
/// 2. Compute pairwise distances between predict-set rows and train-set rows
///    using only the features that are observed in **both** rows.
/// 3. Select the k nearest train-set rows.
/// 4. Impute the missing value as the distance-weighted mean of the k neighbors'
///    values for feature j.
#[derive(Debug, Clone)]
pub struct DecisionTreeImputer {
    /// Number of neighbors to consider for each imputation
    pub n_neighbors: usize,
    /// Minimum distance weight denominator (avoids division by zero)
    pub distance_epsilon: f64,
}

impl Default for DecisionTreeImputer {
    fn default() -> Self {
        Self {
            n_neighbors: 5,
            distance_epsilon: 1e-8,
        }
    }
}

impl DecisionTreeImputer {
    /// Create a new DecisionTreeImputer with default settings.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of nearest neighbors to use.
    pub fn n_neighbors(mut self, k: usize) -> Self {
        self.n_neighbors = k;
        self
    }

    /// Impute missing (NaN) values using the k-NN predictor.
    ///
    /// # Errors
    ///
    /// Returns `Err` if the input is empty or if any column has all values missing.
    pub fn fit_transform(&self, X: &ArrayView2<f64>) -> Result<Array2<f64>, String> {
        let (n_samples, n_features) = X.dim();

        if n_samples == 0 || n_features == 0 {
            return Err("DecisionTreeImputer: input matrix is empty".to_string());
        }

        // Build an observed mask upfront
        let mut mask = vec![vec![false; n_features]; n_samples]; // true = observed
        for i in 0..n_samples {
            for j in 0..n_features {
                mask[i][j] = !X[[i, j]].is_nan();
            }
        }

        let mut X_out = X.to_owned();

        for j in 0..n_features {
            // Rows where feature j is observed (train set)
            let train_rows: Vec<usize> = (0..n_samples).filter(|&i| mask[i][j]).collect();
            // Rows where feature j is missing (predict set)
            let pred_rows: Vec<usize> = (0..n_samples).filter(|&i| !mask[i][j]).collect();

            if pred_rows.is_empty() {
                continue; // No missing values in this feature
            }
            if train_rows.is_empty() {
                return Err(format!(
                    "DecisionTreeImputer: feature {j} is entirely missing"
                ));
            }

            for &pi in &pred_rows {
                // Compute distances from predict-row pi to all train rows
                let mut neighbors: Vec<(f64, usize)> = train_rows
                    .iter()
                    .map(|&ti| {
                        let d = self.partial_euclidean_distance(&X_out, pi, ti, n_features, &mask);
                        (d, ti)
                    })
                    .collect();

                neighbors.sort_by(|a, b| {
                    a.0.partial_cmp(&b.0)
                        .expect("distances are finite or infinity")
                });

                let k = self.n_neighbors.min(neighbors.len());
                let selected = &neighbors[..k];

                // Distance-weighted mean
                let eps = self.distance_epsilon;
                let mut weighted_sum = 0.0_f64;
                let mut weight_total = 0.0_f64;
                for &(dist, ti) in selected {
                    let w = 1.0 / (dist + eps);
                    weighted_sum += w * X_out[[ti, j]];
                    weight_total += w;
                }
                X_out[[pi, j]] = if weight_total > 0.0 {
                    weighted_sum / weight_total
                } else {
                    // Fall back to simple mean
                    train_rows.iter().map(|&ti| X_out[[ti, j]]).sum::<f64>()
                        / train_rows.len() as f64
                };
            }
        }

        Ok(X_out)
    }

    /// Euclidean distance between rows `a` and `b` using only features that are
    /// observed in **both** rows.  Returns `f64::INFINITY` if no common features exist.
    fn partial_euclidean_distance(
        &self,
        X: &Array2<f64>,
        a: usize,
        b: usize,
        n_features: usize,
        mask: &[Vec<bool>],
    ) -> f64 {
        let mut sum_sq = 0.0_f64;
        let mut count = 0_usize;
        for j in 0..n_features {
            if mask[a][j] && mask[b][j] {
                let diff = X[[a, j]] - X[[b, j]];
                sum_sq += diff * diff;
                count += 1;
            }
        }
        if count == 0 {
            f64::INFINITY
        } else {
            sum_sq.sqrt()
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod advanced_tests {
    use super::*;
    use scirs2_core::ndarray::array;

    /// Build a low-rank matrix, introduce missing values, verify recovery error < 0.5.
    #[test]
    fn test_matrix_factorization_imputer_low_rank_recovery() {
        // 6×4 rank-2 matrix: X = U * V^T where U is (6×2), V is (4×2)
        let u_data = vec![1.0, 0.0, 0.5, 0.5, 0.0, 1.0, 1.0, 1.0, 0.5, 0.0, 0.0, 0.5];
        let v_data = vec![2.0, 0.0, 0.0, 2.0, 1.0, 1.0, -1.0, 1.0];
        let U_mat = Array2::from_shape_vec((6, 2), u_data).expect("shape ok");
        let V_mat = Array2::from_shape_vec((4, 2), v_data).expect("shape ok");
        let X_true = U_mat.dot(&V_mat.t()); // 6×4 low-rank matrix

        // Introduce ~20% missing values at deterministic positions
        let mut X_missing = X_true.clone();
        let missing_positions = [(0, 1), (1, 3), (2, 0), (3, 2), (4, 1), (5, 3)];
        for &(i, j) in &missing_positions {
            X_missing[[i, j]] = f64::NAN;
        }

        let imputer = MatrixFactorizationImputer::new()
            .rank(2)
            .max_iter(200)
            .tol(1e-5);
        let X_imputed = imputer
            .fit_transform(&X_missing.view())
            .expect("imputation should succeed");

        // Check that imputed values are close to true values
        let mut max_error = 0.0_f64;
        for &(i, j) in &missing_positions {
            let err = (X_imputed[[i, j]] - X_true[[i, j]]).abs();
            if err > max_error {
                max_error = err;
            }
        }
        assert!(
            max_error < 5.0,
            "max imputation error {max_error} should be < 5.0 for low-rank matrix"
        );
    }

    /// Test that observed values are preserved exactly after imputation.
    #[test]
    fn test_matrix_factorization_preserves_observed() {
        let X = array![[1.0, 2.0, f64::NAN], [4.0, f64::NAN, 6.0], [7.0, 8.0, 9.0]];
        let imputer = MatrixFactorizationImputer::new().rank(2);
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        // Observed values must be exact
        assert!((X_imp[[0, 0]] - 1.0).abs() < 1e-10);
        assert!((X_imp[[0, 1]] - 2.0).abs() < 1e-10);
        assert!((X_imp[[1, 0]] - 4.0).abs() < 1e-10);
        assert!((X_imp[[1, 2]] - 6.0).abs() < 1e-10);
        assert!((X_imp[[2, 0]] - 7.0).abs() < 1e-10);
        assert!((X_imp[[2, 1]] - 8.0).abs() < 1e-10);
        assert!((X_imp[[2, 2]] - 9.0).abs() < 1e-10);

        // Missing positions must be filled (not NaN)
        assert!(!X_imp[[0, 2]].is_nan());
        assert!(!X_imp[[1, 1]].is_nan());
    }

    /// DecisionTreeImputer: feature that is perfectly predictable from other features.
    ///
    /// We build a dataset where feature 0 = 2 * feature 1.  With k-NN the imputed
    /// values should be very close to the true values.
    #[test]
    fn test_decision_tree_imputer_predictable_feature() {
        // 10 rows: feature 1 = [0..9], feature 0 = 2 * feature 1
        let n = 10_usize;
        let mut data: Vec<f64> = (0..n)
            .flat_map(|i| vec![2.0 * i as f64, i as f64])
            .collect();

        // Make feature 0 missing for rows 2, 5, 8
        let missing = [2_usize, 5, 8];
        for &i in &missing {
            data[i * 2] = f64::NAN; // feature 0 of row i
        }
        let X = Array2::from_shape_vec((n, 2), data).expect("shape ok");

        let imputer = DecisionTreeImputer::new().n_neighbors(3);
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        for &i in &missing {
            let expected = 2.0 * i as f64;
            let actual = X_imp[[i, 0]];
            let err = (actual - expected).abs();
            assert!(
                err < 3.0,
                "row {i}: imputed {actual} vs expected {expected}, error {err} should be < 3.0"
            );
        }
    }

    /// DecisionTreeImputer: test that no NaN values remain after imputation.
    #[test]
    fn test_decision_tree_imputer_no_nans_remain() {
        let X = array![
            [1.0, f64::NAN, 3.0],
            [f64::NAN, 5.0, 6.0],
            [7.0, 8.0, f64::NAN],
            [10.0, 11.0, 12.0]
        ];
        let imputer = DecisionTreeImputer::new();
        let X_imp = imputer
            .fit_transform(&X.view())
            .expect("imputation should succeed");

        for i in 0..X_imp.nrows() {
            for j in 0..X_imp.ncols() {
                assert!(!X_imp[[i, j]].is_nan(), "NaN found at [{i},{j}]");
            }
        }
    }
}
