//! Canonical Correlation Analysis (CCA)
//!
//! This module provides Canonical Correlation Analysis, a method for finding linear
//! relationships between two multidimensional variables. CCA finds basis vectors for
//! two sets of variables such that the correlations between the projections of the
//! variables onto these basis vectors are mutually maximized.

use scirs2_core::ndarray::{s, Array1, Array2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Fit, Transform},
};

/// Canonical Correlation Analysis
///
/// CCA finds linear combinations of variables from two datasets that have maximum correlation.
/// It's useful for finding relationships between two sets of variables and for dimensionality
/// reduction in multi-view learning scenarios.
///
/// # Mathematical Background
///
/// Given two datasets X (n×p) and Y (n×q), CCA finds canonical variates:
/// - U = X * A (canonical variates for X)
/// - V = Y * B (canonical variates for Y)
///
/// Such that the correlations between corresponding columns of U and V are maximized.
///
/// The solution involves solving the generalized eigenvalue problem:
/// - (Sxx^(-1) * Sxy * Syy^(-1) * Syx) * A = λ * A
/// - (Syy^(-1) * Syx * Sxx^(-1) * Sxy) * B = λ * B
///
/// Where Sxx, Syy are covariance matrices and Sxy, Syx are cross-covariance matrices.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CanonicalCorrelationAnalysis {
    /// Number of canonical components to compute
    pub n_components: usize,
    /// Whether to copy the input data
    pub copy: bool,
    /// Regularization parameter for numerical stability
    pub regularization: f64,
    /// Maximum number of iterations for iterative solvers
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
}

/// Fitted CCA model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FittedCCA {
    /// Canonical coefficients for X
    pub x_weights: Array2<f64>,
    /// Canonical coefficients for Y  
    pub y_weights: Array2<f64>,
    /// Canonical correlations (eigenvalues)
    pub canonical_correlations: Array1<f64>,
    /// Mean of X training data
    pub x_mean: Array1<f64>,
    /// Mean of Y training data
    pub y_mean: Array1<f64>,
    /// Number of features in X
    pub n_features_x: usize,
    /// Number of features in Y
    pub n_features_y: usize,
    /// Number of canonical components
    pub n_components: usize,
}

impl Default for CanonicalCorrelationAnalysis {
    fn default() -> Self {
        Self::new(2)
    }
}

impl CanonicalCorrelationAnalysis {
    /// Create a new CCA instance
    ///
    /// # Parameters
    /// - `n_components`: Number of canonical components to compute
    ///
    /// # Examples
    /// ```
    /// use sklears_decomposition::CanonicalCorrelationAnalysis;
    ///
    /// let cca = CanonicalCorrelationAnalysis::new(3);
    /// assert_eq!(cca.n_components, 3);
    /// ```
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            copy: true,
            regularization: 1e-6,
            max_iter: 500,
            tol: 1e-6,
        }
    }

    /// Set whether to copy input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Center the data by subtracting the mean
    fn center_data(&self, data: &Array2<f64>) -> Result<(Array2<f64>, Array1<f64>)> {
        let mean = data.mean_axis(Axis(0)).ok_or_else(|| {
            SklearsError::NumericalError("Failed to compute mean along axis".to_string())
        })?;
        let centered = data - &mean.clone().insert_axis(Axis(0));
        Ok((centered, mean))
    }

    /// Compute covariance matrix
    fn compute_covariance(&self, x: &Array2<f64>, y: Option<&Array2<f64>>) -> Result<Array2<f64>> {
        let n_samples = x.nrows() as f64;

        match y {
            Some(y) => {
                // Cross-covariance
                if x.nrows() != y.nrows() {
                    return Err(SklearsError::InvalidInput(
                        "X and Y must have the same number of samples".to_string(),
                    ));
                }
                let cov = x.t().dot(y) / (n_samples - 1.0);
                Ok(cov)
            }
            None => {
                // Auto-covariance
                let cov = x.t().dot(x) / (n_samples - 1.0);
                Ok(cov)
            }
        }
    }

    /// Add regularization to diagonal of matrix for numerical stability
    fn regularize_matrix(&self, mut matrix: Array2<f64>) -> Array2<f64> {
        let n = matrix.nrows().min(matrix.ncols());
        for i in 0..n {
            matrix[[i, i]] += self.regularization;
        }
        matrix
    }

    /// Solve the CCA generalized eigenvalue problem using Cholesky + SVD.
    ///
    /// Algorithm:
    /// 1. Cholesky factorize regularized covariance matrices: Sxx = Lx·Lxᵀ, Syy = Ly·Lyᵀ
    /// 2. Form the cross-covariance in the whitened space: A = Lx⁻¹ · Sxy · Ly⁻ᵀ
    /// 3. SVD: A = U · Σ · Vᵀ  — canonical correlations are the singular values
    /// 4. Back-transform weights: x_weights = Lx⁻ᵀ · U, y_weights = Ly⁻ᵀ · V
    fn solve_cca_eigenvalue_problem(
        &self,
        sxx: &Array2<f64>,
        syy: &Array2<f64>,
        sxy: &Array2<f64>,
    ) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>)> {
        // Regularize covariance matrices for numerical stability
        let sxx_reg = self.regularize_matrix(sxx.clone());
        let syy_reg = self.regularize_matrix(syy.clone());

        // Cholesky factorization: Sxx = Lx · Lxᵀ, Syy = Ly · Lyᵀ
        let lx = sxx_reg.cholesky().map_err(|_| {
            SklearsError::NumericalError(
                "Cholesky factorization of Sxx failed — matrix may not be positive definite"
                    .to_string(),
            )
        })?;
        let ly = syy_reg.cholesky().map_err(|_| {
            SklearsError::NumericalError(
                "Cholesky factorization of Syy failed — matrix may not be positive definite"
                    .to_string(),
            )
        })?;

        // Compute Lx⁻¹ and Ly⁻¹ (lower-triangular inverses)
        let lx_inv = lx.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to invert Cholesky factor Lx".to_string())
        })?;
        let ly_inv = ly.inv().map_err(|_| {
            SklearsError::NumericalError("Failed to invert Cholesky factor Ly".to_string())
        })?;

        // A = Lx⁻¹ · Sxy · Ly⁻ᵀ
        // (the singular values of A are the canonical correlations)
        let a = lx_inv.dot(sxy).dot(&ly_inv.t());

        // Thin SVD: A = U · Σ · Vᵀ
        // compute_uv = true → returns (U, s, Vt)
        let (u, sigma, vt) = a.svd(true).map_err(|_| {
            SklearsError::NumericalError(
                "SVD of whitened cross-covariance matrix failed".to_string(),
            )
        })?;

        // Number of components to return (bounded by rank of A)
        let n_comp = self.n_components.min(sigma.len());

        // Canonical correlations = σ_i clamped to [0, 1] (in descending order from SVD)
        let correlations: Vec<f64> = sigma
            .iter()
            .take(n_comp)
            .map(|&sv| sv.clamp(0.0, 1.0))
            .collect();

        // Back-transform: x_weights = Lx⁻ᵀ · U,  y_weights = Ly⁻ᵀ · V
        // Lx⁻ᵀ = (Lx⁻¹)ᵀ
        let lx_inv_t = lx_inv.t().to_owned();
        let ly_inv_t = ly_inv.t().to_owned();

        // V = Vtᵀ
        let v = vt.t().to_owned();

        // Select top n_comp columns of U and V
        let u_top = u.slice(s![.., ..n_comp]).to_owned();
        let v_top = v.slice(s![.., ..n_comp]).to_owned();

        let x_weights = lx_inv_t.dot(&u_top);
        let y_weights = ly_inv_t.dot(&v_top);

        Ok((x_weights, y_weights, Array1::from_vec(correlations)))
    }
}

impl Fit<(Array2<f64>, Array2<f64>), ()> for CanonicalCorrelationAnalysis {
    type Fitted = FittedCCA;

    fn fit(self, data: &(Array2<f64>, Array2<f64>), _target: &()) -> Result<Self::Fitted> {
        let (x, y) = data;

        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for CCA".to_string(),
            ));
        }

        let n_components = self.n_components.min(x.ncols().min(y.ncols()));

        // Center the data
        let (x_centered, x_mean) = self.center_data(x)?;
        let (y_centered, y_mean) = self.center_data(y)?;

        // Compute covariance matrices
        let sxx = self.compute_covariance(&x_centered, None)?;
        let syy = self.compute_covariance(&y_centered, None)?;
        let sxy = self.compute_covariance(&x_centered, Some(&y_centered))?;

        // Solve the CCA eigenvalue problem
        let (x_weights, y_weights, correlations) =
            self.solve_cca_eigenvalue_problem(&sxx, &syy, &sxy)?;

        Ok(FittedCCA {
            x_weights,
            y_weights,
            canonical_correlations: correlations,
            x_mean,
            y_mean,
            n_features_x: x.ncols(),
            n_features_y: y.ncols(),
            n_components,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedCCA {
    fn transform(&self, data: &Array2<f64>) -> Result<Array2<f64>> {
        if data.ncols() != self.n_features_x {
            let expected = self.n_features_x;
            let got = data.ncols();
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} features for X, got {got}"
            )));
        }

        // Center the data
        let centered = data - &self.x_mean.clone().insert_axis(Axis(0));

        // Transform using X weights
        let transformed = centered.dot(&self.x_weights);
        Ok(transformed)
    }
}

impl FittedCCA {
    /// Transform Y data using the fitted CCA model
    pub fn transform_y(&self, y: &Array2<f64>) -> Result<Array2<f64>> {
        if y.ncols() != self.n_features_y {
            let expected = self.n_features_y;
            let got = y.ncols();
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} features for Y, got {got}"
            )));
        }

        // Center the data
        let centered = y - &self.y_mean.clone().insert_axis(Axis(0));

        // Transform using Y weights
        let transformed = centered.dot(&self.y_weights);
        Ok(transformed)
    }

    /// Get the canonical correlations
    pub fn correlations(&self) -> &Array1<f64> {
        &self.canonical_correlations
    }

    /// Get the X canonical weights
    pub fn x_weights(&self) -> &Array2<f64> {
        &self.x_weights
    }

    /// Get the Y canonical weights
    pub fn y_weights(&self) -> &Array2<f64> {
        &self.y_weights
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cca_creation() {
        let cca = CanonicalCorrelationAnalysis::new(2);
        assert_eq!(cca.n_components, 2);
        assert!(cca.copy);
        assert_eq!(cca.regularization, 1e-6);
    }

    #[test]
    fn test_cca_builder_pattern() {
        let cca = CanonicalCorrelationAnalysis::new(3)
            .copy(false)
            .regularization(1e-4)
            .max_iter(1000)
            .tolerance(1e-8);

        assert_eq!(cca.n_components, 3);
        assert!(!cca.copy);
        assert_eq!(cca.regularization, 1e-4);
        assert_eq!(cca.max_iter, 1000);
        assert_eq!(cca.tol, 1e-8);
    }

    #[test]
    fn test_cca_fit_transform() {
        // Use full-rank data: 6 samples, 3 features for X, 2 for Y.
        // Rows chosen so the 3×3 covariance matrix Sxx has full rank.
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [0.0, 0.0, 3.0],
            [1.0, 1.0, 1.0],
            [2.0, -1.0, 0.5],
            [-1.0, 1.5, -0.5],
        ];

        let y = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.5, 0.5],
            [1.5, -0.5],
            [-0.5, 1.0],
        ];

        let cca = CanonicalCorrelationAnalysis::new(2);
        let fitted = cca
            .fit(&(x.clone(), y.clone()), &())
            .expect("model fitting should succeed");

        assert_eq!(fitted.n_features_x, 3);
        assert_eq!(fitted.n_features_y, 2);
        assert_eq!(fitted.n_components, 2);
        assert_eq!(fitted.x_weights.dim(), (3, 2));
        assert_eq!(fitted.y_weights.dim(), (2, 2));
        assert_eq!(fitted.canonical_correlations.len(), 2);

        // Test transformation
        let x_transformed = fitted.transform(&x).expect("transformation should succeed");
        let y_transformed = fitted
            .transform_y(&y)
            .expect("Y transformation should succeed");

        assert_eq!(x_transformed.dim(), (6, 2));
        assert_eq!(y_transformed.dim(), (6, 2));
    }

    #[test]
    fn test_cca_mismatched_samples() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0], [3.0]]; // Different number of samples

        let cca = CanonicalCorrelationAnalysis::new(1);
        let result = cca.fit(&(x, y), &());
        assert!(result.is_err());
    }

    #[test]
    fn test_cca_insufficient_samples() {
        let x = array![[1.0, 2.0]];
        let y = array![[1.0]];

        let cca = CanonicalCorrelationAnalysis::new(1);
        let result = cca.fit(&(x, y), &());
        assert!(result.is_err());
    }

    #[test]
    fn test_cca_feature_mismatch() {
        // Use full-rank data: 5 samples gives rank-3 Sxx and rank-2 Syy
        let x = array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ];
        let y = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5], [1.5, -0.5],];

        let cca = CanonicalCorrelationAnalysis::new(1);
        let fitted = cca.fit(&(x, y), &()).expect("model fitting should succeed");

        // Test with wrong number of features
        let x_wrong = array![[1.0, 2.0]]; // Should have 3 features
        let result = fitted.transform(&x_wrong);
        assert!(result.is_err());

        let y_wrong = array![[1.0]]; // Should have 2 features
        let result = fitted.transform_y(&y_wrong);
        assert!(result.is_err());
    }

    #[test]
    fn test_cca_correlations() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        // Y is perfectly correlated with X
        let y = x.clone();

        let cca = CanonicalCorrelationAnalysis::new(2);
        let fitted = cca.fit(&(x, y), &()).expect("model fitting should succeed");

        // Correlations should be close to 1 for perfectly correlated data
        let correlations = fitted.correlations();
        for &corr in correlations.iter() {
            assert!(corr >= 0.9, "Correlation should be high: {}", corr);
        }
    }

    #[test]
    fn test_cca_components_access() {
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0],];

        let y = array![[2.0], [4.0], [6.0],];

        let cca = CanonicalCorrelationAnalysis::new(1);
        let fitted = cca.fit(&(x, y), &()).expect("model fitting should succeed");

        let x_weights = fitted.x_weights();
        let y_weights = fitted.y_weights();
        let correlations = fitted.correlations();

        assert_eq!(x_weights.dim(), (2, 1));
        assert_eq!(y_weights.dim(), (1, 1));
        assert_eq!(correlations.len(), 1);
    }
}
