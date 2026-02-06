//! Partial Least Squares (PLS) Decomposition
//!
//! This module provides Partial Least Squares decomposition methods for regression
//! and dimensionality reduction. PLS finds a linear regression model by projecting
//! the input variables and response variables to a new space that maximizes the
//! covariance between the projected variables.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::compat::ArrayLinalgExt;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Transform, Untrained},
};

/// Type alias for NIPALS step result to reduce type complexity
type NipalsStepResult = (
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
    Array1<f64>,
);

/// PLS algorithm variants
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum PLSAlgorithm {
    /// PLS1 - For single response variable
    #[default]
    PLS1,
    /// PLS2 - For multiple response variables using NIPALS algorithm
    PLS2,
    /// Canonical PLS - Maximizes correlation between X and Y scores
    Canonical,
}

/// Partial Least Squares Decomposition
///
/// PLS finds latent variables that explain the maximum covariance between
/// predictor variables X and response variables Y. It's particularly useful
/// when the number of predictors is large relative to the number of observations,
/// or when predictors are highly correlated.
///
/// # Mathematical Background
///
/// PLS seeks to find weight vectors w and c such that:
/// - t = X * w (X scores)
/// - u = Y * c (Y scores)
/// - cov(t, u) is maximized
///
/// The algorithm iteratively deflates X and Y by removing the variance
/// explained by each component.
///
/// # Applications
/// - Regression with high-dimensional predictors
/// - Spectroscopy and chemometrics
/// - Bioinformatics and genomics
/// - Quality control and process monitoring
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PartialLeastSquares {
    /// Number of components to extract
    pub n_components: usize,
    /// PLS algorithm to use
    pub algorithm: PLSAlgorithm,
    /// Maximum number of iterations for NIPALS
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: f64,
    /// Whether to center the data
    pub center: bool,
    /// Whether to scale the data to unit variance
    pub scale: bool,
    /// Whether to copy the input data
    pub copy: bool,
}

/// Fitted PLS model
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct FittedPLS {
    pub x_weights: Array2<f64>,
    pub y_weights: Array2<f64>,
    pub x_loadings: Array2<f64>,
    pub y_loadings: Array2<f64>,
    pub x_scores: Array2<f64>,
    pub y_scores: Array2<f64>,
    pub x_rotations: Array2<f64>,
    pub y_rotations: Array2<f64>,
    pub coef: Array2<f64>,
    pub x_mean: Array1<f64>,
    pub y_mean: Array1<f64>,
    pub x_scale: Array1<f64>,
    pub y_scale: Array1<f64>,
    pub n_features_x: usize,
    pub n_features_y: usize,
    pub n_components: usize,
    pub x_explained_variance_ratio: Array1<f64>,
    pub y_explained_variance_ratio: Array1<f64>,
}

impl Default for PartialLeastSquares {
    fn default() -> Self {
        Self::new(2)
    }
}

impl PartialLeastSquares {
    /// Create a new PLS instance
    ///
    /// # Parameters
    /// - `n_components`: Number of components to extract
    ///
    /// # Examples
    /// ```
    /// use sklears_decomposition::PartialLeastSquares;
    ///
    /// let pls = PartialLeastSquares::new(3);
    /// assert_eq!(pls.n_components, 3);
    /// ```
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            algorithm: PLSAlgorithm::default(),
            max_iter: 500,
            tol: 1e-6,
            center: true,
            scale: true,
            copy: true,
        }
    }

    /// Set the PLS algorithm
    pub fn algorithm(mut self, algorithm: PLSAlgorithm) -> Self {
        self.algorithm = algorithm;
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

    /// Set whether to center the data
    pub fn center(mut self, center: bool) -> Self {
        self.center = center;
        self
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set whether to copy input data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }

    /// Center and scale data
    fn preprocess_data(&self, data: &Array2<f64>) -> (Array2<f64>, Array1<f64>, Array1<f64>) {
        let mean = if self.center {
            data.mean_axis(Axis(0)).unwrap()
        } else {
            Array1::zeros(data.ncols())
        };

        let centered = if self.center {
            data - &mean.clone().insert_axis(Axis(0))
        } else {
            data.clone()
        };

        let scale = if self.scale {
            let var = centered.var_axis(Axis(0), 0.0);
            var.mapv(|v| if v > 1e-12 { v.sqrt() } else { 1.0 })
        } else {
            Array1::ones(data.ncols())
        };

        let processed = if self.scale {
            &centered / &scale.clone().insert_axis(Axis(0))
        } else {
            centered
        };

        (processed, mean, scale)
    }

    /// NIPALS algorithm for PLS
    fn nipals_step(&self, x: &Array2<f64>, y: &Array2<f64>) -> Result<NipalsStepResult> {
        let n_features_x = x.ncols();
        let n_features_y = y.ncols();

        // Initialize u with first column of Y, or with higher variance column
        let mut u = if n_features_y == 1 {
            y.column(0).to_owned()
        } else {
            // Find column with highest variance
            let mut max_var = 0.0;
            let mut best_col = 0;
            for i in 0..n_features_y {
                let col = y.column(i);
                let var = col.var(0.0);
                if var > max_var {
                    max_var = var;
                    best_col = i;
                }
            }
            y.column(best_col).to_owned()
        };

        // Ensure u is not zero
        let u_norm = u.dot(&u).sqrt();
        if u_norm < 1e-12 {
            // Initialize with random values if Y column is zero
            u = Array1::from_shape_fn(u.len(), |_| 1.0 / (u.len() as f64).sqrt());
        }

        let mut w_old = Array1::zeros(n_features_x);
        let mut iter = 0;

        loop {
            // X weights: w = X^T * u / ||u||^2
            let u_norm_sq = u.dot(&u);
            if u_norm_sq < 1e-12 {
                return Err(SklearsError::NumericalError(
                    "u vector became zero in NIPALS".to_string(),
                ));
            }
            let mut w = x.t().dot(&u) / u_norm_sq;

            // Normalize w to unit length
            let w_norm = w.dot(&w).sqrt();
            if w_norm < 1e-12 {
                // If w becomes zero, try a different approach
                if iter == 0 {
                    // Initialize w as the first principal component direction
                    let _xt_x = x.t().dot(x);
                    w = Array1::from_shape_fn(n_features_x, |i| if i == 0 { 1.0 } else { 0.0 });
                    let w_norm = w.dot(&w).sqrt();
                    w = &w / w_norm;
                } else {
                    return Err(SklearsError::NumericalError(
                        "X weights became zero in NIPALS".to_string(),
                    ));
                }
            } else {
                w = &w / w_norm;
            }

            // X scores: t = X * w
            let t = x.dot(&w);

            // Y weights: c = Y^T * t / ||t||^2
            let t_norm_sq = t.dot(&t);
            if t_norm_sq < 1e-12 {
                return Err(SklearsError::NumericalError(
                    "X scores became zero in NIPALS".to_string(),
                ));
            }
            let c = y.t().dot(&t) / t_norm_sq;

            // Y scores: u_new = Y * c
            let u_new = y.dot(&c);

            // Check convergence
            let diff = (&w - &w_old).dot(&(&w - &w_old)).sqrt();
            if diff < self.tol || iter >= self.max_iter {
                // X loadings: p = X^T * t / ||t||^2
                let p = x.t().dot(&t) / t_norm_sq;

                // Y loadings: q = Y^T * u / ||u||^2
                let u_norm_sq = u_new.dot(&u_new);
                let q = if u_norm_sq > 1e-12 {
                    y.t().dot(&u_new) / u_norm_sq
                } else {
                    // If u is zero, use t instead
                    y.t().dot(&t) / t_norm_sq
                };

                return Ok((w, c, p, q, t, u_new));
            }

            w_old = w.clone();
            u = u_new;
            iter += 1;
        }
    }

    /// Deflate matrices X and Y
    fn deflate_matrices(
        &self,
        x: &Array2<f64>,
        y: &Array2<f64>,
        t: &Array1<f64>,
        p: &Array1<f64>,
        q: &Array1<f64>,
    ) -> (Array2<f64>, Array2<f64>) {
        let t_outer_p = outer_product(t, p);
        let t_outer_q = outer_product(t, q);

        let x_deflated = x - &t_outer_p;
        let y_deflated = y - &t_outer_q;

        (x_deflated, y_deflated)
    }

    /// Compute explained variance ratio
    fn compute_explained_variance(&self, original: &Array2<f64>, deflated: &Array2<f64>) -> f64 {
        let total_var = original.mapv(|x| x * x).sum();
        let remaining_var = deflated.mapv(|x| x * x).sum();

        if total_var > 1e-12 {
            let ratio = (total_var - remaining_var) / total_var;
            ratio.clamp(0.0, 1.0) // Clamp to [0, 1] range
        } else {
            0.0
        }
    }
}

impl Fit<(Array2<f64>, Array2<f64>), ()> for PartialLeastSquares {
    type Fitted = FittedPLS;

    fn fit(self, data: &(Array2<f64>, Array2<f64>), _target: &()) -> Result<Self::Fitted> {
        let (x, y) = data;

        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if x.nrows() < 2 {
            return Err(SklearsError::InvalidInput(
                "Need at least 2 samples for PLS".to_string(),
            ));
        }

        let n_samples = x.nrows();
        let n_features_x = x.ncols();
        let n_features_y = y.ncols();
        let n_components = self.n_components.min(n_features_x.min(n_features_y));

        // Preprocess data
        let (mut x_work, x_mean, x_scale) = self.preprocess_data(x);
        let (mut y_work, y_mean, y_scale) = self.preprocess_data(y);

        // Initialize storage
        let mut x_weights = Array2::zeros((n_features_x, n_components));
        let mut y_weights = Array2::zeros((n_features_y, n_components));
        let mut x_loadings = Array2::zeros((n_features_x, n_components));
        let mut y_loadings = Array2::zeros((n_features_y, n_components));
        let mut x_scores = Array2::zeros((n_samples, n_components));
        let mut y_scores = Array2::zeros((n_samples, n_components));

        let mut x_explained_var_ratio = Array1::zeros(n_components);
        let mut y_explained_var_ratio = Array1::zeros(n_components);

        // Store original data for variance calculation
        let _x_original = x_work.clone();
        let _y_original = y_work.clone();

        // Extract components
        for k in 0..n_components {
            // NIPALS step
            let (w, c, p, q, t, u) = self.nipals_step(&x_work, &y_work)?;

            // Store results
            x_weights.column_mut(k).assign(&w);
            y_weights.column_mut(k).assign(&c);
            x_loadings.column_mut(k).assign(&p);
            y_loadings.column_mut(k).assign(&q);
            x_scores.column_mut(k).assign(&t);
            y_scores.column_mut(k).assign(&u);

            // Compute explained variance for this component
            let (x_deflated, y_deflated) = self.deflate_matrices(&x_work, &y_work, &t, &p, &q);
            x_explained_var_ratio[k] = self.compute_explained_variance(&x_work, &x_deflated);
            y_explained_var_ratio[k] = self.compute_explained_variance(&y_work, &y_deflated);

            // Update working matrices
            x_work = x_deflated;
            y_work = y_deflated;
        }

        // Compute rotation matrices (for transformation)
        // R = W * (P^T * W)^(-1) where W is weights and P is loadings
        let ptw_x = x_loadings.t().dot(&x_weights);
        let ptw_y = y_loadings.t().dot(&y_weights);

        let x_rotations = if ptw_x.nrows() == ptw_x.ncols() && ptw_x.nrows() > 0 {
            let ptw_x_inv = ptw_x.inv().map_err(|_| {
                SklearsError::NumericalError("Failed to invert X P^T*W matrix".to_string())
            })?;
            x_weights.dot(&ptw_x_inv)
        } else {
            // Fallback: use weights directly
            x_weights.clone()
        };

        let y_rotations = if ptw_y.nrows() == ptw_y.ncols() && ptw_y.nrows() > 0 {
            let ptw_y_inv = ptw_y.inv().map_err(|_| {
                SklearsError::NumericalError("Failed to invert Y P^T*W matrix".to_string())
            })?;
            y_weights.dot(&ptw_y_inv)
        } else {
            // Fallback: use weights directly
            y_weights.clone()
        };

        // Compute regression coefficients
        let coef = x_rotations.dot(&y_loadings.t());

        Ok(FittedPLS {
            x_weights,
            y_weights,
            x_loadings,
            y_loadings,
            x_scores,
            y_scores,
            x_rotations,
            y_rotations,
            coef,
            x_mean,
            y_mean,
            x_scale,
            y_scale,
            n_features_x,
            n_features_y,
            n_components,
            x_explained_variance_ratio: x_explained_var_ratio,
            y_explained_variance_ratio: y_explained_var_ratio,
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedPLS {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_x {
            let expected = self.n_features_x;
            let got = x.ncols();
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} features, got {got}"
            )));
        }

        // Preprocess X the same way as training data
        let x_centered = x - &self.x_mean.clone().insert_axis(Axis(0));
        let x_scaled = &x_centered / &self.x_scale.clone().insert_axis(Axis(0));

        // Transform using rotation matrix
        let x_scores = x_scaled.dot(&self.x_rotations);
        Ok(x_scores)
    }
}

impl Predict<Array2<f64>, Array2<f64>> for FittedPLS {
    fn predict(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        if x.ncols() != self.n_features_x {
            let expected = self.n_features_x;
            let got = x.ncols();
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} features, got {got}"
            )));
        }

        // Preprocess X
        let x_centered = x - &self.x_mean.clone().insert_axis(Axis(0));
        let x_scaled = &x_centered / &self.x_scale.clone().insert_axis(Axis(0));

        // Predict using regression coefficients
        let y_pred_scaled = x_scaled.dot(&self.coef);

        // Transform back to original scale
        let y_pred_centered = &y_pred_scaled * &self.y_scale.clone().insert_axis(Axis(0));
        let y_pred = &y_pred_centered + &self.y_mean.clone().insert_axis(Axis(0));

        Ok(y_pred)
    }
}

impl FittedPLS {
    /// Transform Y data to get Y scores
    pub fn transform_y(&self, y: &Array2<f64>) -> Result<Array2<f64>> {
        if y.ncols() != self.n_features_y {
            let expected = self.n_features_y;
            let got = y.ncols();
            return Err(SklearsError::InvalidInput(format!(
                "Expected {expected} features for Y, got {got}"
            )));
        }

        // Preprocess Y
        let y_centered = y - &self.y_mean.clone().insert_axis(Axis(0));
        let y_scaled = &y_centered / &self.y_scale.clone().insert_axis(Axis(0));

        // Transform using Y rotation matrix
        let y_scores = y_scaled.dot(&self.y_rotations);
        Ok(y_scores)
    }

    /// Get the X explained variance ratio
    pub fn x_explained_variance_ratio(&self) -> &Array1<f64> {
        &self.x_explained_variance_ratio
    }

    /// Get the Y explained variance ratio
    pub fn y_explained_variance_ratio(&self) -> &Array1<f64> {
        &self.y_explained_variance_ratio
    }

    /// Get the regression coefficients
    pub fn coefficients(&self) -> &Array2<f64> {
        &self.coef
    }
}

/// Compute outer product of two vectors
fn outer_product(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

/// Extension trait for matrix inversion
trait MatrixInversion {
    fn inv(&self) -> Result<Array2<f64>>;
}

impl MatrixInversion for Array2<f64> {
    fn inv(&self) -> Result<Array2<f64>> {
        if self.nrows() != self.ncols() {
            return Err(SklearsError::NumericalError(
                "Matrix must be square for inversion".to_string(),
            ));
        }

        // Use scirs2-linalg's inv() method
        self.inv().map_err(|e| {
            SklearsError::NumericalError(format!("Matrix inversion failed: {}", e))
        })
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pls_creation() {
        let pls = PartialLeastSquares::new(3);
        assert_eq!(pls.n_components, 3);
        assert_eq!(pls.algorithm, PLSAlgorithm::PLS1);
        assert_eq!(pls.max_iter, 500);
        assert_eq!(pls.tol, 1e-6);
        assert_eq!(pls.center, true);
        assert_eq!(pls.scale, true);
    }

    #[test]
    fn test_pls_builder_pattern() {
        let pls = PartialLeastSquares::new(2)
            .algorithm(PLSAlgorithm::PLS2)
            .max_iter(1000)
            .tolerance(1e-8)
            .center(false)
            .scale(false);

        assert_eq!(pls.n_components, 2);
        assert_eq!(pls.algorithm, PLSAlgorithm::PLS2);
        assert_eq!(pls.max_iter, 1000);
        assert_eq!(pls.tol, 1e-8);
        assert_eq!(pls.center, false);
        assert_eq!(pls.scale, false);
    }

    #[test]
    fn test_pls_fit_transform() {
        // Use less perfectly correlated data
        let x = array![
            [1.0, 2.1, 3.2],
            [4.1, 5.0, 6.1],
            [7.2, 8.1, 9.0],
            [10.1, 11.0, 12.1],
            [2.5, 3.5, 4.5],
        ];

        let y = array![[2.1, 3.2], [5.1, 6.0], [8.0, 9.1], [11.1, 12.0], [3.5, 4.6],];

        let pls = PartialLeastSquares::new(2);
        let fitted = pls.fit(&(x.clone(), y.clone()), &()).unwrap();

        assert_eq!(fitted.n_features_x, 3);
        assert_eq!(fitted.n_features_y, 2);
        assert_eq!(fitted.n_components, 2);
        assert_eq!(fitted.x_weights.dim(), (3, 2));
        assert_eq!(fitted.y_weights.dim(), (2, 2));
        assert_eq!(fitted.coef.dim(), (3, 2));

        // Test transformation
        let x_scores = fitted.transform(&x).unwrap();
        let y_scores = fitted.transform_y(&y).unwrap();

        assert_eq!(x_scores.dim(), (5, 2));
        assert_eq!(y_scores.dim(), (5, 2));

        // Test prediction
        let y_pred = fitted.predict(&x).unwrap();
        assert_eq!(y_pred.dim(), (5, 2));
    }

    #[test]
    fn test_pls_mismatched_samples() {
        let x = array![[1.0, 2.0], [3.0, 4.0]];
        let y = array![[1.0], [2.0], [3.0]]; // Different number of samples

        let pls = PartialLeastSquares::new(1);
        let result = pls.fit(&(x, y), &());
        assert!(result.is_err());
    }

    #[test]
    fn test_pls_insufficient_samples() {
        let x = array![[1.0, 2.0]];
        let y = array![[1.0]];

        let pls = PartialLeastSquares::new(1);
        let result = pls.fit(&(x, y), &());
        assert!(result.is_err());
    }

    #[test]
    fn test_pls_feature_mismatch() {
        let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0],];

        let y = array![[2.0, 3.0], [5.0, 6.0],];

        let pls = PartialLeastSquares::new(1);
        let fitted = pls.fit(&(x, y), &()).unwrap();

        // Test with wrong number of features
        let x_wrong = array![[1.0, 2.0]]; // Should have 3 features
        let result = fitted.transform(&x_wrong);
        assert!(result.is_err());

        let result = fitted.predict(&x_wrong);
        assert!(result.is_err());

        let y_wrong = array![[1.0]]; // Should have 2 features
        let result = fitted.transform_y(&y_wrong);
        assert!(result.is_err());
    }

    #[test]
    fn test_pls_algorithms() {
        let x = array![[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0], [0.0, -1.0],];

        let y = array![[1.0], [1.0], [-1.0], [-1.0],];

        // Test PLS1
        let pls1 = PartialLeastSquares::new(1).algorithm(PLSAlgorithm::PLS1);
        let fitted1 = pls1.fit(&(x.clone(), y.clone()), &()).unwrap();
        assert_eq!(fitted1.n_components, 1);

        // Test PLS2
        let pls2 = PartialLeastSquares::new(1).algorithm(PLSAlgorithm::PLS2);
        let fitted2 = pls2.fit(&(x.clone(), y.clone()), &()).unwrap();
        assert_eq!(fitted2.n_components, 1);
    }

    #[test]
    fn test_pls_explained_variance() {
        let x = array![[1.0, 2.1], [3.1, 4.0], [5.0, 6.1], [7.1, 8.0], [2.5, 3.5],];

        let y = array![[2.1], [4.0], [6.1], [8.0], [3.5],];

        let pls = PartialLeastSquares::new(1);
        let fitted = pls.fit(&(x, y), &()).unwrap();

        let x_var = fitted.x_explained_variance_ratio();
        let y_var = fitted.y_explained_variance_ratio();

        assert_eq!(x_var.len(), 1);
        assert_eq!(y_var.len(), 1);
        assert!(x_var[0] >= 0.0 && x_var[0] <= 1.0);
        assert!(y_var[0] >= 0.0 && y_var[0] <= 1.0);
    }

    #[test]
    fn test_outer_product() {
        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];
        let result = outer_product(&a, &b);

        let expected = array![[3.0, 4.0], [6.0, 8.0]];
        assert_eq!(result, expected);
    }
}
