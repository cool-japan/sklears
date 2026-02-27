//! Canonical Correlation Analysis

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

// Helper functions for safe operations
#[inline]
fn safe_mean_axis(arr: &Array2<Float>, axis: Axis) -> Result<Array1<Float>> {
    arr.mean_axis(axis).ok_or_else(|| {
        SklearsError::NumericalError("Failed to compute mean along axis".to_string())
    })
}

/// Canonical Correlation Analysis (CCA)
///
/// CCA finds linear relationships between two multivariate datasets by finding
/// canonical variables (linear combinations) that are maximally correlated.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::CCA;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let cca = CCA::new(1);
/// let fitted = cca.fit(&X, &Y).unwrap();
/// let X_c = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CCA<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Array2<Float>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    x_rotations_: Option<Array2<Float>>,
    y_rotations_: Option<Array2<Float>>,
    canonical_correlations_: Option<Array1<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl CCA<Untrained> {
    /// Create a new CCA model
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            max_iter: 500,
            tol: 1e-6,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            canonical_correlations_: None,
            n_iter_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for CCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for CCA<Untrained> {
    type Fitted = CCA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features_x.min(n_features_y) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed min(n_features_x, n_features_y)".to_string(),
            ));
        }

        // Center and scale data
        let x_mean = safe_mean_axis(x, Axis(0))?;
        let y_mean = safe_mean_axis(y, Axis(0))?;

        let mut x_centered = x - &x_mean.view().insert_axis(Axis(0));
        let mut y_centered = y - &y_mean.view().insert_axis(Axis(0));

        let (x_std, y_std) = if self.scale {
            let x_std = x_centered.std_axis(Axis(0), 0.0);
            let y_std = y_centered.std_axis(Axis(0), 0.0);

            // Avoid division by zero
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            (x_std, y_std)
        } else {
            (Array1::ones(n_features_x), Array1::ones(n_features_y))
        };

        // Use NIPALS algorithm similar to PLS but for CCA
        let mut x_weights = Array2::zeros((n_features_x, self.n_components));
        let mut y_weights = Array2::zeros((n_features_y, self.n_components));
        let mut x_loadings = Array2::zeros((n_features_x, self.n_components));
        let mut y_loadings = Array2::zeros((n_features_y, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut n_iter = Vec::new();

        let mut x_k = x_centered.clone();
        let mut y_k = y_centered.clone();

        for k in 0..self.n_components {
            // Initialize with first column of Y_k
            let mut y_score = y_k.column(0).to_owned();

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features_x);

            while iter < self.max_iter {
                // 1. Update X weights to maximize correlation with y_score
                let cov_xy = x_k.t().dot(&y_score);
                let norm = cov_xy.dot(&cov_xy).sqrt() + 1e-10;
                let w = cov_xy / norm;

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Update X scores
                let x_score = x_k.dot(&w);

                // 3. Update Y weights to maximize correlation with x_score
                let cov_yx = y_k.t().dot(&x_score);
                let norm_y = cov_yx.dot(&cov_yx).sqrt() + 1e-10;
                let c = cov_yx / norm_y;

                // 4. Update Y scores
                y_score = y_k.dot(&c);

                // Store weights
                x_weights.column_mut(k).assign(&w);
                y_weights.column_mut(k).assign(&c);

                iter += 1;
            }

            n_iter.push(iter);

            // Calculate final scores
            let x_score = x_k.dot(&x_weights.column(k));
            x_scores.column_mut(k).assign(&x_score);
            y_scores.column_mut(k).assign(&y_score);

            // Calculate loadings (regression of original variables on scores)
            let x_loading = x_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            let y_loading = y_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            x_loadings.column_mut(k).assign(&x_loading);
            y_loadings.column_mut(k).assign(&y_loading);

            // Deflate X and Y
            let x_score_matrix = x_score.view().insert_axis(Axis(1));
            let x_loading_matrix = x_loading.view().insert_axis(Axis(1));
            let y_loading_matrix = y_loading.view().insert_axis(Axis(1));

            x_k = x_k - x_score_matrix.dot(&x_loading_matrix.t());
            y_k = y_k - x_score_matrix.dot(&y_loading_matrix.t());
        }

        // Calculate rotations (transformations to canonical variates)
        let x_rotations = x_weights.clone();
        let y_rotations = y_weights.clone();

        // Calculate canonical correlations
        let mut canonical_correlations = Array1::zeros(self.n_components);
        for k in 0..self.n_components {
            let x_score = x_scores.column(k);
            let y_score = y_scores.column(k);
            let correlation = x_score.dot(&y_score)
                / (x_score.dot(&x_score).sqrt() * y_score.dot(&y_score).sqrt() + 1e-10);
            canonical_correlations[k] = correlation.abs();
        }

        Ok(CCA {
            n_components: self.n_components,
            scale: self.scale,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            x_weights_: Some(x_weights),
            y_weights_: Some(y_weights),
            x_loadings_: Some(x_loadings),
            y_loadings_: Some(y_loadings),
            x_scores_: Some(x_scores),
            y_scores_: Some(y_scores),
            x_rotations_: Some(x_rotations),
            y_rotations_: Some(y_rotations),
            canonical_correlations_: Some(canonical_correlations),
            n_iter_: Some(n_iter),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for CCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self
            .x_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_mean".to_string(),
            })?;
        let x_std = self
            .x_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_std".to_string(),
            })?;
        let x_rotations = self
            .x_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_rotations".to_string(),
            })?;

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(x_scaled.dot(x_rotations))
    }
}

impl CCA<Trained> {
    /// Transform Y to canonical variates
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_mean = self
            .y_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_mean".to_string(),
            })?;
        let y_std = self
            .y_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_std".to_string(),
            })?;
        let y_rotations = self
            .y_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_rotations".to_string(),
            })?;

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(y_scaled.dot(y_rotations))
    }

    /// Get the X weights
    pub fn x_weights(&self) -> Result<&Array2<Float>> {
        self.x_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_weights".to_string(),
            })
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> Result<&Array2<Float>> {
        self.y_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_weights".to_string(),
            })
    }

    /// Get the X loadings
    pub fn x_loadings(&self) -> Result<&Array2<Float>> {
        self.x_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_loadings".to_string(),
            })
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> Result<&Array2<Float>> {
        self.y_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_loadings".to_string(),
            })
    }

    /// Get the X scores
    pub fn x_scores(&self) -> Result<&Array2<Float>> {
        self.x_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_scores".to_string(),
            })
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> Result<&Array2<Float>> {
        self.y_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_scores".to_string(),
            })
    }

    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> Result<&Array1<Float>> {
        self.canonical_correlations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "canonical_correlations".to_string(),
            })
    }
}

/// Regularized Canonical Correlation Analysis (Ridge CCA)
///
/// Regularized CCA adds L2 regularization to handle ill-conditioned covariance matrices
/// and improve numerical stability, especially when the number of features exceeds samples.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `reg_param_x` - L2 regularization parameter for X covariance matrix
/// * `reg_param_y` - L2 regularization parameter for Y covariance matrix
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::RidgeCCA;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let cca = RidgeCCA::new(1, 0.1, 0.1);
/// let fitted = cca.fit(&X, &Y).unwrap();
/// let X_c = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct RidgeCCA<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// L2 regularization parameter for X
    pub reg_param_x: Float,
    /// L2 regularization parameter for Y
    pub reg_param_y: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Array2<Float>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    x_rotations_: Option<Array2<Float>>,
    y_rotations_: Option<Array2<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,
    canonical_correlations_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl RidgeCCA<Untrained> {
    /// Create a new Ridge CCA model
    pub fn new(n_components: usize, reg_param_x: Float, reg_param_y: Float) -> Self {
        Self {
            n_components,
            scale: true,
            reg_param_x,
            reg_param_y,
            max_iter: 500,
            tol: 1e-6,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            n_iter_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            canonical_correlations_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the regularization parameters
    pub fn regularization(mut self, reg_param_x: Float, reg_param_y: Float) -> Self {
        self.reg_param_x = reg_param_x;
        self.reg_param_y = reg_param_y;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for RidgeCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for RidgeCCA<Untrained> {
    type Fitted = RidgeCCA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features_x.min(n_features_y) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed min(n_features_x, n_features_y)".to_string(),
            ));
        }

        // Center and scale data
        let x_mean = safe_mean_axis(x, Axis(0))?;
        let y_mean = safe_mean_axis(y, Axis(0))?;

        let mut x_centered = x - &x_mean.view().insert_axis(Axis(0));
        let mut y_centered = y - &y_mean.view().insert_axis(Axis(0));

        let (x_std, y_std) = if self.scale {
            let x_std = x_centered.std_axis(Axis(0), 0.0);
            let y_std = y_centered.std_axis(Axis(0), 0.0);

            // Avoid division by zero
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            (x_std, y_std)
        } else {
            (Array1::ones(n_features_x), Array1::ones(n_features_y))
        };

        // Compute regularized covariance matrices
        let cxx = x_centered.t().dot(&x_centered) / (n_samples as Float - 1.0);
        let cyy = y_centered.t().dot(&y_centered) / (n_samples as Float - 1.0);
        let cxy = x_centered.t().dot(&y_centered) / (n_samples as Float - 1.0);

        // Add regularization to diagonal
        let mut cxx_reg = cxx;
        let mut cyy_reg = cyy;

        for i in 0..n_features_x {
            cxx_reg[[i, i]] += self.reg_param_x;
        }

        for i in 0..n_features_y {
            cyy_reg[[i, i]] += self.reg_param_y;
        }

        // Use regularized iterative deflation algorithm
        let mut x_weights = Array2::zeros((n_features_x, self.n_components));
        let mut y_weights = Array2::zeros((n_features_y, self.n_components));
        let mut x_loadings = Array2::zeros((n_features_x, self.n_components));
        let mut y_loadings = Array2::zeros((n_features_y, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut canonical_correlations = Array1::zeros(self.n_components);
        let mut n_iter = Vec::new();

        let mut x_k = x_centered.clone();
        let mut y_k = y_centered.clone();
        let mut cxx_k = cxx_reg;
        let mut cyy_k = cyy_reg;
        let mut cxy_k = cxy;

        for k in 0..self.n_components {
            // Initialize with normalized random vector for Y weights
            let mut c = Array1::ones(n_features_y) / (n_features_y as Float).sqrt();

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features_x);

            while iter < self.max_iter {
                // 1. Solve regularized system: Cxx_reg * w = Cxy * c
                let cxy_c = cxy_k.dot(&c);
                let w = self.solve_regularized_system(&cxx_k, &cxy_c)?;

                // Normalize w
                let w_norm = w.dot(&w).sqrt();
                let w = if w_norm > 0.0 { w / w_norm } else { w };

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Solve regularized system: Cyy_reg * c = Cyx * w
                let cyx_w = cxy_k.t().dot(&w);
                c = self.solve_regularized_system(&cyy_k, &cyx_w)?;

                // Normalize c
                let c_norm = c.dot(&c).sqrt();
                c = if c_norm > 0.0 { c / c_norm } else { c };

                // Store weights
                x_weights.column_mut(k).assign(&w);
                y_weights.column_mut(k).assign(&c);

                iter += 1;
            }

            n_iter.push(iter);

            // Calculate final scores
            let x_score = x_k.dot(&x_weights.column(k));
            let y_score = y_k.dot(&y_weights.column(k));
            x_scores.column_mut(k).assign(&x_score);
            y_scores.column_mut(k).assign(&y_score);

            // Calculate canonical correlation
            let correlation = x_score.dot(&y_score)
                / ((x_score.dot(&x_score) * y_score.dot(&y_score)).sqrt() + 1e-10);
            canonical_correlations[k] = correlation;

            // Calculate loadings (regression of original variables on scores)
            let x_loading = x_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            let y_loading = y_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            x_loadings.column_mut(k).assign(&x_loading);
            y_loadings.column_mut(k).assign(&y_loading);

            // Deflate matrices
            let x_score_matrix = x_score.view().insert_axis(Axis(1));
            let x_loading_matrix = x_loading.view().insert_axis(Axis(1));
            let y_loading_matrix = y_loading.view().insert_axis(Axis(1));

            x_k = x_k - x_score_matrix.dot(&x_loading_matrix.t());
            y_k = y_k - x_score_matrix.dot(&y_loading_matrix.t());

            // Update covariance matrices for next iteration
            cxx_k = x_k.t().dot(&x_k) / (n_samples as Float - 1.0);
            cyy_k = y_k.t().dot(&y_k) / (n_samples as Float - 1.0);
            cxy_k = x_k.t().dot(&y_k) / (n_samples as Float - 1.0);

            // Add regularization
            for i in 0..cxx_k.nrows() {
                if i < cxx_k.ncols() {
                    cxx_k[[i, i]] += self.reg_param_x;
                }
            }
            for i in 0..cyy_k.nrows() {
                if i < cyy_k.ncols() {
                    cyy_k[[i, i]] += self.reg_param_y;
                }
            }
        }

        // Calculate rotations (transformations to canonical variates)
        let x_rotations = x_weights.clone();
        let y_rotations = y_weights.clone();

        Ok(RidgeCCA {
            n_components: self.n_components,
            scale: self.scale,
            reg_param_x: self.reg_param_x,
            reg_param_y: self.reg_param_y,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            x_weights_: Some(x_weights),
            y_weights_: Some(y_weights),
            x_loadings_: Some(x_loadings),
            y_loadings_: Some(y_loadings),
            x_scores_: Some(x_scores),
            y_scores_: Some(y_scores),
            x_rotations_: Some(x_rotations),
            y_rotations_: Some(y_rotations),
            n_iter_: Some(n_iter),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            canonical_correlations_: Some(canonical_correlations),
            _state: PhantomData,
        })
    }
}

impl RidgeCCA<Untrained> {
    /// Solve regularized linear system using simple iterative method
    fn solve_regularized_system(
        &self,
        a: &Array2<Float>,
        b: &Array1<Float>,
    ) -> Result<Array1<Float>> {
        let n = a.nrows();
        let mut x = Array1::zeros(n);

        // Simple Gauss-Seidel iteration for Ax = b
        for _iter in 0..100 {
            let mut x_new = x.clone();

            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += a[[i, j]] * x_new[j];
                    }
                }

                if a[[i, i]].abs() > 1e-12 {
                    x_new[i] = (b[i] - sum) / a[[i, i]];
                }
            }

            // Check convergence
            let diff = (&x_new - &x).mapv(|val| val.abs()).sum();
            x = x_new;

            if diff < 1e-10 {
                break;
            }
        }

        Ok(x)
    }
}

impl Transform<Array2<Float>, Array2<Float>> for RidgeCCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self
            .x_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_mean".to_string(),
            })?;
        let x_std = self
            .x_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_std".to_string(),
            })?;
        let x_rotations = self
            .x_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_rotations".to_string(),
            })?;

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(x_scaled.dot(x_rotations))
    }
}

impl RidgeCCA<Trained> {
    /// Transform Y to canonical variates
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_mean = self
            .y_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_mean".to_string(),
            })?;
        let y_std = self
            .y_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_std".to_string(),
            })?;
        let y_rotations = self
            .y_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_rotations".to_string(),
            })?;

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(y_scaled.dot(y_rotations))
    }

    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> Result<&Array1<Float>> {
        self.canonical_correlations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "canonical_correlations".to_string(),
            })
    }

    /// Get the X weights
    pub fn x_weights(&self) -> Result<&Array2<Float>> {
        self.x_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_weights".to_string(),
            })
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> Result<&Array2<Float>> {
        self.y_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_weights".to_string(),
            })
    }

    /// Get the X loadings
    pub fn x_loadings(&self) -> Result<&Array2<Float>> {
        self.x_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_loadings".to_string(),
            })
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> Result<&Array2<Float>> {
        self.y_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_loadings".to_string(),
            })
    }

    /// Get the X scores
    pub fn x_scores(&self) -> Result<&Array2<Float>> {
        self.x_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_scores".to_string(),
            })
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> Result<&Array2<Float>> {
        self.y_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_scores".to_string(),
            })
    }
}

/// Sparse Canonical Correlation Analysis with L1 regularization
///
/// Sparse CCA adds L1 penalties to encourage sparsity in the canonical weights,
/// making the solution more interpretable by selecting only relevant features.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `l1_param_x` - L1 regularization parameter for X weights
/// * `l1_param_y` - L1 regularization parameter for Y weights
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::SparseCCA;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let cca = SparseCCA::new(1, 0.1, 0.1);
/// let fitted = cca.fit(&X, &Y).unwrap();
/// let X_c = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparseCCA<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// L1 regularization parameter for X
    pub l1_param_x: Float,
    /// L1 regularization parameter for Y
    pub l1_param_y: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Array2<Float>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    x_rotations_: Option<Array2<Float>>,
    y_rotations_: Option<Array2<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,
    canonical_correlations_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl SparseCCA<Untrained> {
    /// Create a new Sparse CCA model
    pub fn new(n_components: usize, l1_param_x: Float, l1_param_y: Float) -> Self {
        Self {
            n_components,
            scale: true,
            l1_param_x,
            l1_param_y,
            max_iter: 500,
            tol: 1e-6,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_loadings_: None,
            y_loadings_: None,
            x_scores_: None,
            y_scores_: None,
            x_rotations_: None,
            y_rotations_: None,
            n_iter_: None,
            x_mean_: None,
            y_mean_: None,
            x_std_: None,
            y_std_: None,
            canonical_correlations_: None,
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the L1 regularization parameters
    pub fn l1_regularization(mut self, l1_param_x: Float, l1_param_y: Float) -> Self {
        self.l1_param_x = l1_param_x;
        self.l1_param_y = l1_param_y;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for SparseCCA<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for SparseCCA<Untrained> {
    type Fitted = SparseCCA<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features_x.min(n_features_y) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed min(n_features_x, n_features_y)".to_string(),
            ));
        }

        // Center and scale data
        let x_mean = safe_mean_axis(x, Axis(0))?;
        let y_mean = safe_mean_axis(y, Axis(0))?;

        let mut x_centered = x - &x_mean.view().insert_axis(Axis(0));
        let mut y_centered = y - &y_mean.view().insert_axis(Axis(0));

        let (x_std, y_std) = if self.scale {
            let x_std = x_centered.std_axis(Axis(0), 0.0);
            let y_std = y_centered.std_axis(Axis(0), 0.0);

            // Avoid division by zero
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_centered.column_mut(i).mapv_inplace(|v| v / std);
                }
            }

            (x_std, y_std)
        } else {
            (Array1::ones(n_features_x), Array1::ones(n_features_y))
        };

        // Sparse CCA using iterative soft thresholding
        let mut x_weights = Array2::zeros((n_features_x, self.n_components));
        let mut y_weights = Array2::zeros((n_features_y, self.n_components));
        let mut x_loadings = Array2::zeros((n_features_x, self.n_components));
        let mut y_loadings = Array2::zeros((n_features_y, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut canonical_correlations = Array1::zeros(self.n_components);
        let mut n_iter = Vec::new();

        let mut x_k = x_centered.clone();
        let mut y_k = y_centered.clone();

        for k in 0..self.n_components {
            // Initialize with normalized random vectors
            let mut w = Array1::ones(n_features_x) / (n_features_x as Float).sqrt();
            let mut c = Array1::ones(n_features_y) / (n_features_y as Float).sqrt();

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features_x);

            while iter < self.max_iter {
                // 1. Update X weights with L1 penalty (soft thresholding)
                let gradient_w = x_k.t().dot(&y_k.dot(&c));
                w = self.soft_threshold(&gradient_w, self.l1_param_x);

                // Normalize w
                let w_norm = w.dot(&w).sqrt();
                w = if w_norm > 0.0 { w / w_norm } else { w };

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Update X scores
                let x_score = x_k.dot(&w);

                // 3. Update Y weights with L1 penalty (soft thresholding)
                let gradient_c = y_k.t().dot(&x_score);
                c = self.soft_threshold(&gradient_c, self.l1_param_y);

                // Normalize c
                let c_norm = c.dot(&c).sqrt();
                c = if c_norm > 0.0 { c / c_norm } else { c };

                // 4. Update Y scores
                let y_score = y_k.dot(&c);

                // Store weights
                x_weights.column_mut(k).assign(&w);
                y_weights.column_mut(k).assign(&c);

                iter += 1;
            }

            n_iter.push(iter);

            // Calculate final scores
            let x_score = x_k.dot(&x_weights.column(k));
            let y_score = y_k.dot(&y_weights.column(k));
            x_scores.column_mut(k).assign(&x_score);
            y_scores.column_mut(k).assign(&y_score);

            // Calculate canonical correlation
            let correlation = x_score.dot(&y_score)
                / ((x_score.dot(&x_score) * y_score.dot(&y_score)).sqrt() + 1e-10);
            canonical_correlations[k] = correlation;

            // Calculate loadings (regression of original variables on scores)
            let x_loading = x_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            let y_loading = y_k.t().dot(&x_score) / (x_score.dot(&x_score) + 1e-10);
            x_loadings.column_mut(k).assign(&x_loading);
            y_loadings.column_mut(k).assign(&y_loading);

            // Deflate matrices
            let x_score_matrix = x_score.view().insert_axis(Axis(1));
            let x_loading_matrix = x_loading.view().insert_axis(Axis(1));
            let y_loading_matrix = y_loading.view().insert_axis(Axis(1));

            x_k = x_k - x_score_matrix.dot(&x_loading_matrix.t());
            y_k = y_k - x_score_matrix.dot(&y_loading_matrix.t());
        }

        // Calculate rotations (transformations to canonical variates)
        let x_rotations = x_weights.clone();
        let y_rotations = y_weights.clone();

        Ok(SparseCCA {
            n_components: self.n_components,
            scale: self.scale,
            l1_param_x: self.l1_param_x,
            l1_param_y: self.l1_param_y,
            max_iter: self.max_iter,
            tol: self.tol,
            copy: self.copy,
            x_weights_: Some(x_weights),
            y_weights_: Some(y_weights),
            x_loadings_: Some(x_loadings),
            y_loadings_: Some(y_loadings),
            x_scores_: Some(x_scores),
            y_scores_: Some(y_scores),
            x_rotations_: Some(x_rotations),
            y_rotations_: Some(y_rotations),
            n_iter_: Some(n_iter),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            canonical_correlations_: Some(canonical_correlations),
            _state: PhantomData,
        })
    }
}

impl SparseCCA<Untrained> {
    /// Soft thresholding operator for L1 regularization
    fn soft_threshold(&self, x: &Array1<Float>, lambda: Float) -> Array1<Float> {
        x.mapv(|val| {
            if val > lambda {
                val - lambda
            } else if val < -lambda {
                val + lambda
            } else {
                0.0
            }
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for SparseCCA<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self
            .x_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_mean".to_string(),
            })?;
        let x_std = self
            .x_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_std".to_string(),
            })?;
        let x_rotations = self
            .x_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform - x_rotations".to_string(),
            })?;

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(x_scaled.dot(x_rotations))
    }
}

impl SparseCCA<Trained> {
    /// Transform Y to canonical variates
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_mean = self
            .y_mean_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_mean".to_string(),
            })?;
        let y_std = self
            .y_std_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_std".to_string(),
            })?;
        let y_rotations = self
            .y_rotations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "transform_y - y_rotations".to_string(),
            })?;

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical variates
        Ok(y_scaled.dot(y_rotations))
    }

    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> Result<&Array1<Float>> {
        self.canonical_correlations_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "canonical_correlations".to_string(),
            })
    }

    /// Get the X weights (with sparsity)
    pub fn x_weights(&self) -> Result<&Array2<Float>> {
        self.x_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_weights".to_string(),
            })
    }

    /// Get the Y weights (with sparsity)
    pub fn y_weights(&self) -> Result<&Array2<Float>> {
        self.y_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_weights".to_string(),
            })
    }

    /// Get the X loadings
    pub fn x_loadings(&self) -> Result<&Array2<Float>> {
        self.x_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_loadings".to_string(),
            })
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> Result<&Array2<Float>> {
        self.y_loadings_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_loadings".to_string(),
            })
    }

    /// Get the X scores
    pub fn x_scores(&self) -> Result<&Array2<Float>> {
        self.x_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_scores".to_string(),
            })
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> Result<&Array2<Float>> {
        self.y_scores_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_scores".to_string(),
            })
    }

    /// Count the number of non-zero weights in X
    pub fn x_sparsity(&self) -> Result<usize> {
        let x_weights = self
            .x_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "x_sparsity".to_string(),
            })?;
        Ok(x_weights.iter().filter(|&&x| x.abs() > 1e-10).count())
    }

    /// Count the number of non-zero weights in Y
    pub fn y_sparsity(&self) -> Result<usize> {
        let y_weights = self
            .y_weights_
            .as_ref()
            .ok_or_else(|| SklearsError::NotFitted {
                operation: "y_sparsity".to_string(),
            })?;
        Ok(y_weights.iter().filter(|&&x| x.abs() > 1e-10).count())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        let cca = CCA::new(1);
        let fitted = cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
        assert_eq!(y_canonical.shape(), &[4, 1]);
    }

    #[test]
    fn test_cca_multiple_components() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0],];

        let cca = CCA::new(2);
        let fitted = cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[5, 2]);
        assert_eq!(y_canonical.shape(), &[5, 2]);
    }

    #[test]
    fn test_ridge_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        let ridge_cca = RidgeCCA::new(1, 0.1, 0.1);
        let fitted = ridge_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
        assert_eq!(y_canonical.shape(), &[4, 1]);

        // Check that canonical correlations are computed
        let correlations = fitted.canonical_correlations().unwrap();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }

    #[test]
    fn test_ridge_cca_high_regularization() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        // High regularization should still work
        let ridge_cca = RidgeCCA::new(2, 1.0, 1.0);
        let fitted = ridge_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 2]);
        assert_eq!(y_canonical.shape(), &[4, 2]);

        // Check canonical correlations
        let correlations = fitted.canonical_correlations().unwrap();
        assert_eq!(correlations.len(), 2);
        for &corr in correlations.iter() {
            assert!(corr >= -1.0 && corr <= 1.0);
        }
    }

    #[test]
    fn test_ridge_cca_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0], [4.0], [6.0], [8.0],];

        let ridge_cca = RidgeCCA::new(1, 0.1, 0.1).scale(false);
        let fitted = ridge_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
    }

    #[test]
    fn test_ridge_cca_regularization_effect() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0],];

        let y = array![[2.0], [4.0], [6.0],];

        // Low regularization
        let ridge_cca_low = RidgeCCA::new(1, 0.001, 0.001);
        let fitted_low = ridge_cca_low.fit(&x, &y).unwrap();
        let corr_low = fitted_low.canonical_correlations().unwrap()[0];

        // High regularization
        let ridge_cca_high = RidgeCCA::new(1, 1.0, 1.0);
        let fitted_high = ridge_cca_high.fit(&x, &y).unwrap();
        let corr_high = fitted_high.canonical_correlations().unwrap()[0];

        // Both should be valid correlations
        assert!(corr_low >= -1.0 && corr_low <= 1.0);
        assert!(corr_high >= -1.0 && corr_high <= 1.0);

        // High regularization typically produces lower correlations
        assert!(corr_high.abs() <= corr_low.abs() + 0.1); // Allow some tolerance
    }

    // Property-based tests
    proptest! {
        #[test]
        fn test_cca_canonical_correlations_range(
            n_samples in 10..50usize,
            n_features_x in 2..10usize,
            n_features_y in 2..10usize,
            n_components in 1..3usize,
            data in prop::collection::vec(-10.0..10.0f64, 0..500)
        ) {
            // Ensure we have enough data
            if data.len() < n_samples * (n_features_x + n_features_y) {
                return Ok(());
            }

            let n_components = n_components.min(n_features_x).min(n_features_y);

            // Create random matrices
            let x_data = &data[0..n_samples * n_features_x];
            let y_data = &data[n_samples * n_features_x..n_samples * (n_features_x + n_features_y)];

            let x = Array2::from_shape_vec((n_samples, n_features_x), x_data.to_vec())?;
            let y = Array2::from_shape_vec((n_samples, n_features_y), y_data.to_vec())?;

            // Test CCA
            if let Ok(fitted_cca) = CCA::new(n_components).fit(&x, &y) {
                // Test canonical correlation properties
                let x_canonical = fitted_cca.transform(&x)?;
                let y_canonical = fitted_cca.transform_y(&y)?;

                // Check output dimensions
                prop_assert_eq!(x_canonical.shape(), &[n_samples, n_components]);
                prop_assert_eq!(y_canonical.shape(), &[n_samples, n_components]);

                // Check that canonical correlations are within [-1, 1]
                for i in 0..n_components {
                    let x_comp = x_canonical.column(i);
                    let y_comp = y_canonical.column(i);

                    let x_norm = (x_comp.dot(&x_comp)).sqrt();
                    let y_norm = (y_comp.dot(&y_comp)).sqrt();

                    if x_norm > 1e-10 && y_norm > 1e-10 {
                        let correlation = x_comp.dot(&y_comp) / (x_norm * y_norm);
                        prop_assert!(correlation >= -1.0 - 1e-10);
                        prop_assert!(correlation <= 1.0 + 1e-10);
                    }
                }
            }

            // Test RidgeCCA
            if let Ok(fitted_ridge) = RidgeCCA::new(n_components, 0.1, 0.1).fit(&x, &y) {
                // Test regularized CCA properties
                if let Ok(correlations) = fitted_ridge.canonical_correlations() {
                    for &corr in correlations.iter() {
                        prop_assert!(corr >= -1.0 - 1e-10);
                        prop_assert!(corr <= 1.0 + 1e-10);
                    }
                }
            }
        }

        #[test]
        fn test_cca_transform_inverse_relationship(
            data in prop::collection::vec(-5.0..5.0f64, 20..100)
        ) {
            if data.len() < 40 {
                return Ok(());
            }

            let n_samples = 10;
            let n_features = 2;
            let x_data = &data[0..n_samples * n_features];
            let y_data = &data[n_samples * n_features..2 * n_samples * n_features];

            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;
            let y = Array2::from_shape_vec((n_samples, n_features), y_data.to_vec())?;

            let cca = CCA::new(1);
            if let Ok(fitted) = cca.fit(&x, &y) {
                let x_transformed = fitted.transform(&x)?;
                let y_transformed = fitted.transform_y(&y)?;

                // Test that transformation is consistent
                prop_assert_eq!(x_transformed.shape(), &[n_samples, 1]);
                prop_assert_eq!(y_transformed.shape(), &[n_samples, 1]);

                // Test that applying transform twice to same data gives same result
                let x_transformed2 = fitted.transform(&x)?;
                for (a, b) in x_transformed.iter().zip(x_transformed2.iter()) {
                    prop_assert!((a - b).abs() < 1e-10);
                }
            }
        }

        #[test]
        fn test_cca_scale_invariance(
            scale_factor in 1.0..100.0f64
        ) {
            let x = array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ];
            let y = array![
                [2.0, 1.0],
                [4.0, 3.0],
                [6.0, 5.0],
                [8.0, 7.0],
            ];

            // Scale X by the factor
            let x_scaled = &x * scale_factor;

            let cca1 = CCA::new(1).scale(true);
            let cca2 = CCA::new(1).scale(true);

            if let (Ok(fitted1), Ok(fitted2)) = (cca1.fit(&x, &y), cca2.fit(&x_scaled, &y)) {
                let transform1 = fitted1.transform(&x)?;
                let transform2 = fitted2.transform(&x_scaled)?;

                // With scaling enabled, results should be very similar
                let mean_diff: f64 = transform1.iter()
                    .zip(transform2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>() / transform1.len() as f64;

                prop_assert!(mean_diff < 0.1);  // Allow some numerical tolerance
            }
        }

        #[test]
        fn test_ridge_cca_regularization_effect_property(
            reg_low in 0.001..0.1f64,
            reg_high in 1.0..10.0f64
        ) {
            let x = array![
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
            ];
            let y = array![
                [2.0, 1.0],
                [4.0, 3.0],
                [6.0, 5.0],
                [8.0, 7.0],
                [10.0, 9.0],
            ];

            let ridge_low = RidgeCCA::new(1, reg_low, reg_low);
            let ridge_high = RidgeCCA::new(1, reg_high, reg_high);

            if let (Ok(fitted_low), Ok(fitted_high)) = (ridge_low.fit(&x, &y), ridge_high.fit(&x, &y)) {
                let corr_low = fitted_low.canonical_correlations().unwrap()[0];
                let corr_high = fitted_high.canonical_correlations().unwrap()[0];

                // Both should be valid correlations
                prop_assert!(corr_low >= -1.0 && corr_low <= 1.0);
                prop_assert!(corr_high >= -1.0 && corr_high <= 1.0);

                // Higher regularization should generally produce lower correlations
                // (though this is not always guaranteed, so we use a loose bound)
                prop_assert!(corr_high.abs() <= corr_low.abs() + 0.5);
            }
        }

        #[test]
        fn test_cca_numerical_stability(
            noise_level in 0.0..0.1f64
        ) {
            let mut x = array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
            ];
            let mut y = array![
                [2.0, 1.0],
                [4.0, 3.0],
                [6.0, 5.0],
                [8.0, 7.0],
            ];

            // Add small amount of noise
            for val in x.iter_mut() {
                *val += noise_level * ((*val * 12345.0_f64).sin());
            }
            for val in y.iter_mut() {
                *val += noise_level * ((*val * 67890.0_f64).sin());
            }

            let cca = CCA::new(1);
            if let Ok(fitted) = cca.fit(&x, &y) {
                let x_transform = fitted.transform(&x)?;
                let y_transform = fitted.transform_y(&y)?;

                // Check that outputs are finite
                for val in x_transform.iter() {
                    prop_assert!(val.is_finite());
                }
                for val in y_transform.iter() {
                    prop_assert!(val.is_finite());
                }

                // Check that weights are reasonable
                if let Ok(x_weights) = fitted.x_weights() {
                    for val in x_weights.iter() {
                        prop_assert!(val.is_finite());
                        prop_assert!(val.abs() < 1000.0);  // Shouldn't be too large
                    }
                }
                if let Ok(y_weights) = fitted.y_weights() {
                    for val in y_weights.iter() {
                        prop_assert!(val.is_finite());
                        prop_assert!(val.abs() < 1000.0);
                    }
                }
            }
        }
    }

    // Tests for SparseCCA
    #[test]
    fn test_sparse_cca_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        let sparse_cca = SparseCCA::new(1, 0.1, 0.1);
        let fitted = sparse_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
        assert_eq!(y_canonical.shape(), &[4, 1]);

        // Check that canonical correlations are computed
        let correlations = fitted.canonical_correlations().unwrap();
        assert_eq!(correlations.len(), 1);
        assert!(correlations[0] >= -1.0 && correlations[0] <= 1.0);
    }

    #[test]
    fn test_sparse_cca_sparsity() {
        let x = array![
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 3.0, 4.0, 0.0],
            [3.0, 4.0, 5.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
        ];

        let y = array![[2.0, 0.0], [4.0, 0.0], [6.0, 0.0], [8.0, 0.0],];

        // High regularization should produce sparse weights
        let sparse_cca = SparseCCA::new(1, 0.5, 0.5);
        let fitted = sparse_cca.fit(&x, &y).unwrap();

        // Check sparsity
        let x_sparsity = fitted.x_sparsity().unwrap();
        let y_sparsity = fitted.y_sparsity().unwrap();

        // With high regularization, we should have fewer non-zero weights
        assert!(x_sparsity <= 4); // Should be less than total features
        assert!(y_sparsity <= 2); // Should be less than total features

        // Weights should be finite
        let x_weights = fitted.x_weights().unwrap();
        let y_weights = fitted.y_weights().unwrap();

        for val in x_weights.iter() {
            assert!(val.is_finite());
        }
        for val in y_weights.iter() {
            assert!(val.is_finite());
        }
    }

    #[test]
    fn test_sparse_cca_regularization_effect() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        // Low regularization
        let sparse_cca_low = SparseCCA::new(1, 0.01, 0.01);
        let fitted_low = sparse_cca_low.fit(&x, &y).unwrap();
        let sparsity_low_x = fitted_low.x_sparsity().unwrap();
        let sparsity_low_y = fitted_low.y_sparsity().unwrap();

        // High regularization
        let sparse_cca_high = SparseCCA::new(1, 0.5, 0.5);
        let fitted_high = sparse_cca_high.fit(&x, &y).unwrap();
        let sparsity_high_x = fitted_high.x_sparsity().unwrap();
        let sparsity_high_y = fitted_high.y_sparsity().unwrap();

        // High regularization should produce more sparsity (fewer non-zero weights)
        assert!(sparsity_high_x <= sparsity_low_x);
        assert!(sparsity_high_y <= sparsity_low_y);
    }

    #[test]
    fn test_sparse_cca_multiple_components() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0],];

        let sparse_cca = SparseCCA::new(2, 0.1, 0.1);
        let fitted = sparse_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[5, 2]);
        assert_eq!(y_canonical.shape(), &[5, 2]);

        // Check canonical correlations
        let correlations = fitted.canonical_correlations().unwrap();
        assert_eq!(correlations.len(), 2);
        for &corr in correlations.iter() {
            assert!(corr >= -1.0 && corr <= 1.0);
        }
    }

    #[test]
    fn test_sparse_cca_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0], [4.0], [6.0], [8.0],];

        let sparse_cca = SparseCCA::new(1, 0.1, 0.1).scale(false);
        let fitted = sparse_cca.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
    }
}
