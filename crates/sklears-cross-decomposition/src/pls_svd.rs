//! PLS using SVD

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Partial Least Squares using Singular Value Decomposition
///
/// PLSSVD implements PLS using SVD decomposition which is more numerically stable
/// for certain cases than the iterative NIPALS algorithm.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::PLSSVD;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let pls = PLSSVD::new(1);
/// let fitted = pls.fit(&X, &Y).unwrap();
/// let X_transformed = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PLSSVD<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl PLSSVD<Untrained> {
    /// Create a new PLSSVD model
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            copy: true,
            x_weights_: None,
            y_weights_: None,
            x_scores_: None,
            y_scores_: None,
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

    /// Set whether to copy the data
    pub fn copy(mut self, copy: bool) -> Self {
        self.copy = copy;
        self
    }
}

impl Estimator for PLSSVD<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for PLSSVD<Untrained> {
    type Fitted = PLSSVD<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features_x.min(n_features_y).min(n_samples) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed min(n_features_x, n_features_y, n_samples)".to_string(),
            ));
        }

        // Center and scale data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();

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

        // Compute cross-covariance matrix C = X^T * Y
        let c = x_centered.t().dot(&y_centered);

        // Perform SVD of C = U * Σ * V^T
        // For simplicity, we'll use a power iteration method to find the leading singular vectors
        let (u, v, _singular_values) = self.svd_power_iteration(&c, self.n_components)?;

        // The weights are the left and right singular vectors
        let x_weights = u;
        let y_weights = v;

        // Compute scores (projections onto the weight vectors)
        let x_scores = x_centered.dot(&x_weights);
        let y_scores = y_centered.dot(&y_weights);

        Ok(PLSSVD {
            n_components: self.n_components,
            scale: self.scale,
            copy: self.copy,
            x_weights_: Some(x_weights),
            y_weights_: Some(y_weights),
            x_scores_: Some(x_scores),
            y_scores_: Some(y_scores),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            _state: PhantomData,
        })
    }
}

impl PLSSVD<Untrained> {
    /// Simple power iteration for SVD (simplified implementation)
    fn svd_power_iteration(
        &self,
        matrix: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array2<Float>, Array2<Float>, Array1<Float>)> {
        let (m, n) = matrix.dim();
        let k = n_components.min(m).min(n);

        let mut u = Array2::zeros((m, k));
        let mut v = Array2::zeros((n, k));
        let mut s = Array1::zeros(k);

        let mut mat = matrix.clone();

        for i in 0..k {
            // Power iteration to find dominant singular vector
            let mut v_i = Array1::ones(n) / (n as Float).sqrt();

            for _iter in 0..100 {
                let u_temp = mat.dot(&v_i);
                let u_norm = u_temp.dot(&u_temp).sqrt();
                let u_i = if u_norm > 0.0 {
                    u_temp / u_norm
                } else {
                    u_temp
                };

                let v_temp = mat.t().dot(&u_i);
                let v_norm = v_temp.dot(&v_temp).sqrt();
                let v_new = if v_norm > 0.0 {
                    v_temp / v_norm
                } else {
                    v_temp
                };

                // Check convergence
                let diff = (&v_new - &v_i).mapv(|x| x.abs()).sum();
                v_i = v_new;

                if diff < 1e-10 {
                    break;
                }
            }

            // Compute corresponding u vector
            let u_i = mat.dot(&v_i);
            let sigma_i = u_i.dot(&u_i).sqrt();
            let u_i = if sigma_i > 0.0 { u_i / sigma_i } else { u_i };

            // Store results
            u.column_mut(i).assign(&u_i);
            v.column_mut(i).assign(&v_i);
            s[i] = sigma_i;

            // Deflate matrix for next component
            let u_mat = u_i.view().insert_axis(Axis(1));
            let v_mat = v_i.view().insert_axis(Axis(1));
            mat = mat - sigma_i * u_mat.dot(&v_mat.t());
        }

        Ok((u, v, s))
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PLSSVD<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let x_weights = self.x_weights_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform using X weights
        Ok(x_scaled.dot(x_weights))
    }
}

impl PLSSVD<Trained> {
    /// Transform Y using Y weights
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();
        let y_weights = self.y_weights_.as_ref().unwrap();

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform using Y weights
        Ok(y_scaled.dot(y_weights))
    }

    /// Get the X weights (left singular vectors)
    pub fn x_weights(&self) -> &Array2<Float> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights (right singular vectors)
    pub fn y_weights(&self) -> &Array2<Float> {
        self.y_weights_.as_ref().unwrap()
    }

    /// Get the X scores
    pub fn x_scores(&self) -> &Array2<Float> {
        self.x_scores_.as_ref().unwrap()
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> &Array2<Float> {
        self.y_scores_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pls_svd_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        let pls = PLSSVD::new(1);
        let fitted = pls.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
        assert_eq!(y_transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_pls_svd_multiple_components() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0],];

        let pls = PLSSVD::new(2);
        let fitted = pls.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();
        let y_transformed = fitted.transform_y(&y).unwrap();

        assert_eq!(x_transformed.shape(), &[5, 2]);
        assert_eq!(y_transformed.shape(), &[5, 2]);
    }

    #[test]
    fn test_pls_svd_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0], [4.0], [6.0], [8.0],];

        let pls = PLSSVD::new(1).scale(false);
        let fitted = pls.fit(&x, &y).unwrap();

        let x_transformed = fitted.transform(&x).unwrap();

        assert_eq!(x_transformed.shape(), &[4, 1]);
    }
}
