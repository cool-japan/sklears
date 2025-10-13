//! PLS in canonical mode

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Partial Least Squares in canonical mode
///
/// PLSCanonical implements the canonical mode of PLS which is similar to CCA
/// but uses deflation to find orthogonal latent variables.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `scale` - Whether to scale the data
/// * `algorithm` - Algorithm to use (default: 'nipals')
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::PLSCanonical;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]];
///
/// let pls = PLSCanonical::new(1);
/// let fitted = pls.fit(&X, &Y).unwrap();
/// let X_c = fitted.transform(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PLSCanonical<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// Algorithm to use
    pub algorithm: String,
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

    _state: PhantomData<State>,
}

impl PLSCanonical<Untrained> {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            scale: true,
            algorithm: "nipals".to_string(),
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
            _state: PhantomData,
        }
    }

    /// Set whether to scale the data
    pub fn scale(mut self, scale: bool) -> Self {
        self.scale = scale;
        self
    }

    /// Set the algorithm
    pub fn algorithm(mut self, algorithm: String) -> Self {
        self.algorithm = algorithm;
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

impl Estimator for PLSCanonical<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for PLSCanonical<Untrained> {
    type Fitted = PLSCanonical<Trained>;

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

        // NIPALS algorithm for canonical PLS
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
            // Initialize with dominant singular vector (simplified)
            let mut u = y_k.column(0).to_owned();

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features_x);

            while iter < self.max_iter {
                // 1. Update X weights
                let w = x_k.t().dot(&u);
                let w_norm = w.dot(&w).sqrt();
                let w = if w_norm > 0.0 { w / w_norm } else { w };

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Update X scores
                let t = x_k.dot(&w);

                // 3. Update Y weights
                let c = y_k.t().dot(&t);
                let c_norm = c.dot(&c).sqrt();
                let c = if c_norm > 0.0 { c / c_norm } else { c };

                // 4. Update Y scores
                u = y_k.dot(&c);

                // Store weights
                x_weights.column_mut(k).assign(&w);
                y_weights.column_mut(k).assign(&c);

                iter += 1;
            }

            n_iter.push(iter);

            // Calculate final scores
            let t = x_k.dot(&x_weights.column(k));
            x_scores.column_mut(k).assign(&t);
            y_scores.column_mut(k).assign(&u);

            // Calculate loadings
            let p = x_k.t().dot(&t) / (t.dot(&t) + 1e-10);
            let q = y_k.t().dot(&u) / (u.dot(&u) + 1e-10);
            x_loadings.column_mut(k).assign(&p);
            y_loadings.column_mut(k).assign(&q);

            // Deflate X and Y (canonical mode: deflate both with respect to their own scores)
            let t_matrix = t.view().insert_axis(Axis(1));
            let u_matrix = u.view().insert_axis(Axis(1));
            let p_matrix = p.view().insert_axis(Axis(1));
            let q_matrix = q.view().insert_axis(Axis(1));

            x_k = x_k - t_matrix.dot(&p_matrix.t());
            y_k = y_k - u_matrix.dot(&q_matrix.t());
        }

        // Calculate rotations (for canonical mode, these are orthogonal)
        let mut x_rotations = Array2::zeros((n_features_x, self.n_components));
        let mut y_rotations = Array2::zeros((n_features_y, self.n_components));

        // Use Gram-Schmidt orthogonalization for rotations
        for k in 0..self.n_components {
            let mut w_k = x_weights.column(k).to_owned();
            let mut c_k = y_weights.column(k).to_owned();

            // Orthogonalize against previous components
            for j in 0..k {
                let proj_w = x_rotations.column(j).dot(&w_k);
                w_k = w_k - proj_w * &x_rotations.column(j);

                let proj_c = y_rotations.column(j).dot(&c_k);
                c_k = c_k - proj_c * &y_rotations.column(j);
            }

            // Normalize
            let w_norm = w_k.dot(&w_k).sqrt();
            let c_norm = c_k.dot(&c_k).sqrt();

            if w_norm > 0.0 {
                w_k = w_k / w_norm;
            }
            if c_norm > 0.0 {
                c_k = c_k / c_norm;
            }

            x_rotations.column_mut(k).assign(&w_k);
            y_rotations.column_mut(k).assign(&c_k);
        }

        Ok(PLSCanonical {
            n_components: self.n_components,
            scale: self.scale,
            algorithm: self.algorithm,
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
            _state: PhantomData,
        })
    }
}

impl Transform<Array2<Float>, Array2<Float>> for PLSCanonical<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let x_rotations = self.x_rotations_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical space
        Ok(x_scaled.dot(x_rotations))
    }
}

impl PLSCanonical<Trained> {
    /// Transform Y to canonical space
    pub fn transform_y(&self, y: &Array2<Float>) -> Result<Array2<Float>> {
        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();
        let y_rotations = self.y_rotations_.as_ref().unwrap();

        // Center and scale Y
        let mut y_scaled = y - &y_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Transform to canonical space
        Ok(y_scaled.dot(y_rotations))
    }

    /// Get the X weights
    pub fn x_weights(&self) -> &Array2<Float> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> &Array2<Float> {
        self.y_weights_.as_ref().unwrap()
    }

    /// Get the X loadings
    pub fn x_loadings(&self) -> &Array2<Float> {
        self.x_loadings_.as_ref().unwrap()
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> &Array2<Float> {
        self.y_loadings_.as_ref().unwrap()
    }

    /// Get the X scores
    pub fn x_scores(&self) -> &Array2<Float> {
        self.x_scores_.as_ref().unwrap()
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> &Array2<Float> {
        self.y_scores_.as_ref().unwrap()
    }

    /// Get the X rotations (orthogonal transformations)
    pub fn x_rotations(&self) -> &Array2<Float> {
        self.x_rotations_.as_ref().unwrap()
    }

    /// Get the Y rotations (orthogonal transformations)
    pub fn y_rotations(&self) -> &Array2<Float> {
        self.y_rotations_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pls_canonical_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0],];

        let pls = PLSCanonical::new(1);
        let fitted = pls.fit(&x, &y).unwrap();

        let x_canonical = fitted.transform(&x).unwrap();
        let y_canonical = fitted.transform_y(&y).unwrap();

        assert_eq!(x_canonical.shape(), &[4, 1]);
        assert_eq!(y_canonical.shape(), &[4, 1]);
    }

    #[test]
    fn test_pls_canonical_orthogonality() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
            [5.0, 6.0, 7.0],
        ];

        let y = array![[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0], [10.0, 9.0],];

        let pls = PLSCanonical::new(2);
        let fitted = pls.fit(&x, &y).unwrap();

        let x_rotations = fitted.x_rotations();

        // Check orthogonality of rotation vectors
        let dot_product = x_rotations.column(0).dot(&x_rotations.column(1));
        assert!(
            dot_product.abs() < 1e-10,
            "Rotation vectors should be orthogonal"
        );
    }
}
