//! Orthogonal Partial Least Squares (OPLS)

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Orthogonal Partial Least Squares (OPLS)
///
/// OPLS separates the systematic variation in X into two parts: variation correlated
/// with Y (predictive components) and variation orthogonal to Y (orthogonal components).
/// This decomposition improves model interpretability by isolating Y-related variation.
///
/// # Parameters
///
/// * `n_components` - Number of predictive components to keep
/// * `n_orthogonal` - Number of orthogonal components to remove
/// * `scale` - Whether to scale the data
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::OPLS;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
/// let Y = array![[1.5], [2.5], [3.5], [4.5]];
///
/// let opls = OPLS::new(1, 1);
/// let fitted = opls.fit(&X, &Y).unwrap();
/// let predictions = fitted.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct OPLS<State = Untrained> {
    /// Number of predictive components to keep
    pub n_components: usize,
    /// Number of orthogonal components to remove
    pub n_orthogonal: usize,
    /// Whether to scale X and Y
    pub scale: bool,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Whether to copy X and Y
    pub copy: bool,

    // Fitted attributes - Predictive components
    x_weights_: Option<Array2<Float>>,
    y_weights_: Option<Array2<Float>>,
    x_loadings_: Option<Array2<Float>>,
    y_loadings_: Option<Array2<Float>>,
    x_scores_: Option<Array2<Float>>,
    y_scores_: Option<Array2<Float>>,

    // Orthogonal components
    x_weights_orthogonal_: Option<Array2<Float>>,
    x_loadings_orthogonal_: Option<Array2<Float>>,
    x_scores_orthogonal_: Option<Array2<Float>>,

    // Model parameters
    coef_: Option<Array2<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl OPLS<Untrained> {
    /// Create a new OPLS model
    pub fn new(n_components: usize, n_orthogonal: usize) -> Self {
        Self {
            n_components,
            n_orthogonal,
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
            x_weights_orthogonal_: None,
            x_loadings_orthogonal_: None,
            x_scores_orthogonal_: None,
            coef_: None,
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

impl Estimator for OPLS<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for OPLS<Untrained> {
    type Fitted = OPLS<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array2<Float>) -> Result<Self::Fitted> {
        let (n_samples, n_features) = x.dim();
        let (_, n_targets) = y.dim();

        if n_samples != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "X and Y must have the same number of samples".to_string(),
            ));
        }

        if self.n_components > n_features.min(n_targets) {
            return Err(SklearsError::InvalidInput(
                "n_components cannot exceed min(n_features, n_targets)".to_string(),
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
            (Array1::ones(n_features), Array1::ones(n_targets))
        };

        // OPLS algorithm
        let mut x_weights = Array2::zeros((n_features, self.n_components));
        let mut y_weights = Array2::zeros((n_targets, self.n_components));
        let mut x_loadings = Array2::zeros((n_features, self.n_components));
        let mut y_loadings = Array2::zeros((n_targets, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));

        let mut x_weights_orthogonal = Array2::zeros((n_features, self.n_orthogonal));
        let mut x_loadings_orthogonal = Array2::zeros((n_features, self.n_orthogonal));
        let mut x_scores_orthogonal = Array2::zeros((n_samples, self.n_orthogonal));

        let mut n_iter = Vec::new();

        let mut x_current = x_centered.clone();
        let y_work = y_centered.clone();

        // Step 1: Remove orthogonal components from X
        for orth_comp in 0..self.n_orthogonal {
            // Find the first PLS component between X and Y
            let (w_temp, _c_temp, iter) = self.pls_component(&x_current, &y_work)?;

            // Calculate orthogonal weight as the difference between
            // the dominant direction in X and the Y-related direction
            let x_cov = x_current.t().dot(&x_current) / (n_samples as Float - 1.0);
            let (dominant_direction, _) = self.power_iteration(&x_cov, 1)?;

            // Make orthogonal to Y-related direction
            let w_orth =
                &dominant_direction.column(0) - w_temp.dot(&dominant_direction.column(0)) * &w_temp;
            let w_orth_norm = w_orth.dot(&w_orth).sqrt();
            let w_orth = if w_orth_norm > 0.0 {
                w_orth / w_orth_norm
            } else {
                w_orth
            };

            // Calculate orthogonal scores and loadings
            let t_orth = x_current.dot(&w_orth);
            let p_orth = x_current.t().dot(&t_orth) / (t_orth.dot(&t_orth) + 1e-10);

            // Store orthogonal components
            x_weights_orthogonal.column_mut(orth_comp).assign(&w_orth);
            x_scores_orthogonal.column_mut(orth_comp).assign(&t_orth);
            x_loadings_orthogonal.column_mut(orth_comp).assign(&p_orth);

            // Deflate X by removing orthogonal component
            let t_orth_matrix = t_orth.view().insert_axis(Axis(1));
            let p_orth_matrix = p_orth.view().insert_axis(Axis(1));
            x_current = x_current - t_orth_matrix.dot(&p_orth_matrix.t());

            n_iter.push(iter);
        }

        // Step 2: Build predictive PLS model on orthogonally corrected X
        let mut x_k = x_current;
        let mut y_k = y_work.clone();

        for k in 0..self.n_components {
            let (w, c, iter) = self.pls_component(&x_k, &y_k)?;

            // Calculate scores
            let t = x_k.dot(&w);
            let u = y_k.dot(&c);

            // Calculate loadings
            let p = x_k.t().dot(&t) / (t.dot(&t) + 1e-10);
            let q = y_k.t().dot(&t) / (t.dot(&t) + 1e-10);

            // Store components
            x_weights.column_mut(k).assign(&w);
            y_weights.column_mut(k).assign(&c);
            x_scores.column_mut(k).assign(&t);
            y_scores.column_mut(k).assign(&u);
            x_loadings.column_mut(k).assign(&p);
            y_loadings.column_mut(k).assign(&q);

            // Deflate X and Y
            let t_matrix = t.view().insert_axis(Axis(1));
            let p_matrix = p.view().insert_axis(Axis(1));
            let q_matrix = q.view().insert_axis(Axis(1));

            x_k = x_k - t_matrix.dot(&p_matrix.t());
            y_k = y_k - t_matrix.dot(&q_matrix.t());

            n_iter.push(iter);
        }

        // Calculate regression coefficients
        let coef = x_weights.dot(&y_loadings.t());

        Ok(OPLS {
            n_components: self.n_components,
            n_orthogonal: self.n_orthogonal,
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
            x_weights_orthogonal_: Some(x_weights_orthogonal),
            x_loadings_orthogonal_: Some(x_loadings_orthogonal),
            x_scores_orthogonal_: Some(x_scores_orthogonal),
            coef_: Some(coef),
            n_iter_: Some(n_iter),
            x_mean_: Some(x_mean),
            y_mean_: Some(y_mean),
            x_std_: Some(x_std),
            y_std_: Some(y_std),
            _state: PhantomData,
        })
    }
}

impl OPLS<Untrained> {
    /// Extract a single PLS component
    fn pls_component(
        &self,
        x: &Array2<Float>,
        y: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array1<Float>, usize)> {
        let (n_samples, n_features_x) = x.dim();
        let (_, n_features_y) = y.dim();

        // Initialize with first column of Y
        let mut u = if n_features_y > 0 {
            y.column(0).to_owned()
        } else {
            Array1::ones(n_samples)
        };

        let mut w_old = Array1::zeros(n_features_x);
        let mut iter = 0;

        while iter < self.max_iter {
            // Update X weights
            let w = x.t().dot(&u);
            let w_norm = w.dot(&w).sqrt();
            let w = if w_norm > 0.0 { w / w_norm } else { w };

            // Check convergence
            let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
            if diff < self.tol {
                break;
            }
            w_old = w.clone();

            // Update X scores
            let t = x.dot(&w);

            // Update Y weights
            let c = y.t().dot(&t);
            let c_norm = c.dot(&c).sqrt();
            let c = if c_norm > 0.0 { c / c_norm } else { c };

            // Update Y scores
            u = y.dot(&c);

            iter += 1;
        }

        let w = x.t().dot(&u);
        let w_norm = w.dot(&w).sqrt();
        let w = if w_norm > 0.0 { w / w_norm } else { w };

        let t = x.dot(&w);
        let c = y.t().dot(&t);
        let c_norm = c.dot(&c).sqrt();
        let c = if c_norm > 0.0 { c / c_norm } else { c };

        Ok((w, c, iter))
    }

    /// Simple power iteration for dominant eigenvector
    fn power_iteration(
        &self,
        matrix: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array2<Float>, Array1<Float>)> {
        let n = matrix.nrows();
        let k = n_components.min(n);

        let mut vectors = Array2::zeros((n, k));
        let mut values = Array1::zeros(k);
        let mut mat = matrix.clone();

        for i in 0..k {
            let mut v = Array1::ones(n) / (n as Float).sqrt();

            for _iter in 0..100 {
                let v_new = mat.dot(&v);
                let norm = v_new.dot(&v_new).sqrt();
                let v_new = if norm > 0.0 { v_new / norm } else { v_new };

                let diff = (&v_new - &v).mapv(|x| x.abs()).sum();
                v = v_new;

                if diff < 1e-10 {
                    break;
                }
            }

            let eigenvalue = v.dot(&mat.dot(&v));
            vectors.column_mut(i).assign(&v);
            values[i] = eigenvalue;

            // Deflate matrix
            let v_matrix = v.view().insert_axis(Axis(1));
            mat = mat - eigenvalue * v_matrix.dot(&v_matrix.t());
        }

        Ok((vectors, values))
    }
}

impl Predict<Array2<Float>, Array2<Float>> for OPLS<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_transformed = self.transform_orthogonal_corrected(x)?;
        let coef = self.coef_.as_ref().unwrap();
        let y_mean = self.y_mean_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();

        // Predict using corrected X
        let mut y_pred = x_transformed.dot(coef);

        // Unscale and uncenter Y
        if self.scale {
            for (i, &std) in y_std.iter().enumerate() {
                if std > 0.0 {
                    y_pred.column_mut(i).mapv_inplace(|v| v * std);
                }
            }
        }

        y_pred += &y_mean.view().insert_axis(Axis(0));

        Ok(y_pred)
    }
}

impl Transform<Array2<Float>> for OPLS<Trained> {
    fn transform(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_corrected = self.transform_orthogonal_corrected(x)?;
        let x_weights = self.x_weights_.as_ref().unwrap();

        // Transform to PLS predictive space
        Ok(x_corrected.dot(x_weights))
    }
}

impl OPLS<Trained> {
    /// Transform X by removing orthogonal components
    pub fn transform_orthogonal_corrected(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let x_weights_orth = self.x_weights_orthogonal_.as_ref().unwrap();
        let x_loadings_orth = self.x_loadings_orthogonal_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Remove orthogonal components
        let mut x_corrected = x_scaled;
        for orth_comp in 0..self.n_orthogonal {
            let w_orth = x_weights_orth.column(orth_comp);
            let p_orth = x_loadings_orth.column(orth_comp);

            let t_orth = x_corrected.dot(&w_orth);
            let t_orth_matrix = t_orth.view().insert_axis(Axis(1));
            let p_orth_matrix = p_orth.view().insert_axis(Axis(1));

            x_corrected = x_corrected - t_orth_matrix.dot(&p_orth_matrix.t());
        }

        Ok(x_corrected)
    }

    /// Get the predictive X weights
    pub fn x_weights(&self) -> &Array2<Float> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights
    pub fn y_weights(&self) -> &Array2<Float> {
        self.y_weights_.as_ref().unwrap()
    }

    /// Get the predictive X loadings
    pub fn x_loadings(&self) -> &Array2<Float> {
        self.x_loadings_.as_ref().unwrap()
    }

    /// Get the Y loadings
    pub fn y_loadings(&self) -> &Array2<Float> {
        self.y_loadings_.as_ref().unwrap()
    }

    /// Get the predictive X scores
    pub fn x_scores(&self) -> &Array2<Float> {
        self.x_scores_.as_ref().unwrap()
    }

    /// Get the Y scores
    pub fn y_scores(&self) -> &Array2<Float> {
        self.y_scores_.as_ref().unwrap()
    }

    /// Get the orthogonal X weights
    pub fn x_weights_orthogonal(&self) -> &Array2<Float> {
        self.x_weights_orthogonal_.as_ref().unwrap()
    }

    /// Get the orthogonal X loadings
    pub fn x_loadings_orthogonal(&self) -> &Array2<Float> {
        self.x_loadings_orthogonal_.as_ref().unwrap()
    }

    /// Get the orthogonal X scores
    pub fn x_scores_orthogonal(&self) -> &Array2<Float> {
        self.x_scores_orthogonal_.as_ref().unwrap()
    }

    /// Get the regression coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_opls_basic() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let opls = OPLS::new(1, 1);
        let fitted = opls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }

    #[test]
    fn test_opls_orthogonal_correction() {
        let x = array![
            [1.0, 10.0], // Second column has large variance but uncorrelated with Y
            [2.0, 5.0],
            [3.0, 15.0],
            [4.0, 2.0],
        ];

        let y = array![[1.0], [2.0], [3.0], [4.0]]; // Only correlated with first column

        let opls = OPLS::new(1, 1);
        let fitted = opls.fit(&x, &y).unwrap();

        // Test that we can transform to orthogonally corrected space
        let x_corrected = fitted.transform_orthogonal_corrected(&x).unwrap();
        assert_eq!(x_corrected.shape(), &[4, 2]);

        // Test predictions
        let predictions = fitted.predict(&x).unwrap();
        assert_eq!(predictions.shape(), &[4, 1]);
    }

    #[test]
    fn test_opls_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let opls = OPLS::new(1, 1);
        let fitted = opls.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[4, 1]);
    }

    #[test]
    fn test_opls_no_orthogonal() {
        // Test OPLS with no orthogonal components (should behave like regular PLS)
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let opls = OPLS::new(1, 0);
        let fitted = opls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }

    #[test]
    fn test_opls_multiple_components() {
        let x = array![
            [1.0, 2.0, 3.0, 4.0],
            [2.0, 3.0, 4.0, 5.0],
            [3.0, 4.0, 5.0, 6.0],
            [4.0, 5.0, 6.0, 7.0],
            [5.0, 6.0, 7.0, 8.0],
        ];

        let y = array![[1.5, 0.5], [2.5, 1.5], [3.5, 2.5], [4.5, 3.5], [5.5, 4.5]];

        let opls = OPLS::new(2, 1);
        let fitted = opls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(predictions.shape(), &[5, 2]);
        assert_eq!(transformed.shape(), &[5, 2]);
    }

    #[test]
    fn test_opls_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let opls = OPLS::new(1, 1).scale(false);
        let fitted = opls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }
}
