//! Sparse Partial Least Squares

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Sparse Partial Least Squares Regression
///
/// Sparse PLS adds L1 penalties to PLS to encourage sparsity in the weight vectors,
/// making the model more interpretable by selecting only relevant features.
/// This is particularly useful in high-dimensional settings where feature selection
/// is important.
///
/// # Parameters
///
/// * `n_components` - Number of components to keep
/// * `l1_param_x` - L1 regularization parameter for X weights (sparsity level)
/// * `l1_param_y` - L1 regularization parameter for Y weights (sparsity level)
/// * `scale` - Whether to scale the data
/// * `max_iter` - Maximum number of iterations for the algorithm
/// * `tol` - Tolerance for convergence
/// * `copy` - Whether to copy X and Y during fit
///
/// # Examples
///
/// ```
/// use sklears_cross_decomposition::SparsePLS;
/// use sklears_core::traits::{Predict, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 0.0], [3.0, 4.0, 0.0], [5.0, 6.0, 0.0], [7.0, 8.0, 0.0]];
/// let Y = array![[1.5], [2.5], [3.5], [4.5]];
///
/// let spls = SparsePLS::new(1, 0.1, 0.1);
/// let fitted = spls.fit(&X, &Y).unwrap();
/// let predictions = fitted.predict(&X).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SparsePLS<State = Untrained> {
    /// Number of components to keep
    pub n_components: usize,
    /// L1 regularization parameter for X weights
    pub l1_param_x: Float,
    /// L1 regularization parameter for Y weights
    pub l1_param_y: Float,
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
    coef_: Option<Array2<Float>>,
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl SparsePLS<Untrained> {
    /// Create a new Sparse PLS model
    pub fn new(n_components: usize, l1_param_x: Float, l1_param_y: Float) -> Self {
        Self {
            n_components,
            l1_param_x,
            l1_param_y,
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

impl Estimator for SparsePLS<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for SparsePLS<Untrained> {
    type Fitted = SparsePLS<Trained>;

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

        // Sparse PLS algorithm using iterative soft thresholding
        let mut x_weights = Array2::zeros((n_features, self.n_components));
        let mut y_weights = Array2::zeros((n_targets, self.n_components));
        let mut x_loadings = Array2::zeros((n_features, self.n_components));
        let mut y_loadings = Array2::zeros((n_targets, self.n_components));
        let mut x_scores = Array2::zeros((n_samples, self.n_components));
        let mut y_scores = Array2::zeros((n_samples, self.n_components));
        let mut n_iter = Vec::new();

        let mut x_k = x_centered.clone();
        let mut y_k = y_centered.clone();

        for k in 0..self.n_components {
            // Initialize y_score as first column of Y_k
            let mut y_score = if y_k.ncols() > 0 {
                y_k.column(0).to_owned()
            } else {
                Array1::ones(n_samples)
            };

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features);

            while iter < self.max_iter {
                // 1. Update X weights with L1 penalty (soft thresholding)
                let gradient_w = x_k.t().dot(&y_score);
                let w = self.soft_threshold(&gradient_w, self.l1_param_x);

                // Normalize w (with sparsity-aware normalization)
                let w_norm = w.dot(&w).sqrt();
                let w = if w_norm > 0.0 { w / w_norm } else { w };

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
                let c = self.soft_threshold(&gradient_c, self.l1_param_y);

                // Normalize c (with sparsity-aware normalization)
                let c_norm = c.dot(&c).sqrt();
                let c = if c_norm > 0.0 { c / c_norm } else { c };

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

            // Calculate loadings
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

        // Calculate rotations and regression coefficients
        let x_rotations = x_weights.clone();
        let coef = x_weights.dot(&y_loadings.t());

        Ok(SparsePLS {
            n_components: self.n_components,
            l1_param_x: self.l1_param_x,
            l1_param_y: self.l1_param_y,
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

impl SparsePLS<Untrained> {
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

impl Predict<Array2<Float>, Array2<Float>> for SparsePLS<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array2<Float>> {
        let x_mean = self.x_mean_.as_ref().unwrap();
        let y_mean = self.y_mean_.as_ref().unwrap();
        let x_std = self.x_std_.as_ref().unwrap();
        let y_std = self.y_std_.as_ref().unwrap();
        let coef = self.coef_.as_ref().unwrap();

        // Center and scale X
        let mut x_scaled = x - &x_mean.view().insert_axis(Axis(0));

        if self.scale {
            for (i, &std) in x_std.iter().enumerate() {
                if std > 0.0 {
                    x_scaled.column_mut(i).mapv_inplace(|v| v / std);
                }
            }
        }

        // Predict
        let mut y_pred = x_scaled.dot(coef);

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

impl Transform<Array2<Float>> for SparsePLS<Trained> {
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

        // Transform to PLS space
        Ok(x_scaled.dot(x_rotations))
    }
}

impl SparsePLS<Trained> {
    /// Get the X weights (with sparsity)
    pub fn x_weights(&self) -> &Array2<Float> {
        self.x_weights_.as_ref().unwrap()
    }

    /// Get the Y weights (with sparsity)
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

    /// Get the regression coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
    }

    /// Count the number of non-zero weights in X
    pub fn x_sparsity(&self) -> usize {
        let x_weights = self.x_weights_.as_ref().unwrap();
        x_weights.iter().filter(|&&x| x.abs() > 1e-10).count()
    }

    /// Count the number of non-zero weights in Y
    pub fn y_sparsity(&self) -> usize {
        let y_weights = self.y_weights_.as_ref().unwrap();
        y_weights.iter().filter(|&&x| x.abs() > 1e-10).count()
    }

    /// Get the sparsity ratio for X weights (proportion of zero weights)
    pub fn x_sparsity_ratio(&self) -> Float {
        let x_weights = self.x_weights_.as_ref().unwrap();
        let total_weights = x_weights.len();
        let non_zero_weights = self.x_sparsity();
        1.0 - (non_zero_weights as Float / total_weights as Float)
    }

    /// Get the sparsity ratio for Y weights (proportion of zero weights)
    pub fn y_sparsity_ratio(&self) -> Float {
        let y_weights = self.y_weights_.as_ref().unwrap();
        let total_weights = y_weights.len();
        let non_zero_weights = self.y_sparsity();
        1.0 - (non_zero_weights as Float / total_weights as Float)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_sparse_pls_basic() {
        let x = array![
            [1.0, 2.0, 0.0],
            [2.0, 3.0, 0.0],
            [3.0, 4.0, 0.0],
            [4.0, 5.0, 0.0],
        ];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let spls = SparsePLS::new(1, 0.1, 0.1);
        let fitted = spls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }

    #[test]
    fn test_sparse_pls_sparsity() {
        let x = array![
            [1.0, 2.0, 0.0, 0.0], // Last two columns are irrelevant
            [2.0, 3.0, 0.1, 0.1],
            [3.0, 4.0, 0.0, 0.0],
            [4.0, 5.0, 0.1, 0.1],
        ];

        let y = array![[1.5], [2.5], [3.5], [4.5]]; // Only correlated with first two columns

        // High regularization should produce sparse weights
        let spls_high = SparsePLS::new(1, 0.5, 0.1);
        let fitted_high = spls_high.fit(&x, &y).unwrap();

        // Low regularization should produce less sparse weights
        let spls_low = SparsePLS::new(1, 0.01, 0.01);
        let fitted_low = spls_low.fit(&x, &y).unwrap();

        // Check sparsity
        let sparsity_high_x = fitted_high.x_sparsity();
        let sparsity_low_x = fitted_low.x_sparsity();

        // High regularization should produce more sparsity (fewer non-zero weights)
        assert!(sparsity_high_x <= sparsity_low_x);

        // Check sparsity ratios
        let sparsity_ratio_high = fitted_high.x_sparsity_ratio();
        let sparsity_ratio_low = fitted_low.x_sparsity_ratio();

        assert!(sparsity_ratio_high >= sparsity_ratio_low);
        assert!(sparsity_ratio_high >= 0.0 && sparsity_ratio_high <= 1.0);
        assert!(sparsity_ratio_low >= 0.0 && sparsity_ratio_low <= 1.0);
    }

    #[test]
    fn test_sparse_pls_regularization_effect() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        // Test different regularization levels
        let spls_no_reg = SparsePLS::new(1, 0.0, 0.0);
        let fitted_no_reg = spls_no_reg.fit(&x, &y).unwrap();

        let spls_reg = SparsePLS::new(1, 0.3, 0.3);
        let fitted_reg = spls_reg.fit(&x, &y).unwrap();

        // Regularized version should have more sparsity
        let sparsity_no_reg = fitted_no_reg.x_sparsity();
        let sparsity_reg = fitted_reg.x_sparsity();

        assert!(sparsity_reg <= sparsity_no_reg);

        // Both should produce valid predictions
        let pred_no_reg = fitted_no_reg.predict(&x).unwrap();
        let pred_reg = fitted_reg.predict(&x).unwrap();

        assert_eq!(pred_no_reg.shape(), &[4, 1]);
        assert_eq!(pred_reg.shape(), &[4, 1]);
    }

    #[test]
    fn test_sparse_pls_multiple_components() {
        let x = array![
            [1.0, 2.0, 3.0, 0.0],
            [2.0, 3.0, 4.0, 0.0],
            [3.0, 4.0, 5.0, 0.0],
            [4.0, 5.0, 6.0, 0.0],
            [5.0, 6.0, 7.0, 0.0],
        ];

        let y = array![[1.5, 0.5], [2.5, 1.5], [3.5, 2.5], [4.5, 3.5], [5.5, 4.5]];

        let spls = SparsePLS::new(2, 0.1, 0.1);
        let fitted = spls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(predictions.shape(), &[5, 2]);
        assert_eq!(transformed.shape(), &[5, 2]);

        // Check that weights are sparse
        let x_sparsity = fitted.x_sparsity();
        let total_weights = fitted.x_weights().len();
        assert!(x_sparsity <= total_weights);
    }

    #[test]
    fn test_sparse_pls_transform() {
        let x = array![
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
            [3.0, 4.0, 5.0],
            [4.0, 5.0, 6.0],
        ];

        let y = array![[1.5, 0.5], [2.5, 1.5], [3.5, 2.5], [4.5, 3.5]];

        let spls = SparsePLS::new(2, 0.1, 0.1);
        let fitted = spls.fit(&x, &y).unwrap();
        let transformed = fitted.transform(&x).unwrap();

        assert_eq!(transformed.shape(), &[4, 2]);

        // Transform should be consistent
        let transformed2 = fitted.transform(&x).unwrap();
        for (a, b) in transformed.iter().zip(transformed2.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    #[test]
    fn test_sparse_pls_no_scaling() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.5], [2.5], [3.5], [4.5]];

        let spls = SparsePLS::new(1, 0.1, 0.1).scale(false);
        let fitted = spls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }

    // Property-based tests
    proptest! {
        #[test]
        fn test_sparse_pls_prediction_consistency(
            n_samples in 5..15usize,
            n_features in 2..6usize,
            n_targets in 1..3usize,
            l1_param in 0.01..0.3f64,
            data in prop::collection::vec(-3.0..3.0f64, 0..150)
        ) {
            if data.len() < n_samples * (n_features + n_targets) {
                return Ok(());
            }

            let n_components = 1.min(n_features).min(n_targets);

            let x_data = &data[0..n_samples * n_features];
            let y_data = &data[n_samples * n_features..n_samples * (n_features + n_targets)];

            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;
            let y = Array2::from_shape_vec((n_samples, n_targets), y_data.to_vec())?;

            if let Ok(fitted) = SparsePLS::new(n_components, l1_param, l1_param).fit(&x, &y) {
                // Test prediction consistency
                let predictions1 = fitted.predict(&x)?;
                let predictions2 = fitted.predict(&x)?;

                prop_assert_eq!(predictions1.shape(), predictions2.shape());
                for (a, b) in predictions1.iter().zip(predictions2.iter()) {
                    prop_assert!((a - b).abs() < 1e-12);
                }

                // Test sparsity properties
                let x_sparsity = fitted.x_sparsity();
                let total_x_weights = fitted.x_weights().len();
                prop_assert!(x_sparsity <= total_x_weights);

                let sparsity_ratio = fitted.x_sparsity_ratio();
                prop_assert!(sparsity_ratio >= 0.0);
                prop_assert!(sparsity_ratio <= 1.0);

                // Check that outputs are finite
                for val in predictions1.iter() {
                    prop_assert!(val.is_finite());
                }

                // Check that weights are finite
                let x_weights = fitted.x_weights();
                for val in x_weights.iter() {
                    prop_assert!(val.is_finite());
                }
            }
        }

        #[test]
        fn test_sparse_pls_sparsity_increases_with_regularization(
            l1_low in 0.01..0.1f64,
            l1_high in 0.2..0.5f64
        ) {
            let x = array![
                [1.0, 2.0, 3.0, 0.0],
                [2.0, 3.0, 4.0, 0.1],
                [3.0, 4.0, 5.0, 0.0],
                [4.0, 5.0, 6.0, 0.1],
                [5.0, 6.0, 7.0, 0.0],
            ];
            let y = array![[1.5], [2.5], [3.5], [4.5], [5.5]];

            let spls_low = SparsePLS::new(1, l1_low, l1_low);
            let spls_high = SparsePLS::new(1, l1_high, l1_high);

            if let (Ok(fitted_low), Ok(fitted_high)) = (spls_low.fit(&x, &y), spls_high.fit(&x, &y)) {
                let sparsity_low = fitted_low.x_sparsity();
                let sparsity_high = fitted_high.x_sparsity();
                let ratio_low = fitted_low.x_sparsity_ratio();
                let ratio_high = fitted_high.x_sparsity_ratio();

                // Higher regularization should lead to more sparsity
                prop_assert!(sparsity_high <= sparsity_low);
                prop_assert!(ratio_high >= ratio_low);

                // Both should be valid
                prop_assert!(ratio_low >= 0.0 && ratio_low <= 1.0);
                prop_assert!(ratio_high >= 0.0 && ratio_high <= 1.0);
            }
        }
    }
}
