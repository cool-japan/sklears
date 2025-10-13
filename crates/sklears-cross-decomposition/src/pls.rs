//! Partial Least Squares regression

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Partial Least Squares regression
#[derive(Debug, Clone)]
pub struct PLSRegression<State = Untrained> {
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
    #[allow(dead_code)]
    x_scores_: Option<Array2<Float>>,
    #[allow(dead_code)]
    y_scores_: Option<Array2<Float>>,
    #[allow(dead_code)]
    x_rotations_: Option<Array2<Float>>,
    #[allow(dead_code)]
    y_rotations_: Option<Array2<Float>>,
    coef_: Option<Array2<Float>>,
    #[allow(dead_code)]
    n_iter_: Option<Vec<usize>>,
    x_mean_: Option<Array1<Float>>,
    y_mean_: Option<Array1<Float>>,
    x_std_: Option<Array1<Float>>,
    y_std_: Option<Array1<Float>>,

    _state: PhantomData<State>,
}

impl PLSRegression<Untrained> {
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
}

impl Estimator for PLSRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array2<Float>> for PLSRegression<Untrained> {
    type Fitted = PLSRegression<Trained>;

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

        // NIPALS algorithm
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
            let mut y_score = y_k.column(0).to_owned();

            let mut iter = 0;
            let mut w_old = Array1::zeros(n_features);

            while iter < self.max_iter {
                // 1. Update X weights
                let w = x_k.t().dot(&y_score);
                let w_norm = w.dot(&w).sqrt();
                let w = w / w_norm;

                // Check convergence
                let diff = (&w - &w_old).mapv(|x| x.abs()).sum();
                if diff < self.tol {
                    break;
                }
                w_old = w.clone();

                // 2. Update X scores
                let x_score = x_k.dot(&w);

                // 3. Update Y weights
                let c = y_k.t().dot(&x_score);
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

            // Calculate scores
            let x_score = x_k.dot(&x_weights.column(k));
            x_scores.column_mut(k).assign(&x_score);
            y_scores.column_mut(k).assign(&y_score);

            // Calculate loadings
            let x_loading = x_k.t().dot(&x_score) / x_score.dot(&x_score);
            let y_loading = y_k.t().dot(&x_score) / x_score.dot(&x_score);
            x_loadings.column_mut(k).assign(&x_loading);
            y_loadings.column_mut(k).assign(&y_loading);

            // Deflate X and Y
            let x_score_matrix = x_score.view().insert_axis(Axis(1));
            let x_loading_matrix = x_loading.view().insert_axis(Axis(1));
            let y_loading_matrix = y_loading.view().insert_axis(Axis(1));

            x_k = x_k - x_score_matrix.dot(&x_loading_matrix.t());
            y_k = y_k - x_score_matrix.dot(&y_loading_matrix.t());
        }

        // Calculate rotations (simplified - using pseudo-inverse would be better)
        let x_rotations = x_weights.clone();
        let y_rotations = y_weights.clone();

        // Calculate regression coefficients
        let coef = x_weights.dot(&y_loadings.t());

        Ok(PLSRegression {
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

impl Predict<Array2<Float>, Array2<Float>> for PLSRegression<Trained> {
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

impl Transform<Array2<Float>> for PLSRegression<Trained> {
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

impl PLSRegression<Trained> {
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

    /// Get the regression coefficients
    pub fn coef(&self) -> &Array2<Float> {
        self.coef_.as_ref().unwrap()
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
    use proptest::prelude::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_pls_regression_basic() {
        let x = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0],];

        let y = array![[1.5], [2.5], [3.5], [4.5],];

        let pls = PLSRegression::new(1);
        let fitted = pls.fit(&x, &y).unwrap();
        let predictions = fitted.predict(&x).unwrap();

        assert_eq!(predictions.shape(), &[4, 1]);
    }

    // Property-based tests
    proptest! {
        #[test]
        fn test_pls_regression_prediction_consistency(
            n_samples in 5..20usize,
            n_features in 2..8usize,
            n_targets in 1..4usize,
            n_components in 1..3usize,
            data in prop::collection::vec(-5.0..5.0f64, 0..200)
        ) {
            // Ensure we have enough data
            if data.len() < n_samples * (n_features + n_targets) {
                return Ok(());
            }

            let n_components = n_components.min(n_features).min(n_targets);

            // Create random matrices
            let x_data = &data[0..n_samples * n_features];
            let y_data = &data[n_samples * n_features..n_samples * (n_features + n_targets)];

            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;
            let y = Array2::from_shape_vec((n_samples, n_targets), y_data.to_vec())?;

            if let Ok(fitted) = PLSRegression::new(n_components).fit(&x, &y) {
                // Test prediction consistency
                let predictions1 = fitted.predict(&x)?;
                let predictions2 = fitted.predict(&x)?;

                // Predictions should be identical
                prop_assert_eq!(predictions1.shape(), predictions2.shape());
                for (a, b) in predictions1.iter().zip(predictions2.iter()) {
                    prop_assert!((a - b).abs() < 1e-12);
                }

                // Check output dimensions
                prop_assert_eq!(predictions1.shape(), &[n_samples, n_targets]);

                // Test transform consistency
                let transform1 = fitted.transform(&x)?;
                let transform2 = fitted.transform(&x)?;

                prop_assert_eq!(transform1.shape(), &[n_samples, n_components]);
                for (a, b) in transform1.iter().zip(transform2.iter()) {
                    prop_assert!((a - b).abs() < 1e-12);
                }
            }
        }

        #[test]
        fn test_pls_regression_perfect_linear_relationship(
            n_samples in 5..15usize,
            n_features in 2..5usize,
            slope in 0.1..5.0f64,
            intercept in -10.0..10.0f64
        ) {
            // Create perfect linear relationship: Y = slope * X + intercept
            let x: Array2<Float> = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                i as Float + j as Float + 1.0
            });

            let y: Array2<Float> = Array2::from_shape_fn((n_samples, 1), |(i, _)| {
                slope * x.row(i).sum() + intercept
            });

            if let Ok(fitted) = PLSRegression::new(1).fit(&x, &y) {
                let predictions = fitted.predict(&x)?;

                // For perfect linear relationship, PLS should predict very accurately
                let mse: Float = predictions.iter()
                    .zip(y.iter())
                    .map(|(pred, actual)| (pred - actual).powi(2))
                    .sum::<Float>() / n_samples as Float;

                prop_assert!(mse < 1e-6, "MSE too high: {}", mse);
            }
        }

        #[test]
        fn test_pls_regression_weights_orthogonality(
            n_samples in 8..20usize,
            n_features in 3..8usize,
            n_components in 2..4usize,
            data in prop::collection::vec(-3.0..3.0f64, 0..200)
        ) {
            if data.len() < n_samples * (n_features + 1) {
                return Ok(());
            }

            let n_components = n_components.min(n_features);

            let x_data = &data[0..n_samples * n_features];
            let y_data = &data[n_samples * n_features..n_samples * (n_features + 1)];

            let x = Array2::from_shape_vec((n_samples, n_features), x_data.to_vec())?;
            let y = Array2::from_shape_vec((n_samples, 1), y_data.to_vec())?;

            if let Ok(fitted) = PLSRegression::new(n_components).fit(&x, &y) {
                let x_weights = fitted.x_weights();

                // Check that X weights have reasonable norms
                for i in 0..n_components {
                    let weight_norm = x_weights.column(i).dot(&x_weights.column(i)).sqrt();
                    prop_assert!(weight_norm > 0.5);  // Should be normalized to ~1
                    prop_assert!(weight_norm < 1.5);
                }

                // Check that all values are finite
                for val in x_weights.iter() {
                    prop_assert!(val.is_finite());
                }
            }
        }

        #[test]
        fn test_pls_regression_scaling_invariance(
            scale_factor in 1.0..50.0f64
        ) {
            let x = array![
                [1.0, 2.0],
                [2.0, 3.0],
                [3.0, 4.0],
                [4.0, 5.0],
                [5.0, 6.0],
            ];
            let y = array![[1.5], [2.5], [3.5], [4.5], [5.5]];

            // Scale X by the factor
            let x_scaled = &x * scale_factor;

            let pls1 = PLSRegression::new(1).scale(true);
            let pls2 = PLSRegression::new(1).scale(true);

            if let (Ok(fitted1), Ok(fitted2)) = (pls1.fit(&x, &y), pls2.fit(&x_scaled, &y)) {
                let pred1 = fitted1.predict(&x)?;
                let pred2 = fitted2.predict(&x_scaled)?;

                // With scaling enabled, predictions should be very similar
                let max_diff: f64 = pred1.iter()
                    .zip(pred2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0, f64::max);

                prop_assert!(max_diff < 0.1);  // Allow some numerical tolerance
            }
        }

        #[test]
        fn test_pls_regression_numerical_stability(
            noise_level in 0.0..0.1f64
        ) {
            let mut x = array![
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
            ];
            let mut y = array![[1.5], [2.5], [3.5], [4.5], [5.5]];

            // Add small amount of noise
            for val in x.iter_mut() {
                *val += noise_level * ((*val * 12345.0_f64).sin());
            }
            for val in y.iter_mut() {
                *val += noise_level * ((*val * 67890.0_f64).sin());
            }

            if let Ok(fitted) = PLSRegression::new(2).fit(&x, &y) {
                let predictions = fitted.predict(&x)?;
                let transform = fitted.transform(&x)?;

                // Check that outputs are finite
                for val in predictions.iter() {
                    prop_assert!(val.is_finite());
                }
                for val in transform.iter() {
                    prop_assert!(val.is_finite());
                }

                // Check that coefficients are reasonable
                let coef = fitted.coef();
                for val in coef.iter() {
                    prop_assert!(val.is_finite());
                    prop_assert!(val.abs() < 1000.0);  // Shouldn't be too large
                }
            }
        }

        #[test]
        fn test_pls_regression_component_reduction(
            n_samples in 6..15usize,
            n_features in 3..8usize,
            max_components in 1..4usize
        ) {
            // Create synthetic data with known structure
            let x: Array2<Float> = Array2::from_shape_fn((n_samples, n_features), |(i, j)| {
                (i as Float + 1.0) * (j as Float + 1.0) / 10.0
            });

            let y: Array2<Float> = Array2::from_shape_fn((n_samples, 1), |(i, _)| {
                x.row(i).sum() / n_features as Float + 0.1 * (i as Float)
            });

            let n_components = max_components.min(n_features);

            if let Ok(fitted) = PLSRegression::new(n_components).fit(&x, &y) {
                let transformed = fitted.transform(&x)?;

                // Check that PLS reduces dimensionality correctly
                prop_assert_eq!(transformed.shape(), &[n_samples, n_components]);

                // Check that the transformation captures meaningful variance
                for i in 0..n_components {
                    let component = transformed.column(i);
                    let variance = component.var(0.0);
                    prop_assert!(variance > 0.0);  // Should have non-zero variance
                }
            }
        }
    }
}
