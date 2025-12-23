//! Orthogonal Matching Pursuit (OMP) implementation

use std::marker::PhantomData;

use scirs2_core::ndarray::{Array1, Array2, Axis};
use scirs2_linalg::solve;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::Float,
};

/// Configuration for OrthogonalMatchingPursuit
#[derive(Debug, Clone)]
pub struct OrthogonalMatchingPursuitConfig {
    /// Maximum number of non-zero coefficients in the solution
    pub n_nonzero_coefs: Option<usize>,
    /// Tolerance for the residual
    pub tol: Option<Float>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Whether to normalize/standardize features before fitting
    pub normalize: bool,
}

impl Default for OrthogonalMatchingPursuitConfig {
    fn default() -> Self {
        Self {
            n_nonzero_coefs: None,
            tol: None,
            fit_intercept: true,
            normalize: true,
        }
    }
}

/// Orthogonal Matching Pursuit model
#[derive(Debug, Clone)]
pub struct OrthogonalMatchingPursuit<State = Untrained> {
    config: OrthogonalMatchingPursuitConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    n_features_: Option<usize>,
    n_iter_: Option<usize>,
}

impl OrthogonalMatchingPursuit<Untrained> {
    /// Create a new OMP model
    pub fn new() -> Self {
        Self {
            config: OrthogonalMatchingPursuitConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            n_features_: None,
            n_iter_: None,
        }
    }

    /// Set the maximum number of non-zero coefficients
    pub fn n_nonzero_coefs(mut self, n_nonzero_coefs: usize) -> Self {
        self.config.n_nonzero_coefs = Some(n_nonzero_coefs);
        self
    }

    /// Set the tolerance for the residual
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = Some(tol);
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.config.normalize = normalize;
        self
    }
}

impl Default for OrthogonalMatchingPursuit<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for OrthogonalMatchingPursuit<Untrained> {
    type Float = Float;
    type Config = OrthogonalMatchingPursuitConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for OrthogonalMatchingPursuit<Untrained> {
    type Fitted = OrthogonalMatchingPursuit<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        // Validate inputs
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Determine stopping criterion
        let max_features = if let Some(n) = self.config.n_nonzero_coefs {
            n.min(n_features).min(n_samples)
        } else if self.config.tol.is_some() {
            n_features.min(n_samples)
        } else {
            // Default: min(n_features, n_samples)
            n_features.min(n_samples)
        };

        let tol = self.config.tol.unwrap_or(1e-3);

        // Center X and y
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let mut x_centered = x - &x_mean;

        let y_mean = if self.config.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };
        let y_centered = y - y_mean;

        // Normalize X if requested
        let x_scale = if self.config.normalize {
            let mut scale = Array1::zeros(n_features);
            for j in 0..n_features {
                let col = x_centered.column(j);
                scale[j] = col.dot(&col).sqrt();
                if scale[j] > Float::EPSILON {
                    x_centered.column_mut(j).mapv_inplace(|x| x / scale[j]);
                } else {
                    scale[j] = 1.0;
                }
            }
            scale
        } else {
            Array1::ones(n_features)
        };

        // Initialize OMP algorithm
        let mut coef = Array1::zeros(n_features);
        let mut active: Vec<usize> = Vec::new();
        let mut residual = y_centered.clone();
        let mut n_iter = 0;

        // Main OMP loop
        for _ in 0..max_features {
            // Compute correlations with residual
            let correlations = x_centered.t().dot(&residual);

            // Find the most correlated feature not yet selected
            let mut max_corr = 0.0;
            let mut best_idx = 0;

            for j in 0..n_features {
                if !active.contains(&j) {
                    let corr = correlations[j].abs();
                    if corr > max_corr {
                        max_corr = corr;
                        best_idx = j;
                    }
                }
            }

            // Check stopping criterion
            let residual_norm = residual.dot(&residual).sqrt();
            if residual_norm < tol {
                break;
            }

            // Add the best feature to active set
            active.push(best_idx);
            n_iter += 1;

            // Solve least squares problem on active set
            let n_active = active.len();
            let mut x_active = Array2::zeros((n_samples, n_active));
            for (i, &j) in active.iter().enumerate() {
                x_active.column_mut(i).assign(&x_centered.column(j));
            }

            // Solve normal equations: (X_active^T X_active) coef_active = X_active^T y
            let gram = x_active.t().dot(&x_active);
            let x_active_t_y = x_active.t().dot(&y_centered);

            // Add small regularization to avoid singular matrix
            let mut gram_reg = gram.clone();
            for i in 0..n_active {
                gram_reg[[i, i]] += 1e-10;
            }

            let coef_active = &solve(&gram_reg.view(), &x_active_t_y.view(), None)
                .map_err(|e| SklearsError::NumericalError(format!("Failed to solve: {}", e)))?;

            // Update full coefficient vector
            coef.fill(0.0);
            for (i, &j) in active.iter().enumerate() {
                coef[j] = coef_active[i];
            }

            // Update residual
            residual = &y_centered - &x_centered.dot(&coef);
        }

        // Rescale coefficients if we normalized
        if self.config.normalize {
            for j in 0..n_features {
                if x_scale[j] > 0.0 {
                    coef[j] /= x_scale[j];
                }
            }
        }

        // Compute intercept if needed
        let intercept = if self.config.fit_intercept {
            Some(y_mean - x_mean.dot(&coef))
        } else {
            None
        };

        Ok(OrthogonalMatchingPursuit {
            config: self.config,
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            n_features_: Some(n_features),
            n_iter_: Some(n_iter),
        })
    }
}

impl OrthogonalMatchingPursuit<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model is trained")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Option<Float> {
        self.intercept_
    }

    /// Get the number of iterations run
    pub fn n_iter(&self) -> usize {
        self.n_iter_.expect("Model is trained")
    }
}

impl Predict<Array2<Float>, Array1<Float>> for OrthogonalMatchingPursuit<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_features = self.n_features_.expect("Model is trained");
        validate::check_n_features(x, n_features)?;

        let coef = self.coef_.as_ref().expect("Model is trained");
        let mut predictions = x.dot(coef);

        if let Some(intercept) = self.intercept_ {
            predictions += intercept;
        }

        Ok(predictions)
    }
}

impl Score<Array2<Float>, Array1<Float>> for OrthogonalMatchingPursuit<Trained> {
    type Float = Float;

    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        let predictions = self.predict(x)?;

        // Calculate R² score
        let ss_res = (&predictions - y).mapv(|x| x * x).sum();
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_tot = y.mapv(|yi| (yi - y_mean).powi(2)).sum();

        if ss_tot == 0.0 {
            return Ok(1.0);
        }

        Ok(1.0 - (ss_res / ss_tot))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_omp_simple() {
        // Simple test with orthogonal features
        let x = array![
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ];
        let y = array![2.0, 3.0, 2.0, 3.0, 4.0, 6.0]; // y = 2*x1 + 3*x2

        let model = OrthogonalMatchingPursuit::new()
            .fit_intercept(false)
            .normalize(false)
            .fit(&x, &y)
            .unwrap();

        // Should recover the true coefficients
        let coef = model.coef();
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(coef[1], 3.0, epsilon = 1e-5);

        // Predictions should be perfect
        let predictions = model.predict(&x).unwrap();
        for i in 0..y.len() {
            assert_abs_diff_eq!(predictions[i], y[i], epsilon = 1e-5);
        }
    }

    #[test]
    fn test_omp_max_features() {
        // Test limiting number of features
        let x = array![
            [1.0, 0.1, 0.01],
            [2.0, 0.2, 0.02],
            [3.0, 0.3, 0.03],
            [4.0, 0.4, 0.04],
            [5.0, 0.5, 0.05],
            [6.0, 0.6, 0.06],
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0]; // y = 2*x1

        let model = OrthogonalMatchingPursuit::new()
            .n_nonzero_coefs(1)
            .fit_intercept(false)
            .normalize(false)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert_eq!(n_nonzero, 1);

        // First coefficient should be selected and close to 2.0
        assert_abs_diff_eq!(coef[0], 2.0, epsilon = 1e-3);

        // Check that we ran exactly 1 iteration
        assert_eq!(model.n_iter(), 1);
    }

    #[test]
    fn test_omp_tolerance() {
        // Test stopping based on tolerance
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0]];
        let y = array![2.1, 3.9, 6.05, 7.95, 10.1]; // y ≈ 2x with small noise

        let model = OrthogonalMatchingPursuit::new()
            .tol(0.5) // Relatively high tolerance
            .fit_intercept(false)
            .fit(&x, &y)
            .unwrap();

        // Should get a reasonable approximation
        let _predictions = model.predict(&x).unwrap();
        let r2 = model.score(&x, &y).unwrap();
        assert!(r2 > 0.95);
    }

    #[test]
    fn test_omp_with_intercept() {
        let x = array![[1.0], [2.0], [3.0], [4.0]];
        let y = array![3.0, 5.0, 7.0, 9.0]; // y = 2x + 1

        let model = OrthogonalMatchingPursuit::new()
            .fit_intercept(true)
            .fit(&x, &y)
            .unwrap();

        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 1e-5);
        assert_abs_diff_eq!(model.intercept().unwrap(), 1.0, epsilon = 1e-5);
    }

    #[test]
    fn test_omp_sparse_recovery() {
        // Create sparse signal recovery problem
        let n_samples = 20;
        let n_features = 10;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut true_coef = Array1::zeros(n_features);

        // Generate random-like data deterministically
        for i in 0..n_samples {
            for j in 0..n_features {
                x[[i, j]] = ((i * 7 + j * 13) % 20) as Float / 10.0 - 1.0;
            }
        }

        // True coefficients are sparse (only 3 non-zero)
        true_coef[1] = 2.0;
        true_coef[4] = -1.5;
        true_coef[7] = 1.0;

        let y = x.dot(&true_coef);

        let model = OrthogonalMatchingPursuit::new()
            .n_nonzero_coefs(3)
            .fit_intercept(false)
            .normalize(true)
            .fit(&x, &y)
            .unwrap();

        let coef = model.coef();

        // Should recover the support (non-zero indices)
        for j in 0..n_features {
            if true_coef[j] != 0.0 {
                assert!(
                    coef[j].abs() > 0.1,
                    "Failed to recover non-zero coefficient at index {}",
                    j
                );
            }
        }

        // Should have exactly 3 non-zero coefficients
        let n_nonzero = coef.iter().filter(|&&c| c.abs() > 1e-10).count();
        assert_eq!(n_nonzero, 3);
    }
}
