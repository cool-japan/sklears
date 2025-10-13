//! Ridge regression with built-in cross-validation

use std::marker::PhantomData;

use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};
// use sklears_model_selection::{cross_val_score, CrossValidator, KFold, Scoring}; // Temporarily disabled

use crate::lasso_cv::{cross_val_score, KFold};
use crate::LinearRegression;

/// Configuration for Ridge regression with cross-validation
#[derive(Debug, Clone)]
pub struct RidgeCVConfig {
    /// Regularization strength candidates; must be positive floats.
    /// If None, will use logspace(-6, 3, 10)
    pub alphas: Option<Vec<Float>>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Cross-validation splitting strategy
    pub cv: Option<usize>,
    /// Whether to store the cross-validation values for each alpha
    pub store_cv_values: bool,
}

impl Default for RidgeCVConfig {
    fn default() -> Self {
        Self {
            alphas: None,
            fit_intercept: true,
            cv: Some(5),
            store_cv_values: false,
        }
    }
}

/// Ridge regression with built-in cross-validation
///
/// This model performs Ridge regression with automatic hyperparameter selection
/// using cross-validation.
#[derive(Debug, Clone)]
pub struct RidgeCV<State = Untrained> {
    config: RidgeCVConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    alpha_: Option<Float>,
    cv_values_: Option<Array2<Float>>,
    n_features_: Option<usize>,
}

impl RidgeCV<Untrained> {
    /// Create a new RidgeCV model
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for RidgeCV
    pub fn builder() -> RidgeCVBuilder {
        RidgeCVBuilder::default()
    }

    /// Set the alpha candidates
    pub fn alphas(mut self, alphas: Vec<Float>) -> Self {
        self.config.alphas = Some(alphas);
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv(mut self, cv: usize) -> Self {
        self.config.cv = Some(cv);
        self
    }

    /// Set whether to store cross-validation values
    pub fn store_cv_values(mut self, store: bool) -> Self {
        self.config.store_cv_values = store;
        self
    }

    /// Generate default alpha values
    fn default_alphas() -> Vec<Float> {
        // Generate 10 values logarithmically spaced between 10^-6 and 10^3
        let mut alphas = Vec::with_capacity(10);
        for i in 0..10 {
            let log_alpha = -6.0 + (i as Float) * 9.0 / 9.0;
            alphas.push(10.0_f64.powf(log_alpha));
        }
        alphas
    }
}

impl Default for RidgeCV<Untrained> {
    fn default() -> Self {
        Self {
            config: RidgeCVConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            cv_values_: None,
            n_features_: None,
        }
    }
}

impl Estimator for RidgeCV<Untrained> {
    type Float = Float;
    type Config = RidgeCVConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for RidgeCV<Untrained> {
    type Fitted = RidgeCV<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let _n_samples = x.nrows();
        let n_features = x.ncols();

        // Get alpha candidates
        let alphas = match &self.config.alphas {
            Some(alphas) => {
                if alphas.is_empty() {
                    return Err(SklearsError::InvalidParameter {
                        name: "alphas".to_string(),
                        reason: "cannot be empty".to_string(),
                    });
                }
                for &alpha in alphas {
                    if alpha <= 0.0 {
                        return Err(SklearsError::InvalidParameter {
                            name: "alphas".to_string(),
                            reason: "must be positive".to_string(),
                        });
                    }
                }
                alphas.clone()
            }
            None => Self::default_alphas(),
        };

        // Get CV strategy
        let n_folds = self.config.cv.unwrap_or(5);
        let cv = KFold::new(n_folds);

        // Store CV scores for each alpha
        let mut cv_values = if self.config.store_cv_values {
            Some(Array2::zeros((alphas.len(), n_folds)))
        } else {
            None
        };

        let mut best_alpha = alphas[0];
        let mut best_score = Float::NEG_INFINITY;

        // Try each alpha value
        for (alpha_idx, &alpha) in alphas.iter().enumerate() {
            // Create Ridge model with current alpha
            let ridge = LinearRegression::new()
                .fit_intercept(self.config.fit_intercept)
                .regularization(alpha);

            // Compute cross-validation scores
            let scores = cross_val_score(ridge, x, y, &cv, None, None)?;
            let mean_score = scores.mean().unwrap_or(0.0);

            // Store CV values if requested
            if let Some(ref mut cv_vals) = cv_values {
                for (fold_idx, &score) in scores.iter().enumerate() {
                    cv_vals[[alpha_idx, fold_idx]] = score;
                }
            }

            // Update best alpha if this one is better
            if mean_score > best_score {
                best_score = mean_score;
                best_alpha = alpha;
            }
        }

        // Fit final model with best alpha on all data
        let final_model = LinearRegression::new()
            .fit_intercept(self.config.fit_intercept)
            .regularization(best_alpha)
            .fit(x, y)?;

        // Extract coefficients and intercept from the fitted model
        let coef = final_model.coef().to_owned();
        let intercept = final_model.intercept();

        Ok(RidgeCV {
            config: self.config.clone(),
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            alpha_: Some(best_alpha),
            cv_values_: cv_values,
            n_features_: Some(n_features),
        })
    }
}

impl RidgeCV<Trained> {
    /// Get the coefficients
    pub fn coef(&self) -> &Array1<Float> {
        self.coef_.as_ref().expect("Model not fitted")
    }

    /// Get the intercept
    pub fn intercept(&self) -> Float {
        self.intercept_.unwrap_or(0.0)
    }

    /// Get the selected alpha value
    pub fn alpha(&self) -> Float {
        self.alpha_.expect("Model not fitted")
    }

    /// Get the cross-validation values (if stored)
    pub fn cv_values(&self) -> Option<&Array2<Float>> {
        self.cv_values_.as_ref()
    }
}

impl Predict<Array2<Float>, Array1<Float>> for RidgeCV<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        // Validate input

        let n_features = self.n_features_.expect("Model not fitted");
        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let coef = self.coef();
        let mut y_pred = x.dot(coef);

        if self.config.fit_intercept {
            y_pred += self.intercept();
        }

        Ok(y_pred)
    }
}

impl Score<Array2<Float>, Array1<Float>> for RidgeCV<Trained> {
    type Float = Float;

    /// Return the coefficient of determination R² of the prediction
    fn score(&self, x: &Array2<Float>, y: &Array1<Float>) -> Result<f64> {
        validate::check_consistent_length(x, y)?;

        let y_pred = self.predict(x)?;
        let ss_res = (&y_pred - y).mapv(|e| e * e).sum();
        let y_mean = y.mean().unwrap_or(0.0);
        let ss_tot = y.mapv(|yi| (yi - y_mean) * (yi - y_mean)).sum();

        if ss_tot == 0.0 {
            Ok(0.0)
        } else {
            Ok(1.0 - ss_res / ss_tot)
        }
    }
}

/// Builder for RidgeCV
#[derive(Debug, Default)]
pub struct RidgeCVBuilder {
    config: RidgeCVConfig,
}

impl RidgeCVBuilder {
    /// Set the alpha candidates
    pub fn alphas(mut self, alphas: Vec<Float>) -> Self {
        self.config.alphas = Some(alphas);
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.config.fit_intercept = fit_intercept;
        self
    }

    /// Set the number of cross-validation folds
    pub fn cv(mut self, cv: usize) -> Self {
        self.config.cv = Some(cv);
        self
    }

    /// Set whether to store cross-validation values
    pub fn store_cv_values(mut self, store: bool) -> Self {
        self.config.store_cv_values = store;
        self
    }

    /// Build the RidgeCV model
    pub fn build(self) -> RidgeCV<Untrained> {
        RidgeCV {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            cv_values_: None,
            n_features_: None,
        }
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_ridge_cv_simple() {
        // Simple test case: y = 2x with noise
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]];
        let y = array![2.1, 3.9, 6.1, 7.9, 10.1, 11.9, 14.1, 15.9];

        let model = RidgeCV::new()
            .alphas(vec![0.01, 0.1, 1.0, 10.0])
            .fit(&x, &y)
            .unwrap();

        // Check that coefficients are close to expected value
        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.1);

        // Test prediction
        let x_test = array![[9.0], [10.0]];
        let y_pred = model.predict(&x_test).unwrap();
        assert_abs_diff_eq!(y_pred[0], 18.0, epsilon = 0.5);
        assert_abs_diff_eq!(y_pred[1], 20.0, epsilon = 0.5);

        // Check score
        let score = model.score(&x, &y).unwrap();
        assert!(score > 0.95);
    }

    #[test]
    fn test_ridge_cv_multiple_features() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0]; // y = 1*x1 + 2*x2

        let model = RidgeCV::new()
            .alphas(vec![0.001, 0.01, 0.1, 1.0])
            .cv(3)
            .fit(&x, &y)
            .unwrap();

        // Check coefficients (with more tolerance due to small sample size)
        assert_abs_diff_eq!(model.coef()[0], 1.0, epsilon = 0.6);
        assert_abs_diff_eq!(model.coef()[1], 2.0, epsilon = 0.6);
    }

    #[test]
    fn test_ridge_cv_store_values() {
        let x = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0];

        let model = RidgeCV::new()
            .alphas(vec![0.1, 1.0, 10.0])
            .cv(3)
            .store_cv_values(true)
            .fit(&x, &y)
            .unwrap();

        // Check that CV values were stored
        let cv_values = model.cv_values().unwrap();
        assert_eq!(cv_values.shape(), &[3, 3]); // 3 alphas, 3 folds
    }
}
