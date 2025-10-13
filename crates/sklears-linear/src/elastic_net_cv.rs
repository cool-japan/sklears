//! ElasticNet regression with built-in cross-validation

use std::marker::PhantomData;

use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};
// use sklears_model_selection::{cross_val_score, CrossValidator, KFold, Scoring}; // Temporarily disabled
use crate::lasso_cv::{cross_val_score, KFold};

use crate::LinearRegression;

/// Configuration for ElasticNet regression with cross-validation
#[derive(Debug, Clone)]
pub struct ElasticNetCVConfig {
    /// L1 ratio candidates (between 0 and 1).
    /// If None, will use [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    pub l1_ratios: Option<Vec<Float>>,
    /// Regularization strength candidates; must be positive floats.
    /// If None, will use logspace(-4, 1, 100)
    pub alphas: Option<Vec<Float>>,
    /// Whether to fit the intercept
    pub fit_intercept: bool,
    /// Cross-validation splitting strategy
    pub cv: Option<usize>,
    /// Maximum number of iterations for coordinate descent
    pub max_iter: usize,
    /// Tolerance for convergence
    pub tol: Float,
    /// Number of alphas along the regularization path
    pub n_alphas: usize,
    /// Whether to store the cross-validation values
    pub store_cv_values: bool,
    /// Whether to use warm start (reuse previous solution as initialization)
    pub warm_start: bool,
}

impl Default for ElasticNetCVConfig {
    fn default() -> Self {
        Self {
            l1_ratios: None,
            alphas: None,
            fit_intercept: true,
            cv: Some(5),
            max_iter: 1000,
            tol: 1e-4,
            n_alphas: 100,
            store_cv_values: false,
            warm_start: true,
        }
    }
}

/// ElasticNet regression with built-in cross-validation
///
/// This model performs ElasticNet regression with automatic hyperparameter selection
/// using cross-validation. It optimizes both alpha and l1_ratio parameters.
#[derive(Debug, Clone)]
pub struct ElasticNetCV<State = Untrained> {
    config: ElasticNetCVConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    alpha_: Option<Float>,
    l1_ratio_: Option<Float>,
    alphas_: Option<Array1<Float>>,
    cv_values_: Option<Array2<Float>>,
    n_features_: Option<usize>,
}

impl ElasticNetCV<Untrained> {
    /// Create a new ElasticNetCV model
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for ElasticNetCV
    pub fn builder() -> ElasticNetCVBuilder {
        ElasticNetCVBuilder::default()
    }

    /// Set the l1_ratio candidates
    pub fn l1_ratios(mut self, l1_ratios: Vec<Float>) -> Self {
        self.config.l1_ratios = Some(l1_ratios);
        self
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

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to store cross-validation values
    pub fn store_cv_values(mut self, store: bool) -> Self {
        self.config.store_cv_values = store;
        self
    }

    /// Set whether to use warm start
    pub fn warm_start(mut self, warm_start: bool) -> Self {
        self.config.warm_start = warm_start;
        self
    }

    /// Generate default l1_ratio values
    fn default_l1_ratios() -> Vec<Float> {
        vec![0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]
    }

    /// Generate default alpha values along the regularization path
    fn default_alphas(x: &Array2<Float>, y: &Array1<Float>, l1_ratio: Float) -> Vec<Float> {
        let n_samples = x.nrows() as Float;

        // Compute alpha_max (the smallest value of alpha for which all coefficients are zero)
        // For ElasticNet: alpha_max = ||X^T y||_∞ / (n * l1_ratio) when l1_ratio > 0
        let xy = x.t().dot(y);
        let alpha_max = if l1_ratio > 0.0 {
            xy.mapv(Float::abs).fold(0.0, |a, &b| Float::max(a, b)) / (n_samples * l1_ratio)
        } else {
            // Pure Ridge case
            xy.mapv(Float::abs).fold(0.0, |a, &b| Float::max(a, b)) / n_samples
        };

        // Generate log-spaced alphas from alpha_max to alpha_max * eps
        let eps = 1e-3; // ratio of smallest to largest alpha
        let alpha_min = alpha_max * eps;

        let mut alphas = Vec::with_capacity(100);
        for i in 0..100 {
            let ratio = (i as Float) / 99.0;
            let log_alpha = (1.0 - ratio) * alpha_max.ln() + ratio * alpha_min.ln();
            alphas.push(log_alpha.exp());
        }

        alphas
    }
}

impl Default for ElasticNetCV<Untrained> {
    fn default() -> Self {
        Self {
            config: ElasticNetCVConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            l1_ratio_: None,
            alphas_: None,
            cv_values_: None,
            n_features_: None,
        }
    }
}

impl Estimator for ElasticNetCV<Untrained> {
    type Float = Float;
    type Config = ElasticNetCVConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ElasticNetCV<Untrained> {
    type Fitted = ElasticNetCV<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let n_samples = x.nrows();
        let n_features = x.ncols();

        // Get l1_ratio candidates
        let l1_ratios = match &self.config.l1_ratios {
            Some(ratios) => {
                if ratios.is_empty() {
                    return Err(SklearsError::InvalidParameter {
                        name: "l1_ratios".to_string(),
                        reason: "cannot be empty".to_string(),
                    });
                }
                for &ratio in ratios {
                    if !(0.0..=1.0).contains(&ratio) {
                        return Err(SklearsError::InvalidParameter {
                            name: "l1_ratios".to_string(),
                            reason: "must be between 0 and 1".to_string(),
                        });
                    }
                }
                ratios.clone()
            }
            None => Self::default_l1_ratios(),
        };

        // Get CV strategy
        let n_folds = self.config.cv.unwrap_or(5);
        let cv = KFold::new(n_folds);

        let mut best_alpha = 0.0;
        let mut best_l1_ratio = 0.0;
        let mut best_score = Float::NEG_INFINITY;
        let mut all_alphas = Vec::new();

        // Try each l1_ratio
        for &l1_ratio in &l1_ratios {
            // Get alpha candidates for this l1_ratio
            let alphas = match &self.config.alphas {
                Some(alphas) => {
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
                None => Self::default_alphas(x, y, l1_ratio),
            };

            if l1_ratio == l1_ratios[0] {
                all_alphas = alphas.clone();
            }

            // Try each alpha value for this l1_ratio with warm start
            let mut previous_coef: Option<Array1<Float>> = None;
            let mut previous_intercept: Option<Float> = None;

            // Sort alphas in descending order for better warm start performance
            // (start with most regularized, move to less regularized)
            let mut sorted_alphas = alphas.clone();
            sorted_alphas.sort_by(|a, b| b.partial_cmp(a).unwrap());

            for &alpha in &sorted_alphas {
                let elastic_net = if self.config.warm_start && previous_coef.is_some() {
                    // Use warm start with previous coefficients
                    LinearRegression::elastic_net(alpha, l1_ratio)
                        .fit_intercept(self.config.fit_intercept)
                        .max_iter(self.config.max_iter)
                        .warm_start(true)
                } else {
                    // Standard initialization
                    LinearRegression::elastic_net(alpha, l1_ratio)
                        .fit_intercept(self.config.fit_intercept)
                        .max_iter(self.config.max_iter)
                };

                // Create model for cross-validation
                let cv_model = LinearRegression::elastic_net(alpha, l1_ratio)
                    .fit_intercept(self.config.fit_intercept)
                    .max_iter(self.config.max_iter);

                // If warm start is enabled, fit a temporary model to get coefficients for next iteration
                if self.config.warm_start {
                    let temp_model = if previous_coef.is_some() {
                        let model = LinearRegression::elastic_net(alpha, l1_ratio)
                            .fit_intercept(self.config.fit_intercept)
                            .max_iter(self.config.max_iter);
                        model.fit_with_warm_start(
                            x,
                            y,
                            previous_coef.as_ref(),
                            previous_intercept,
                        )?
                    } else {
                        cv_model.clone().fit(x, y)?
                    };

                    previous_coef = Some(temp_model.coef().clone());
                    previous_intercept = temp_model.intercept();
                }

                // Compute cross-validation scores
                let scores = cross_val_score(cv_model, x, y, &cv, None, None)?;
                let mean_score = scores.mean().unwrap_or(0.0);

                // Update best parameters if this combination is better
                if mean_score > best_score {
                    best_score = mean_score;
                    best_alpha = alpha;
                    best_l1_ratio = l1_ratio;
                }
            }
        }

        // Store CV values if requested (simplified: only for best l1_ratio)
        let cv_values = if self.config.store_cv_values {
            let alphas = match &self.config.alphas {
                Some(alphas) => alphas.clone(),
                None => Self::default_alphas(x, y, best_l1_ratio),
            };

            let mut cv_vals = Array2::zeros((alphas.len(), n_folds));

            for (alpha_idx, &alpha) in alphas.iter().enumerate() {
                let elastic_net = LinearRegression::elastic_net(alpha, best_l1_ratio)
                    .fit_intercept(self.config.fit_intercept)
                    .max_iter(self.config.max_iter);

                let scores = cross_val_score(elastic_net, x, y, &cv, None, None)?;
                for (fold_idx, &score) in scores.iter().enumerate() {
                    cv_vals[[alpha_idx, fold_idx]] = score;
                }
            }

            Some(cv_vals)
        } else {
            None
        };

        // Fit final model with best parameters on all data
        let final_model = LinearRegression::elastic_net(best_alpha, best_l1_ratio)
            .fit_intercept(self.config.fit_intercept)
            .max_iter(self.config.max_iter)
            .fit(x, y)?;

        // Extract coefficients and intercept from the fitted model
        let coef = final_model.coef().to_owned();
        let intercept = final_model.intercept();

        Ok(ElasticNetCV {
            config: self.config.clone(),
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            alpha_: Some(best_alpha),
            l1_ratio_: Some(best_l1_ratio),
            alphas_: Some(Array1::from_vec(all_alphas)),
            cv_values_: cv_values,
            n_features_: Some(n_features),
        })
    }
}

impl ElasticNetCV<Trained> {
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

    /// Get the selected l1_ratio value
    pub fn l1_ratio(&self) -> Float {
        self.l1_ratio_.expect("Model not fitted")
    }

    /// Get all alpha values tested
    pub fn alphas(&self) -> &Array1<Float> {
        self.alphas_.as_ref().expect("Model not fitted")
    }

    /// Get the cross-validation values (if stored)
    pub fn cv_values(&self) -> Option<&Array2<Float>> {
        self.cv_values_.as_ref()
    }

    /// Get the number of non-zero coefficients
    pub fn n_nonzero(&self) -> usize {
        self.coef()
            .iter()
            .filter(|&&coef| coef.abs() > 1e-10)
            .count()
    }
}

impl Predict<Array2<Float>, Array1<Float>> for ElasticNetCV<Trained> {
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

impl Score<Array2<Float>, Array1<Float>> for ElasticNetCV<Trained> {
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

/// Builder for ElasticNetCV
#[derive(Debug, Default)]
pub struct ElasticNetCVBuilder {
    config: ElasticNetCVConfig,
}

impl ElasticNetCVBuilder {
    /// Set the l1_ratio candidates
    pub fn l1_ratios(mut self, l1_ratios: Vec<Float>) -> Self {
        self.config.l1_ratios = Some(l1_ratios);
        self
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

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set whether to store cross-validation values
    pub fn store_cv_values(mut self, store: bool) -> Self {
        self.config.store_cv_values = store;
        self
    }

    /// Build the ElasticNetCV model
    pub fn build(self) -> ElasticNetCV<Untrained> {
        ElasticNetCV {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            l1_ratio_: None,
            alphas_: None,
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
    fn test_elastic_net_cv_simple() {
        // Simple test case with mixture of L1 and L2
        let x = array![
            [1.0, 0.0],
            [2.0, 0.0],
            [3.0, 0.0],
            [4.0, 0.0],
            [5.0, 0.0],
            [6.0, 0.0],
            [7.0, 0.0],
            [8.0, 0.0],
        ];
        let y = array![2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0]; // y = 2*x1

        let model = ElasticNetCV::new()
            .alphas(vec![0.001, 0.01, 0.1, 1.0])
            .l1_ratios(vec![0.1, 0.5, 0.9])
            .fit(&x, &y)
            .unwrap();

        // First coefficient should be close to 2
        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.2);

        // Test prediction
        let x_test = array![[9.0, 0.0], [10.0, 0.0]];
        let y_pred = model.predict(&x_test).unwrap();
        assert_abs_diff_eq!(y_pred[0], 18.0, epsilon = 0.5);
        assert_abs_diff_eq!(y_pred[1], 20.0, epsilon = 0.5);
    }

    #[test]
    fn test_elastic_net_cv_l1_ratio_selection() {
        // Test that ElasticNet selects appropriate l1_ratio
        let n_samples = 20;
        let n_features = 10;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        // Create correlated features
        for i in 0..n_samples {
            x[[i, 0]] = i as Float;
            x[[i, 1]] = (i as Float) + 0.1 * (i as Float).sin(); // Correlated with feature 0
            x[[i, 2]] = (i as Float) * 0.5;
            // Add noise to other features
            for j in 3..n_features {
                x[[i, j]] = ((i * j) as Float).sin();
            }
            y[i] = 2.0 * x[[i, 0]] + 1.5 * x[[i, 1]] + 3.0 * x[[i, 2]];
        }

        let model = ElasticNetCV::new().cv(5).fit(&x, &y).unwrap();

        // Check that l1_ratio is not at extremes (pure Ridge or pure Lasso)
        let l1_ratio = model.l1_ratio();
        assert!(l1_ratio > 0.0 && l1_ratio < 1.0);
    }

    #[test]
    fn test_elastic_net_cv_store_values() {
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];

        let model = ElasticNetCV::new()
            .alphas(vec![0.1, 1.0, 10.0])
            .l1_ratios(vec![0.5, 0.7, 0.9])
            .cv(3)
            .store_cv_values(true)
            .fit(&x, &y)
            .unwrap();

        // Check that CV values were stored
        let cv_values = model.cv_values().unwrap();
        assert_eq!(cv_values.shape(), &[3, 3]); // 3 alphas, 3 folds
    }
}
