//! Lasso regression with built-in cross-validation

use std::marker::PhantomData;

use scirs2_core::ndarray::Axis;
use sklears_core::{
    error::{validate, Result, SklearsError},
    traits::{Estimator, Fit, Predict, Score, Trained, Untrained},
    types::{Array1, Array2, Float},
};
// use sklears_model_selection::{cross_val_score, CrossValidator, KFold, Scoring}; // Temporarily disabled

// Placeholder implementations until sklears_model_selection is available
pub struct KFold {
    n_folds: usize,
    shuffle: bool,
}

impl KFold {
    pub fn new(n_folds: usize) -> Self {
        Self {
            n_folds,
            shuffle: false,
        }
    }

    pub fn shuffle(mut self, shuffle: bool) -> Self {
        self.shuffle = shuffle;
        self
    }

    pub fn random_state(self, _random_state: Option<u64>) -> Self {
        // Placeholder - random state not currently used in this simplified implementation
        self
    }

    pub fn n_splits(&self) -> usize {
        self.n_folds
    }

    pub fn split(
        &self,
        n_samples: usize,
        _groups: Option<&[usize]>,
    ) -> Vec<(Vec<usize>, Vec<usize>)> {
        if n_samples == 0 || self.n_folds == 0 {
            return Vec::new();
        }

        // Create index order (deterministic placeholder for now)
        let mut indices: Vec<usize> = (0..n_samples).collect();
        if self.shuffle && n_samples > 1 {
            // Simple deterministic rotation to mimic shuffling when enabled
            let rotation = (self.n_folds % n_samples).max(1);
            indices.rotate_left(rotation);

            // Distribute indices in a round-robin fashion to encourage balanced folds
            let mut fold_bins: Vec<Vec<usize>> = vec![Vec::new(); self.n_folds];
            for (idx, &sample_idx) in indices.iter().enumerate() {
                fold_bins[idx % self.n_folds].push(sample_idx);
            }

            let mut splits = Vec::with_capacity(self.n_folds);
            for fold_idx in 0..self.n_folds {
                let test_indices = fold_bins[fold_idx].clone();
                let train_indices = fold_bins
                    .iter()
                    .enumerate()
                    .filter(|(idx, _)| *idx != fold_idx)
                    .flat_map(|(_, fold)| fold.iter())
                    .copied()
                    .collect();

                splits.push((train_indices, test_indices));
            }

            return splits;
        }

        let mut splits = Vec::with_capacity(self.n_folds);
        let base_size = n_samples / self.n_folds;
        let remainder = n_samples % self.n_folds;
        let mut current = 0;

        for fold_idx in 0..self.n_folds {
            let mut fold_size = base_size;
            if base_size == 0 {
                // When there are fewer samples than folds, assign one sample per fold until we run out
                fold_size = if fold_idx < n_samples { 1 } else { 0 };
            } else if fold_idx < remainder {
                fold_size += 1;
            }

            let end = (current + fold_size).min(n_samples);
            let test_indices = indices[current..end].to_vec();
            let train_indices = indices
                .iter()
                .take(current)
                .chain(indices.iter().skip(end))
                .copied()
                .collect();

            splits.push((train_indices, test_indices));
            current = end;
        }

        splits
    }
}

/// Cross-validation score calculation for LinearRegression
/// Note: This is a simplified version specifically for LinearRegression to avoid
/// complex trait bound issues with ndarray 0.17
pub fn cross_val_score(
    estimator: LinearRegression<Untrained>,
    x: &Array2<Float>,
    y: &Array1<Float>,
    cv: &KFold,
    _scoring: Option<()>,
    _n_jobs: Option<()>,
) -> Result<Array1<Float>> {
    validate::check_consistent_length(x, y)?;

    let n_samples = x.nrows();
    let splits = cv.split(n_samples, None);

    if splits.is_empty() {
        return Err(SklearsError::InvalidInput(
            "KFold split produced no folds".to_string(),
        ));
    }

    let mut scores = Vec::with_capacity(splits.len());

    for (train_idx, test_idx) in splits {
        if test_idx.is_empty() {
            continue; // Skip empty test folds (can happen when n_samples < n_folds)
        }

        let x_train = x.select(Axis(0), &train_idx);
        let y_train = y.select(Axis(0), &train_idx);
        let x_test = x.select(Axis(0), &test_idx);
        let y_test = y.select(Axis(0), &test_idx);

        let model = estimator.clone().fit(&x_train, &y_train)?;
        let score = model.score(&x_test, &y_test)?;
        scores.push(score);
    }

    if scores.is_empty() {
        return Err(SklearsError::InvalidInput(
            "KFold split resulted in no evaluable folds".to_string(),
        ));
    }

    Ok(Array1::from_vec(scores))
}

use crate::LinearRegression;

/// Configuration for Lasso regression with cross-validation
#[derive(Debug, Clone)]
pub struct LassoCVConfig {
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
    /// Whether to store the cross-validation values for each alpha
    pub store_cv_values: bool,
}

impl Default for LassoCVConfig {
    fn default() -> Self {
        Self {
            alphas: None,
            fit_intercept: true,
            cv: Some(5),
            max_iter: 1000,
            tol: 1e-4,
            n_alphas: 100,
            store_cv_values: false,
        }
    }
}

/// Lasso regression with built-in cross-validation
///
/// This model performs Lasso regression with automatic hyperparameter selection
/// using cross-validation along the regularization path.
#[derive(Debug, Clone)]
pub struct LassoCV<State = Untrained> {
    config: LassoCVConfig,
    state: PhantomData<State>,
    // Trained state fields
    coef_: Option<Array1<Float>>,
    intercept_: Option<Float>,
    alpha_: Option<Float>,
    alphas_: Option<Array1<Float>>,
    cv_values_: Option<Array2<Float>>,
    n_features_: Option<usize>,
}

impl LassoCV<Untrained> {
    /// Create a new LassoCV model
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a builder for LassoCV
    pub fn builder() -> LassoCVBuilder {
        LassoCVBuilder::default()
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

    /// Generate default alpha values along the regularization path
    fn default_alphas(x: &Array2<Float>, y: &Array1<Float>) -> Vec<Float> {
        let n_samples = x.nrows() as Float;

        // Compute alpha_max (the smallest value of alpha for which all coefficients are zero)
        // This is ||X^T y||_∞ / n
        let xy = x.t().dot(y);
        let alpha_max = xy.mapv(Float::abs).fold(0.0, |a, &b| Float::max(a, b)) / n_samples;

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

impl Default for LassoCV<Untrained> {
    fn default() -> Self {
        Self {
            config: LassoCVConfig::default(),
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
            alphas_: None,
            cv_values_: None,
            n_features_: None,
        }
    }
}

impl Estimator for LassoCV<Untrained> {
    type Float = Float;
    type Config = LassoCVConfig;
    type Error = SklearsError;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LassoCV<Untrained> {
    type Fitted = LassoCV<Trained>;

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
            None => Self::default_alphas(x, y),
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
            // Create Lasso model with current alpha
            let lasso = LinearRegression::lasso(alpha)
                .fit_intercept(self.config.fit_intercept)
                .max_iter(self.config.max_iter);

            // Compute cross-validation scores
            let scores = cross_val_score(lasso, x, y, &cv, None, None)?;
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
        let final_model = LinearRegression::lasso(best_alpha)
            .fit_intercept(self.config.fit_intercept)
            .max_iter(self.config.max_iter)
            .fit(x, y)?;

        // Extract coefficients and intercept from the fitted model
        let coef = final_model.coef().to_owned();
        let intercept = final_model.intercept();

        Ok(LassoCV {
            config: self.config.clone(),
            state: PhantomData,
            coef_: Some(coef),
            intercept_: intercept,
            alpha_: Some(best_alpha),
            alphas_: Some(Array1::from_vec(alphas)),
            cv_values_: cv_values,
            n_features_: Some(n_features),
        })
    }
}

impl LassoCV<Trained> {
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

impl Predict<Array2<Float>, Array1<Float>> for LassoCV<Trained> {
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

impl Score<Array2<Float>, Array1<Float>> for LassoCV<Trained> {
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

/// Builder for LassoCV
#[derive(Debug, Default)]
pub struct LassoCVBuilder {
    config: LassoCVConfig,
}

impl LassoCVBuilder {
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

    /// Build the LassoCV model
    pub fn build(self) -> LassoCV<Untrained> {
        LassoCV {
            config: self.config,
            state: PhantomData,
            coef_: None,
            intercept_: None,
            alpha_: None,
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
    fn test_lasso_cv_simple() {
        // Simple test case with sparse solution
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

        let model = LassoCV::new()
            .alphas(vec![0.001, 0.01, 0.1, 1.0])
            .fit(&x, &y)
            .unwrap();

        // First coefficient should be close to 2, second should be 0
        assert_abs_diff_eq!(model.coef()[0], 2.0, epsilon = 0.1);
        assert_abs_diff_eq!(model.coef()[1], 0.0, epsilon = 0.1);

        // Test prediction
        let x_test = array![[9.0, 0.0], [10.0, 0.0]];
        let y_pred = model.predict(&x_test).unwrap();
        assert_abs_diff_eq!(y_pred[0], 18.0, epsilon = 0.5);
        assert_abs_diff_eq!(y_pred[1], 20.0, epsilon = 0.5);
    }

    #[test]
    fn test_lasso_cv_sparsity() {
        // Test that Lasso produces sparse solutions
        let n_samples = 20;
        let n_features = 10;
        let mut x = Array2::zeros((n_samples, n_features));
        let mut y = Array1::zeros(n_samples);

        // Only first 3 features are relevant
        for i in 0..n_samples {
            x[[i, 0]] = i as Float;
            x[[i, 1]] = (i as Float) * 2.0;
            x[[i, 2]] = (i as Float) * 0.5;
            // Add noise to other features
            for j in 3..n_features {
                x[[i, j]] = ((i * j) as Float).sin();
            }
            y[i] = 2.0 * x[[i, 0]] - 1.0 * x[[i, 1]] + 3.0 * x[[i, 2]];
        }

        let model = LassoCV::new().cv(5).fit(&x, &y).unwrap();

        // Check that we have a sparse solution
        let n_nonzero = model.n_nonzero();
        assert!(n_nonzero <= 5); // Should select only relevant features
    }

    #[test]
    fn test_lasso_cv_auto_alphas() {
        // Test automatic alpha generation
        let x = array![
            [1.0, 2.0],
            [2.0, 3.0],
            [3.0, 4.0],
            [4.0, 5.0],
            [5.0, 6.0],
            [6.0, 7.0],
        ];
        let y = array![5.0, 8.0, 11.0, 14.0, 17.0, 20.0];

        let model = LassoCV::new().cv(3).fit(&x, &y).unwrap();

        // Check that alphas were generated
        assert_eq!(model.alphas().len(), 100);

        // Check that alphas are in descending order
        let alphas = model.alphas();
        for i in 1..alphas.len() {
            assert!(alphas[i] < alphas[i - 1]);
        }
    }
}
