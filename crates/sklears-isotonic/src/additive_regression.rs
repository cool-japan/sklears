//! Additive isotonic regression for multi-dimensional data
//!
//! This module implements isotonic regression for multivariate input using an additive model:
//! f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + intercept
//! where each fᵢ is a univariate isotonic function.

use std::marker::PhantomData;
use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result, SklearsError},
    traits::{Estimator, Fit, Predict, Trained, Untrained},
    types::Float,
};

use crate::{isotonic_regression, LossFunction, MonotonicityConstraint};

/// Additive isotonic regression for multi-dimensional data
///
/// Implements isotonic regression for multivariate input using an additive model:
/// f(x) = f₁(x₁) + f₂(x₂) + ... + fₚ(xₚ) + intercept
/// where each fᵢ is a univariate isotonic function.
#[derive(Debug, Clone)]
/// AdditiveIsotonicRegression
pub struct AdditiveIsotonicRegression<State = Untrained> {
    /// Number of features
    pub n_features: usize,
    /// Monotonicity constraints for each feature
    pub constraints: Vec<MonotonicityConstraint>,
    /// Loss function
    pub loss: LossFunction,
    /// Regularization strength
    pub alpha: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,

    // Fitted attributes
    #[allow(dead_code)]
    feature_functions_: Option<Vec<Array1<Float>>>,
    #[allow(dead_code)]
    feature_grids_: Option<Vec<Array1<Float>>>,
    intercept_: Option<Float>,

    _state: PhantomData<State>,
}

impl AdditiveIsotonicRegression<Untrained> {
    /// Create a new additive isotonic regression model
    pub fn new(n_features: usize) -> Self {
        Self {
            n_features,
            constraints: vec![MonotonicityConstraint::Global { increasing: true }; n_features],
            loss: LossFunction::SquaredLoss,
            alpha: 0.0,
            fit_intercept: true,
            feature_functions_: None,
            feature_grids_: None,
            intercept_: None,
            _state: PhantomData,
        }
    }

    /// Set constraints for each feature
    pub fn constraints(mut self, constraints: Vec<MonotonicityConstraint>) -> Self {
        self.constraints = constraints;
        self
    }

    /// Set loss function
    pub fn loss(mut self, loss: LossFunction) -> Self {
        self.loss = loss;
        self
    }

    /// Set regularization strength
    pub fn alpha(mut self, alpha: Float) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }
}

impl Default for AdditiveIsotonicRegression<Untrained> {
    fn default() -> Self {
        Self::new(1)
    }
}

impl Estimator for AdditiveIsotonicRegression<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Untrained> {
    type Fitted = AdditiveIsotonicRegression<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> Result<Self::Fitted> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        if n_samples != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }

        // Initialize feature functions and grids
        let mut feature_functions = Vec::with_capacity(n_features);
        let mut feature_grids = Vec::with_capacity(n_features);

        // Compute intercept if needed
        let intercept = if self.fit_intercept {
            y.mean().unwrap_or(0.0)
        } else {
            0.0
        };

        let y_centered = if self.fit_intercept {
            y - intercept
        } else {
            y.clone()
        };

        // Fit each feature independently using backfitting algorithm
        let max_iter = 100;
        let tolerance = 1e-6;
        let mut residuals = y_centered.clone();

        // Initialize feature functions to zero
        for j in 0..n_features {
            let feature_col = x.column(j);
            let unique_values = self.get_unique_sorted_values(&feature_col);
            feature_grids.push(unique_values.clone());
            feature_functions.push(Array1::zeros(unique_values.len()));
        }

        // Backfitting iterations
        for _iter in 0..max_iter {
            let mut max_change: Float = 0.0;

            for j in 0..n_features {
                let feature_col = x.column(j);

                // Add back the contribution of feature j
                let old_contrib = self.evaluate_feature_function(
                    &feature_functions[j],
                    &feature_grids[j],
                    &feature_col,
                );
                for i in 0..n_samples {
                    residuals[i] += old_contrib[i];
                }

                // Fit isotonic regression for feature j
                let new_function = self.fit_single_feature(&feature_col, &residuals, j)?;
                let new_contrib =
                    self.evaluate_feature_function(&new_function, &feature_grids[j], &feature_col);

                // Subtract the new contribution
                for i in 0..n_samples {
                    residuals[i] -= new_contrib[i];
                }

                // Check convergence
                let change = (&new_function - &feature_functions[j])
                    .mapv(|x| x.abs())
                    .sum();
                max_change = max_change.max(change);

                feature_functions[j] = new_function;
            }

            if max_change < tolerance {
                break;
            }
        }

        Ok(AdditiveIsotonicRegression {
            n_features: self.n_features,
            constraints: self.constraints,
            loss: self.loss,
            alpha: self.alpha,
            fit_intercept: self.fit_intercept,
            feature_functions_: Some(feature_functions),
            feature_grids_: Some(feature_grids),
            intercept_: Some(intercept),
            _state: PhantomData,
        })
    }
}

impl AdditiveIsotonicRegression<Untrained> {
    fn get_unique_sorted_values(&self, feature: &ArrayView1<Float>) -> Array1<Float> {
        let mut values: Vec<Float> = feature.to_vec();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        values.dedup_by(|a, b| (*a - *b).abs() < 1e-10);
        Array1::from(values)
    }

    fn fit_single_feature(
        &self,
        feature: &ArrayView1<Float>,
        targets: &Array1<Float>,
        feature_idx: usize,
    ) -> Result<Array1<Float>> {
        // Map feature values to unique grid points and compute target means
        let grid = self.get_unique_sorted_values(feature);
        let mut grid_targets = vec![Vec::new(); grid.len()];

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point
            let mut closest_idx = 0;
            let mut min_diff = Float::INFINITY;
            for (j, &grid_val) in grid.iter().enumerate() {
                let diff = (feat_val - grid_val).abs();
                if diff < min_diff {
                    min_diff = diff;
                    closest_idx = j;
                }
            }
            grid_targets[closest_idx].push(targets[i]);
        }

        // Compute means for each grid point
        let mut grid_means = Array1::zeros(grid.len());
        for (i, targets) in grid_targets.iter().enumerate() {
            if !targets.is_empty() {
                grid_means[i] = targets.iter().sum::<Float>() / targets.len() as Float;
            }
        }

        // Apply isotonic constraint
        let isotonic_result = match &self.constraints[feature_idx] {
            MonotonicityConstraint::Global { increasing } => {
                isotonic_regression(&grid_means, *increasing)
            }
            _ => {
                return Err(SklearsError::NotImplemented(
                    "Only global constraints supported for additive isotonic regression"
                        .to_string(),
                ))
            }
        };

        Ok(isotonic_result)
    }

    fn evaluate_feature_function(
        &self,
        function: &Array1<Float>,
        grid: &Array1<Float>,
        feature: &ArrayView1<Float>,
    ) -> Array1<Float> {
        let mut result = Array1::zeros(feature.len());

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point and interpolate
            if grid.len() == 1 {
                result[i] = function[0];
                continue;
            }

            // Linear interpolation
            let mut left_idx = 0;
            for j in 0..grid.len() - 1 {
                if feat_val >= grid[j] && feat_val <= grid[j + 1] {
                    left_idx = j;
                    break;
                }
            }

            if left_idx == grid.len() - 1 {
                result[i] = function[left_idx];
            } else {
                let t = (feat_val - grid[left_idx]) / (grid[left_idx + 1] - grid[left_idx]);
                result[i] = function[left_idx] * (1.0 - t) + function[left_idx + 1] * t;
            }
        }

        result
    }
}

impl Predict<Array2<Float>, Array1<Float>> for AdditiveIsotonicRegression<Trained> {
    fn predict(&self, x: &Array2<Float>) -> Result<Array1<Float>> {
        let n_samples = x.nrows();
        let n_features = x.ncols();

        if n_features != self.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected {} features, got {}",
                self.n_features, n_features
            )));
        }

        let feature_functions = self.feature_functions_.as_ref().unwrap();
        let feature_grids = self.feature_grids_.as_ref().unwrap();
        let intercept = self.intercept_.unwrap_or(0.0);

        let mut predictions = Array1::from_elem(n_samples, intercept);

        // Add contribution from each feature
        for j in 0..n_features {
            let feature_col = x.column(j);
            let contrib = self.evaluate_feature_function(
                &feature_functions[j],
                &feature_grids[j],
                &feature_col,
            );
            predictions = predictions + contrib;
        }

        Ok(predictions)
    }
}

impl AdditiveIsotonicRegression<Trained> {
    fn evaluate_feature_function(
        &self,
        function: &Array1<Float>,
        grid: &Array1<Float>,
        feature: &ArrayView1<Float>,
    ) -> Array1<Float> {
        let mut result = Array1::zeros(feature.len());

        for (i, &feat_val) in feature.iter().enumerate() {
            // Find closest grid point and interpolate
            if grid.len() == 1 {
                result[i] = function[0];
                continue;
            }

            // Linear interpolation
            let mut left_idx = 0;
            for j in 0..grid.len() - 1 {
                if feat_val >= grid[j] && feat_val <= grid[j + 1] {
                    left_idx = j;
                    break;
                }
            }

            if left_idx == grid.len() - 1 {
                result[i] = function[left_idx];
            } else {
                let t = (feat_val - grid[left_idx]) / (grid[left_idx + 1] - grid[left_idx]);
                result[i] = function[left_idx] * (1.0 - t) + function[left_idx + 1] * t;
            }
        }

        result
    }
}

/// Function API for additive isotonic regression
pub fn additive_isotonic_regression(
    x: &Array2<Float>,
    y: &Array1<Float>,
    constraints: &[bool],
) -> Result<Array1<Float>> {
    let n_features = x.ncols();
    if constraints.len() != n_features {
        return Err(SklearsError::InvalidInput(
            "Number of constraints must match number of features".to_string(),
        ));
    }

    let constraint_vec: Vec<MonotonicityConstraint> = constraints
        .iter()
        .map(|&increasing| MonotonicityConstraint::Global { increasing })
        .collect();

    let regressor = AdditiveIsotonicRegression::new(n_features).constraints(constraint_vec);

    let fitted = regressor.fit(x, y)?;
    fitted.predict(x)
}