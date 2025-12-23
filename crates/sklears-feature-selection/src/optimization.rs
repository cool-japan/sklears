//! Optimization-based feature selection algorithms
//!
//! This module provides advanced optimization methods for feature selection including
//! convex optimization, semidefinite programming, and proximal gradient methods.

use crate::base::{FeatureSelector, SelectorMixin};
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// Convex optimization-based feature selection
#[derive(Debug, Clone)]
pub struct ConvexFeatureSelector<State = Untrained> {
    k: usize,
    regularization: Float,
    max_iter: usize,
    tolerance: Float,
    state: PhantomData<State>,
    // Trained state
    weights_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    objective_values_: Option<Vec<Float>>,
}

impl Default for ConvexFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvexFeatureSelector<Untrained> {
    /// Create a new convex feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            regularization: 1.0,
            max_iter: 1000,
            tolerance: 1e-6,
            state: PhantomData,
            weights_: None,
            selected_features_: None,
            n_features_: None,
            objective_values_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Solve convex optimization problem for feature selection
    fn solve_convex_optimization(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Vec<Float>)> {
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Initialize weights
        let mut weights = Array1::from_elem(n_features, 1.0 / n_features as Float);
        let mut objective_values = Vec::new();

        // Gradient descent with L1 regularization
        for iter in 0..self.max_iter {
            // Compute predictions
            let predictions = features.dot(&weights);

            // Compute residuals
            let residuals = &predictions - target;

            // Compute gradient
            let data_gradient = features.t().dot(&residuals) / n_samples as Float;

            // L1 regularization subgradient
            let reg_gradient = weights.mapv(|w| {
                if w > 0.0 {
                    self.regularization
                } else if w < 0.0 {
                    -self.regularization
                } else {
                    0.0 // Subgradient at 0
                }
            });

            let gradient = data_gradient + reg_gradient;

            // Compute step size (simple fixed step)
            let step_size = 0.01 / (iter + 1) as Float;

            // Update weights
            let new_weights = &weights - step_size * &gradient;

            // Apply non-negativity constraint (for simplicity)
            let new_weights = new_weights.mapv(|w| w.max(0.0));

            // Compute objective value
            let data_term = residuals.mapv(|r| r * r).sum() / (2.0 * n_samples as Float);
            let reg_term = self.regularization * weights.mapv(|w| w.abs()).sum();
            let objective = data_term + reg_term;
            objective_values.push(objective);

            // Check convergence
            let weight_diff = (&new_weights - &weights).mapv(|d| d.abs()).sum();
            if weight_diff < self.tolerance {
                break;
            }

            weights = new_weights;
        }

        Ok((weights, objective_values))
    }
}

impl Estimator for ConvexFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ConvexFeatureSelector<Untrained> {
    type Fitted = ConvexFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Solve convex optimization
        let (weights, objective_values) = self.solve_convex_optimization(features, target)?;

        // Select top k features based on weights
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            weights[b]
                .partial_cmp(&weights[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(ConvexFeatureSelector {
            k: self.k,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            state: PhantomData,
            weights_: Some(weights),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            objective_values_: Some(objective_values),
        })
    }
}

impl Transform<Array2<Float>> for ConvexFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ConvexFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for ConvexFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl ConvexFeatureSelector<Trained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array1<Float> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the objective values during optimization
    pub fn objective_values(&self) -> &[Float] {
        self.objective_values_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Proximal gradient method for feature selection
#[derive(Debug, Clone)]
pub struct ProximalGradientSelector<State = Untrained> {
    k: usize,
    regularization: Float,
    max_iter: usize,
    tolerance: Float,
    step_size: Float,
    state: PhantomData<State>,
    // Trained state
    weights_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    objective_values_: Option<Vec<Float>>,
}

impl Default for ProximalGradientSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ProximalGradientSelector<Untrained> {
    /// Create a new proximal gradient selector
    pub fn new() -> Self {
        Self {
            k: 10,
            regularization: 1.0,
            max_iter: 1000,
            tolerance: 1e-6,
            step_size: 0.01,
            state: PhantomData,
            weights_: None,
            selected_features_: None,
            n_features_: None,
            objective_values_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the step size
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = step_size;
        self
    }

    /// Soft thresholding operator (proximal operator for L1 norm)
    fn soft_threshold(&self, x: Float, threshold: Float) -> Float {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Solve using proximal gradient method
    fn solve_proximal_gradient(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Vec<Float>)> {
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Initialize weights
        let mut weights = Array1::zeros(n_features);
        let mut objective_values = Vec::new();

        // Proximal gradient iterations
        for _iter in 0..self.max_iter {
            // Compute predictions
            let predictions = features.dot(&weights);

            // Compute residuals
            let residuals = &predictions - target;

            // Compute gradient of smooth part (data term)
            let gradient = features.t().dot(&residuals) / n_samples as Float;

            // Gradient step
            let temp_weights = &weights - self.step_size * &gradient;

            // Proximal step (soft thresholding for L1 regularization)
            let threshold = self.step_size * self.regularization;
            let new_weights = temp_weights.mapv(|w| self.soft_threshold(w, threshold));

            // Compute objective value
            let data_term = residuals.mapv(|r| r * r).sum() / (2.0 * n_samples as Float);
            let reg_term = self.regularization * weights.mapv(|w| w.abs()).sum();
            let objective = data_term + reg_term;
            objective_values.push(objective);

            // Check convergence
            let weight_diff = (&new_weights - &weights).mapv(|d| d.abs()).sum();
            if weight_diff < self.tolerance {
                break;
            }

            weights = new_weights;
        }

        Ok((weights, objective_values))
    }
}

impl Estimator for ProximalGradientSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ProximalGradientSelector<Untrained> {
    type Fitted = ProximalGradientSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Solve using proximal gradient method
        let (weights, objective_values) = self.solve_proximal_gradient(features, target)?;

        // Select top k features based on absolute weights
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            weights[b]
                .abs()
                .partial_cmp(&weights[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(ProximalGradientSelector {
            k: self.k,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            step_size: self.step_size,
            state: PhantomData,
            weights_: Some(weights),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            objective_values_: Some(objective_values),
        })
    }
}

impl Transform<Array2<Float>> for ProximalGradientSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ProximalGradientSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for ProximalGradientSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl ProximalGradientSelector<Trained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array1<Float> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the objective values during optimization
    pub fn objective_values(&self) -> &[Float] {
        self.objective_values_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Alternating Direction Method of Multipliers (ADMM) for feature selection
#[derive(Debug, Clone)]
pub struct ADMMFeatureSelector<State = Untrained> {
    k: usize,
    regularization: Float,
    max_iter: usize,
    tolerance: Float,
    rho: Float, // ADMM penalty parameter
    state: PhantomData<State>,
    // Trained state
    weights_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    objective_values_: Option<Vec<Float>>,
}

impl Default for ADMMFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl ADMMFeatureSelector<Untrained> {
    /// Create a new ADMM feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            regularization: 1.0,
            max_iter: 1000,
            tolerance: 1e-6,
            rho: 1.0,
            state: PhantomData,
            weights_: None,
            selected_features_: None,
            n_features_: None,
            objective_values_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the ADMM penalty parameter
    pub fn rho(mut self, rho: Float) -> Self {
        self.rho = rho;
        self
    }

    /// Soft thresholding operator
    fn soft_threshold(&self, x: Float, threshold: Float) -> Float {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    }

    /// Solve using ADMM
    fn solve_admm(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<Float>, Vec<Float>)> {
        let n_features = features.ncols();
        let n_samples = features.nrows();

        // Initialize variables
        let mut x = Array1::<Float>::zeros(n_features); // Primary variable
        let mut z = Array1::<Float>::zeros(n_features); // Auxiliary variable
        let mut u = Array1::<Float>::zeros(n_features); // Dual variable

        let mut objective_values = Vec::new();

        // Precompute matrices for efficiency
        let xtx = features.t().dot(features);
        let xty = features.t().dot(target);

        // ADMM iterations
        for _iter in 0..self.max_iter {
            let _x_old = x.clone();
            let z_old = z.clone();

            // x-update: solve quadratic subproblem
            // (X^T X + rho I) x = X^T y + rho (z - u)
            let rhs = &xty + self.rho * (&z - &u);

            // Simplified solve: assume diagonal dominance and use Jacobi iterations
            for i in 0..n_features {
                let diag_elem = xtx[[i, i]] + self.rho;
                if diag_elem > 1e-12 {
                    let off_diag = (0..n_features)
                        .filter(|&j| j != i)
                        .map(|j| xtx[[i, j]] * x[j])
                        .sum::<Float>();
                    x[i] = (rhs[i] - off_diag) / diag_elem;
                }
            }

            // z-update: soft thresholding
            let threshold = self.regularization / self.rho;
            for i in 0..n_features {
                z[i] = self.soft_threshold(x[i] + u[i], threshold);
            }

            // u-update: dual variable update
            u = &u + &x - &z;

            // Compute objective value
            let predictions = features.dot(&x);
            let residuals = &predictions - target;
            let data_term = residuals.mapv(|r| r * r).sum() / (2.0 * n_samples as Float);
            let reg_term = self.regularization * z.mapv(|z_i| z_i.abs()).sum();
            let objective = data_term + reg_term;
            objective_values.push(objective);

            // Check convergence
            let primal_residual = (&x - &z).mapv(|r| r.abs()).sum();
            let dual_residual = self.rho * (&z - &z_old).mapv(|r| r.abs()).sum();

            if primal_residual < self.tolerance && dual_residual < self.tolerance {
                break;
            }
        }

        Ok((z, objective_values))
    }
}

impl Estimator for ADMMFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ADMMFeatureSelector<Untrained> {
    type Fitted = ADMMFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Solve using ADMM
        let (weights, objective_values) = self.solve_admm(features, target)?;

        // Select top k features based on absolute weights
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            weights[b]
                .abs()
                .partial_cmp(&weights[a].abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(ADMMFeatureSelector {
            k: self.k,
            regularization: self.regularization,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            rho: self.rho,
            state: PhantomData,
            weights_: Some(weights),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            objective_values_: Some(objective_values),
        })
    }
}

impl Transform<Array2<Float>> for ADMMFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ADMMFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for ADMMFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl ADMMFeatureSelector<Trained> {
    /// Get the learned weights
    pub fn weights(&self) -> &Array1<Float> {
        self.weights_.as_ref().unwrap()
    }

    /// Get the objective values during optimization
    pub fn objective_values(&self) -> &[Float] {
        self.objective_values_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Semidefinite Programming (SDP) approach for feature selection
#[derive(Debug, Clone)]
pub struct SemidefiniteFeatureSelector<State = Untrained> {
    k: usize,
    max_iter: usize,
    tolerance: Float,
    regularization: Float,
    state: PhantomData<State>,
    // Trained state
    feature_matrix_: Option<Array2<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    eigenvalues_: Option<Array1<Float>>,
    objective_values_: Option<Vec<Float>>,
}

impl Default for SemidefiniteFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl SemidefiniteFeatureSelector<Untrained> {
    /// Create a new SDP feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            max_iter: 100,
            tolerance: 1e-6,
            regularization: 1.0,
            state: PhantomData,
            feature_matrix_: None,
            selected_features_: None,
            n_features_: None,
            eigenvalues_: None,
            objective_values_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Project matrix onto positive semidefinite cone
    fn project_psd(&self, matrix: &Array2<Float>) -> SklResult<Array2<Float>> {
        let n = matrix.nrows();

        // Simple symmetric projection for PSD constraint
        let mut projected = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                projected[[i, j]] = (matrix[[i, j]] + matrix[[j, i]]) / 2.0;
            }
        }

        // Zero out negative eigenvalues (simplified PSD projection)
        for i in 0..n {
            if projected[[i, i]] < 0.0 {
                projected[[i, i]] = 0.0;
            }
        }

        Ok(projected)
    }

    /// Solve SDP relaxation for feature selection
    fn solve_sdp_relaxation(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array2<Float>, Array1<Float>, Vec<Float>)> {
        let n_features = features.ncols();

        // Compute feature covariance matrix
        let centered_features = features - &features.mean_axis(Axis(0)).unwrap();
        let cov_matrix =
            centered_features.t().dot(&centered_features) / (features.nrows() - 1) as Float;

        // Compute feature-target correlations
        let target_centered = target - target.mean().unwrap();
        let correlations =
            centered_features.t().dot(&target_centered) / (features.nrows() - 1) as Float;

        // Initialize variable matrix X (relaxation of x*x^T where x is binary)
        let mut x_matrix = Array2::eye(n_features) * 0.5; // Start with diagonal matrix
        let mut objective_values = Vec::new();

        // Projected gradient method for SDP relaxation
        for _iter in 0..self.max_iter {
            let _x_old = x_matrix.clone();

            // Compute gradient
            // Objective: maximize correlations^T * X * correlations - regularization * trace(X * cov_matrix)
            let outer_corr = outer_product(&correlations, &correlations);
            let grad = &outer_corr - self.regularization * &cov_matrix;

            // Gradient step
            let step_size = 0.01;
            let x_new = &x_matrix + step_size * &grad;

            // Project onto constraints: PSD and diagonal constraints
            let mut x_projected = self.project_psd(&x_new)?;

            // Enforce constraint: 0 <= X_{ii} <= 1
            for i in 0..n_features {
                x_projected[[i, i]] = x_projected[[i, i]].clamp(0.0, 1.0);
            }

            // Compute objective value
            let obj = correlations.dot(&x_projected.dot(&correlations))
                - self.regularization * trace(&x_projected.dot(&cov_matrix));
            objective_values.push(obj);

            // Check convergence
            let diff = (&x_projected - &x_matrix).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                break;
            }

            x_matrix = x_projected;
        }

        // Extract eigenvalues from final matrix
        let eigenvalues = extract_diagonal(&x_matrix);

        Ok((x_matrix, eigenvalues, objective_values))
    }
}

impl Estimator for SemidefiniteFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for SemidefiniteFeatureSelector<Untrained> {
    type Fitted = SemidefiniteFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Solve SDP relaxation
        let (feature_matrix, eigenvalues, objective_values) =
            self.solve_sdp_relaxation(features, target)?;

        // Select top k features based on diagonal values (relaxed selection indicators)
        let mut feature_indices: Vec<usize> = (0..n_features).collect();
        feature_indices.sort_by(|&a, &b| {
            eigenvalues[b]
                .partial_cmp(&eigenvalues[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let selected_features = feature_indices.into_iter().take(self.k).collect();

        Ok(SemidefiniteFeatureSelector {
            k: self.k,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            regularization: self.regularization,
            state: PhantomData,
            feature_matrix_: Some(feature_matrix),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            eigenvalues_: Some(eigenvalues),
            objective_values_: Some(objective_values),
        })
    }
}

impl Transform<Array2<Float>> for SemidefiniteFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for SemidefiniteFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        let n_features = self.n_features_.unwrap();
        let selected_features = self.selected_features_.as_ref().unwrap();
        let mut support = Array1::from_elem(n_features, false);

        for &idx in selected_features {
            support[idx] = true;
        }

        Ok(support)
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for SemidefiniteFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl SemidefiniteFeatureSelector<Trained> {
    /// Get the feature matrix from SDP optimization
    pub fn feature_matrix(&self) -> &Array2<Float> {
        self.feature_matrix_.as_ref().unwrap()
    }

    /// Get the eigenvalues (selection indicators)
    pub fn eigenvalues(&self) -> &Array1<Float> {
        self.eigenvalues_.as_ref().unwrap()
    }

    /// Get the objective values during optimization
    pub fn objective_values(&self) -> &[Float] {
        self.objective_values_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

/// Integer Programming approach for feature selection
#[derive(Debug, Clone)]
pub struct IntegerProgrammingFeatureSelector<State = Untrained> {
    k: usize,
    max_iter: usize,
    tolerance: Float,
    greedy_init: bool,
    local_search: bool,
    state: PhantomData<State>,
    // Trained state
    binary_solution_: Option<Array1<bool>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
    objective_value_: Option<Float>,
    improvement_history_: Option<Vec<Float>>,
}

impl Default for IntegerProgrammingFeatureSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl IntegerProgrammingFeatureSelector<Untrained> {
    /// Create a new integer programming feature selector
    pub fn new() -> Self {
        Self {
            k: 10,
            max_iter: 1000,
            tolerance: 1e-6,
            greedy_init: true,
            local_search: true,
            state: PhantomData,
            binary_solution_: None,
            selected_features_: None,
            n_features_: None,
            objective_value_: None,
            improvement_history_: None,
        }
    }

    /// Set the number of features to select
    pub fn k(mut self, k: usize) -> Self {
        self.k = k;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the convergence tolerance
    pub fn tolerance(mut self, tolerance: Float) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Enable/disable greedy initialization
    pub fn greedy_init(mut self, greedy_init: bool) -> Self {
        self.greedy_init = greedy_init;
        self
    }

    /// Enable/disable local search
    pub fn local_search(mut self, local_search: bool) -> Self {
        self.local_search = local_search;
        self
    }

    /// Compute feature importance scores
    fn compute_feature_scores(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<Array1<Float>> {
        let n_features = features.ncols();
        let mut scores = Array1::zeros(n_features);

        // Compute correlation-based importance
        for i in 0..n_features {
            let feature_col = features.column(i);
            let correlation = correlation_coefficient(&feature_col.to_owned(), target)?;
            scores[i] = correlation.abs();
        }

        Ok(scores)
    }

    /// Greedy initialization for IP
    fn greedy_initialization(&self, scores: &Array1<Float>) -> Array1<bool> {
        let n_features = scores.len();
        let mut solution = Array1::from_elem(n_features, false);

        // Select top k features greedily
        let mut indices: Vec<usize> = (0..n_features).collect();
        indices.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &idx in indices.iter().take(self.k) {
            solution[idx] = true;
        }

        solution
    }

    /// Evaluate objective function for binary solution
    fn evaluate_objective(&self, solution: &Array1<bool>, scores: &Array1<Float>) -> Float {
        let mut objective = 0.0;
        let mut selected_count = 0;

        for i in 0..solution.len() {
            if solution[i] {
                objective += scores[i];
                selected_count += 1;
            }
        }

        // Penalty for violating cardinality constraint
        if selected_count != self.k {
            objective -= 1000.0 * (selected_count as Float - self.k as Float).abs();
        }

        objective
    }

    /// Local search improvement (1-opt and 2-opt moves)
    fn local_search_improvement(
        &self,
        solution: &mut Array1<bool>,
        scores: &Array1<Float>,
        best_obj: &mut Float,
    ) -> bool {
        let n_features = solution.len();
        let mut improved = false;

        // 1-opt moves: flip single features
        for i in 0..n_features {
            let original = solution[i];
            solution[i] = !solution[i];

            let new_obj = self.evaluate_objective(solution, scores);
            if new_obj > *best_obj + self.tolerance {
                *best_obj = new_obj;
                improved = true;
            } else {
                solution[i] = original; // Revert if no improvement
            }
        }

        // 2-opt moves: swap pairs of features
        if self.local_search {
            for i in 0..n_features {
                for j in (i + 1)..n_features {
                    if solution[i] == solution[j] {
                        continue; // Skip if both have same value
                    }

                    // Swap
                    let temp = solution[i];
                    solution[i] = solution[j];
                    solution[j] = temp;

                    let new_obj = self.evaluate_objective(solution, scores);
                    if new_obj > *best_obj + self.tolerance {
                        *best_obj = new_obj;
                        improved = true;
                    } else {
                        // Revert swap
                        let temp = solution[i];
                        solution[i] = solution[j];
                        solution[j] = temp;
                    }
                }
            }
        }

        improved
    }

    /// Solve integer programming problem approximately
    fn solve_integer_programming(
        &self,
        features: &Array2<Float>,
        target: &Array1<Float>,
    ) -> SklResult<(Array1<bool>, Float, Vec<Float>)> {
        let scores = self.compute_feature_scores(features, target)?;
        let mut improvement_history = Vec::new();

        // Initialize solution
        let mut solution = if self.greedy_init {
            self.greedy_initialization(&scores)
        } else {
            // Random initialization
            let mut random_solution = Array1::from_elem(scores.len(), false);
            let indices: Vec<usize> = (0..scores.len()).collect();
            for &idx in indices.iter().take(self.k) {
                random_solution[idx] = true;
            }
            random_solution
        };

        let mut best_objective = self.evaluate_objective(&solution, &scores);
        improvement_history.push(best_objective);

        // Iterative improvement
        for _iter in 0..self.max_iter {
            let prev_objective = best_objective;

            let improved =
                self.local_search_improvement(&mut solution, &scores, &mut best_objective);
            improvement_history.push(best_objective);

            if !improved || (best_objective - prev_objective).abs() < self.tolerance {
                break;
            }
        }

        Ok((solution, best_objective, improvement_history))
    }
}

impl Estimator for IntegerProgrammingFeatureSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for IntegerProgrammingFeatureSelector<Untrained> {
    type Fitted = IntegerProgrammingFeatureSelector<Trained>;

    fn fit(self, features: &Array2<Float>, target: &Array1<Float>) -> SklResult<Self::Fitted> {
        let n_features = features.ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "No features provided".to_string(),
            ));
        }

        if self.k > n_features {
            return Err(SklearsError::InvalidInput(format!(
                "k ({}) cannot be greater than number of features ({})",
                self.k, n_features
            )));
        }

        // Solve integer programming problem
        let (binary_solution, objective_value, improvement_history) =
            self.solve_integer_programming(features, target)?;

        // Extract selected features
        let selected_features: Vec<usize> = binary_solution
            .iter()
            .enumerate()
            .filter_map(|(i, &selected)| if selected { Some(i) } else { None })
            .collect();

        Ok(IntegerProgrammingFeatureSelector {
            k: self.k,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            greedy_init: self.greedy_init,
            local_search: self.local_search,
            state: PhantomData,
            binary_solution_: Some(binary_solution),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
            objective_value_: Some(objective_value),
            improvement_history_: Some(improvement_history),
        })
    }
}

impl Transform<Array2<Float>> for IntegerProgrammingFeatureSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let n_selected = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, n_selected));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for IntegerProgrammingFeatureSelector<Trained> {
    fn get_support(&self) -> SklResult<Array1<bool>> {
        Ok(self.binary_solution_.as_ref().unwrap().clone())
    }

    fn transform_features(&self, indices: &[usize]) -> SklResult<Vec<usize>> {
        let selected_features = self.selected_features_.as_ref().unwrap();
        Ok(indices
            .iter()
            .filter_map(|&idx| selected_features.iter().position(|&f| f == idx))
            .collect())
    }
}

impl FeatureSelector for IntegerProgrammingFeatureSelector<Trained> {
    fn selected_features(&self) -> &Vec<usize> {
        self.selected_features_.as_ref().unwrap()
    }
}

impl IntegerProgrammingFeatureSelector<Trained> {
    /// Get the binary solution
    pub fn binary_solution(&self) -> &Array1<bool> {
        self.binary_solution_.as_ref().unwrap()
    }

    /// Get the final objective value
    pub fn objective_value(&self) -> Float {
        self.objective_value_.unwrap()
    }

    /// Get the improvement history during optimization
    pub fn improvement_history(&self) -> &[Float] {
        self.improvement_history_.as_ref().unwrap()
    }

    /// Get the number of selected features
    pub fn n_features_out(&self) -> usize {
        self.selected_features_.as_ref().unwrap().len()
    }
}

// Helper functions
fn outer_product(a: &Array1<Float>, b: &Array1<Float>) -> Array2<Float> {
    let mut result = Array2::zeros((a.len(), b.len()));
    for i in 0..a.len() {
        for j in 0..b.len() {
            result[[i, j]] = a[i] * b[j];
        }
    }
    result
}

fn trace(matrix: &Array2<Float>) -> Float {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|i| matrix[[i, i]]).sum()
}

fn extract_diagonal(matrix: &Array2<Float>) -> Array1<Float> {
    let n = matrix.nrows().min(matrix.ncols());
    let mut diag = Array1::zeros(n);
    for i in 0..n {
        diag[i] = matrix[[i, i]];
    }
    diag
}

fn correlation_coefficient(x: &Array1<Float>, y: &Array1<Float>) -> SklResult<Float> {
    if x.len() != y.len() {
        return Err(SklearsError::InvalidInput(
            "Arrays must have the same length".to_string(),
        ));
    }

    let _n = x.len() as Float;
    let mean_x = x.mean().unwrap();
    let mean_y = y.mean().unwrap();

    let mut num = 0.0;
    let mut den_x = 0.0;
    let mut den_y = 0.0;

    for i in 0..x.len() {
        let diff_x = x[i] - mean_x;
        let diff_y = y[i] - mean_y;
        num += diff_x * diff_y;
        den_x += diff_x * diff_x;
        den_y += diff_y * diff_y;
    }

    if den_x.abs() < 1e-10 || den_y.abs() < 1e-10 {
        return Ok(0.0);
    }

    Ok(num / (den_x * den_y).sqrt())
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::Array2;

    fn create_test_data() -> (Array2<Float>, Array1<Float>) {
        // Create synthetic data with some correlation structure
        let n_samples = 50;
        let n_features = 10;
        let mut features = Array2::zeros((n_samples, n_features));
        let mut target = Array1::zeros(n_samples);

        // Fill with structured data
        for i in 0..n_samples {
            for j in 0..n_features {
                features[[i, j]] = (i as Float * 0.1 + j as Float * 0.01).sin() + 0.1 * j as Float;
            }
            // Make first few features predictive
            target[i] = features[[i, 0]] + 0.5 * features[[i, 1]] + 0.1 * features[[i, 2]];
        }

        (features, target)
    }

    #[test]
    fn test_convex_feature_selector() {
        let (features, target) = create_test_data();

        let selector = ConvexFeatureSelector::new()
            .k(5)
            .regularization(0.1)
            .max_iter(100);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 5);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 5);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test weights
        let weights = trained.weights();
        assert_eq!(weights.len(), features.ncols());
        assert!(weights.iter().all(|&x| x.is_finite()));

        // Test objective values
        let obj_vals = trained.objective_values();
        assert!(!obj_vals.is_empty());
        assert!(obj_vals.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_proximal_gradient_selector() {
        let (features, target) = create_test_data();

        let selector = ProximalGradientSelector::new()
            .k(4)
            .regularization(0.1)
            .step_size(0.01)
            .max_iter(100);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 4);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test weights
        let weights = trained.weights();
        assert_eq!(weights.len(), features.ncols());
        assert!(weights.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_admm_feature_selector() {
        let (features, target) = create_test_data();

        let selector = ADMMFeatureSelector::new()
            .k(3)
            .regularization(0.1)
            .rho(1.0)
            .max_iter(50);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 3);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test weights
        let weights = trained.weights();
        assert_eq!(weights.len(), features.ncols());
        assert!(weights.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_convex_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = ConvexFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_proximal_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = ProximalGradientSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_admm_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = ADMMFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_semidefinite_feature_selector() {
        let (features, target) = create_test_data();

        let selector = SemidefiniteFeatureSelector::new()
            .k(4)
            .regularization(0.1)
            .max_iter(50);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 4);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 4);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test feature matrix
        let feature_matrix = trained.feature_matrix();
        assert_eq!(feature_matrix.nrows(), features.ncols());
        assert_eq!(feature_matrix.ncols(), features.ncols());

        // Test eigenvalues
        let eigenvalues = trained.eigenvalues();
        assert_eq!(eigenvalues.len(), features.ncols());
        assert!(eigenvalues.iter().all(|&x| x.is_finite()));

        // Test objective values
        let obj_vals = trained.objective_values();
        assert!(!obj_vals.is_empty());
        assert!(obj_vals.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_integer_programming_feature_selector() {
        let (features, target) = create_test_data();

        let selector = IntegerProgrammingFeatureSelector::new()
            .k(3)
            .greedy_init(true)
            .local_search(true)
            .max_iter(100);

        let trained = selector.fit(&features, &target).unwrap();
        assert_eq!(trained.n_features_out(), 3);

        // Test transform
        let transformed = trained.transform(&features).unwrap();
        assert_eq!(transformed.ncols(), 3);
        assert_eq!(transformed.nrows(), features.nrows());

        // Test binary solution
        let binary_solution = trained.binary_solution();
        assert_eq!(binary_solution.len(), features.ncols());
        let selected_count = binary_solution.iter().filter(|&&x| x).count();
        assert_eq!(selected_count, 3);

        // Test objective value
        let obj_value = trained.objective_value();
        assert!(obj_value.is_finite());

        // Test improvement history
        let improvement_history = trained.improvement_history();
        assert!(!improvement_history.is_empty());
        assert!(improvement_history.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_semidefinite_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = SemidefiniteFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_integer_programming_selector_invalid_k() {
        let (features, target) = create_test_data();

        let selector = IntegerProgrammingFeatureSelector::new().k(features.ncols() + 1);
        assert!(selector.fit(&features, &target).is_err());
    }

    #[test]
    fn test_correlation_coefficient() {
        let x = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);
        let y = Array1::from_vec(vec![2.0, 4.0, 6.0, 8.0, 10.0]);

        let corr = correlation_coefficient(&x, &y).unwrap();
        assert!((corr - 1.0).abs() < 1e-10); // Perfect correlation

        let z = Array1::from_vec(vec![5.0, 4.0, 3.0, 2.0, 1.0]);
        let corr2 = correlation_coefficient(&x, &z).unwrap();
        assert!((corr2 + 1.0).abs() < 1e-10); // Perfect negative correlation
    }

    #[test]
    fn test_helper_functions() {
        let a = Array1::from_vec(vec![1.0, 2.0]);
        let b = Array1::from_vec(vec![3.0, 4.0]);

        let outer = outer_product(&a, &b);
        assert_eq!(outer[[0, 0]], 3.0);
        assert_eq!(outer[[0, 1]], 4.0);
        assert_eq!(outer[[1, 0]], 6.0);
        assert_eq!(outer[[1, 1]], 8.0);

        let matrix = Array2::from_shape_vec((2, 2), vec![1.0, 2.0, 3.0, 4.0]).unwrap();
        let tr = trace(&matrix);
        assert_eq!(tr, 5.0); // 1 + 4

        let diag = extract_diagonal(&matrix);
        assert_eq!(diag[0], 1.0);
        assert_eq!(diag[1], 4.0);
    }
}
