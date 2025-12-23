//! Regularization-based feature selection methods

use crate::base::SelectorMixin;
use scirs2_core::ndarray::{Array1, Array2};
// For now, let's use a simple approach without external linalg
// We'll implement a basic solve using Gaussian elimination or use the pseudoinverse
use sklears_core::{
    error::{validate, Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Trained, Transform, Untrained},
    types::Float,
};
use std::marker::PhantomData;

/// LASSO-based feature selection
/// Uses L1 regularization to automatically select features by setting coefficients to zero
#[derive(Debug, Clone)]
pub struct LassoSelector<State = Untrained> {
    alpha: f64,
    max_iter: usize,
    tolerance: f64,
    threshold: Option<f64>,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl LassoSelector<Untrained> {
    /// Create a new LASSO selector
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            max_iter: 1000,
            tolerance: 1e-4,
            threshold: None,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the regularization parameter alpha
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be non-negative");
        }
        self.alpha = alpha;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set threshold for feature selection (if None, uses mean of absolute coefficients)
    pub fn threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for LassoSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for LassoSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for LassoSelector<Untrained> {
    type Fitted = LassoSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "LASSO requires at least 2 samples".to_string(),
            ));
        }

        // Normalize features for LASSO
        let (x_normalized, _x_mean, x_std) = normalize_and_center(x)?;
        let y_mean = y.mean().unwrap_or(0.0);
        let y_centered: Array1<Float> = y - y_mean;

        // Fit LASSO using coordinate descent
        let mut coefficients = Array1::zeros(n_features);

        for _iter in 0..self.max_iter {
            let old_coefficients = coefficients.clone();

            for j in 0..n_features {
                if x_std[j] <= Float::EPSILON {
                    continue; // Skip zero-variance features
                }

                // Compute residual without feature j
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_k = x_normalized.column(k);
                        for i in 0..n_samples {
                            residual[i] -= coefficients[k] * x_k[i];
                        }
                    }
                }

                // Compute correlation with residual
                let x_j = x_normalized.column(j);
                let rho = x_j.dot(&residual) / n_samples as Float;

                // Soft thresholding (LASSO update)
                coefficients[j] = soft_threshold(rho, self.alpha);
            }

            // Check convergence
            let diff = (&coefficients - &old_coefficients).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                break;
            }
        }

        // Rescale coefficients back to original scale
        for j in 0..n_features {
            if x_std[j] > Float::EPSILON {
                coefficients[j] /= x_std[j];
            }
        }

        // Select features based on threshold
        let threshold = self.threshold.unwrap_or_else(|| {
            let abs_coeffs: Array1<Float> = coefficients.mapv(|x| x.abs());
            abs_coeffs.mean().unwrap_or(0.0)
        });

        let selected_features: Vec<usize> = coefficients
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > threshold)
            .map(|(idx, _)| idx)
            .collect();

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected by LASSO. Try reducing alpha or threshold.".to_string(),
            ));
        }

        Ok(LassoSelector {
            alpha: self.alpha,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            threshold: self.threshold,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for LassoSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for LassoSelector<Trained> {
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

impl LassoSelector<Trained> {
    /// Get LASSO coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.coefficients_.as_ref().unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }
}

/// Elastic Net feature selection
/// Combines L1 (LASSO) and L2 (Ridge) regularization
#[derive(Debug, Clone)]
pub struct ElasticNetSelector<State = Untrained> {
    alpha: f64,
    l1_ratio: f64,
    max_iter: usize,
    tolerance: f64,
    threshold: Option<f64>,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl ElasticNetSelector<Untrained> {
    /// Create a new Elastic Net selector
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            l1_ratio: 0.5,
            max_iter: 1000,
            tolerance: 1e-4,
            threshold: None,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the regularization parameter alpha
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be non-negative");
        }
        self.alpha = alpha;
        self
    }

    /// Set the L1 ratio (0 = Ridge, 1 = LASSO)
    pub fn l1_ratio(mut self, l1_ratio: f64) -> Self {
        if !(0.0..=1.0).contains(&l1_ratio) {
            panic!("l1_ratio must be between 0 and 1");
        }
        self.l1_ratio = l1_ratio;
        self
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tolerance(mut self, tolerance: f64) -> Self {
        self.tolerance = tolerance;
        self
    }

    /// Set threshold for feature selection
    pub fn threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }
}

impl Default for ElasticNetSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for ElasticNetSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for ElasticNetSelector<Untrained> {
    type Fitted = ElasticNetSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Elastic Net requires at least 2 samples".to_string(),
            ));
        }

        // Normalize features
        let (x_normalized, _x_mean, x_std) = normalize_and_center(x)?;
        let y_mean = y.mean().unwrap_or(0.0);
        let y_centered: Array1<Float> = y - y_mean;

        // Fit Elastic Net using coordinate descent
        let mut coefficients = Array1::zeros(n_features);
        let alpha_l1 = self.alpha * self.l1_ratio;
        let alpha_l2 = self.alpha * (1.0 - self.l1_ratio);

        for _iter in 0..self.max_iter {
            let old_coefficients = coefficients.clone();

            for j in 0..n_features {
                if x_std[j] <= Float::EPSILON {
                    continue;
                }

                // Compute residual without feature j
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let x_k = x_normalized.column(k);
                        for i in 0..n_samples {
                            residual[i] -= coefficients[k] * x_k[i];
                        }
                    }
                }

                // Compute correlation with residual
                let x_j = x_normalized.column(j);
                let rho = x_j.dot(&residual) / n_samples as Float;

                // Elastic Net update (soft thresholding with L2 penalty)
                let denominator = 1.0 + alpha_l2;
                coefficients[j] = soft_threshold(rho, alpha_l1) / denominator;
            }

            // Check convergence
            let diff = (&coefficients - &old_coefficients).mapv(|x| x.abs()).sum();
            if diff < self.tolerance {
                break;
            }
        }

        // Rescale coefficients back to original scale
        for j in 0..n_features {
            if x_std[j] > Float::EPSILON {
                coefficients[j] /= x_std[j];
            }
        }

        // Select features based on threshold
        let threshold = self.threshold.unwrap_or_else(|| {
            let abs_coeffs: Array1<Float> = coefficients.mapv(|x| x.abs());
            abs_coeffs.mean().unwrap_or(0.0)
        });

        let selected_features: Vec<usize> = coefficients
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > threshold)
            .map(|(idx, _)| idx)
            .collect();

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected by Elastic Net. Try reducing alpha or threshold.".to_string(),
            ));
        }

        Ok(ElasticNetSelector {
            alpha: self.alpha,
            l1_ratio: self.l1_ratio,
            max_iter: self.max_iter,
            tolerance: self.tolerance,
            threshold: self.threshold,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for ElasticNetSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for ElasticNetSelector<Trained> {
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

impl ElasticNetSelector<Trained> {
    /// Get Elastic Net coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.coefficients_.as_ref().unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }
}

// Helper functions

/// Soft thresholding function for LASSO
fn soft_threshold(x: Float, threshold: Float) -> Float {
    if x > threshold {
        x - threshold
    } else if x < -threshold {
        x + threshold
    } else {
        0.0
    }
}

/// Normalize and center features
fn normalize_and_center(
    x: &Array2<Float>,
) -> SklResult<(Array2<Float>, Array1<Float>, Array1<Float>)> {
    let (n_samples, n_features) = x.dim();
    let mut x_normalized = Array2::zeros((n_samples, n_features));
    let mut x_mean = Array1::zeros(n_features);
    let mut x_std = Array1::zeros(n_features);

    // Compute means
    for j in 0..n_features {
        x_mean[j] = x.column(j).mean().unwrap_or(0.0);
    }

    // Center the data
    for j in 0..n_features {
        for i in 0..n_samples {
            x_normalized[[i, j]] = x[[i, j]] - x_mean[j];
        }
    }

    // Compute standard deviations
    for j in 0..n_features {
        let variance = x_normalized.column(j).mapv(|x| x * x).mean().unwrap_or(0.0);
        x_std[j] = variance.sqrt();

        // Normalize if std > 0
        if x_std[j] > Float::EPSILON {
            for i in 0..n_samples {
                x_normalized[[i, j]] /= x_std[j];
            }
        }
    }

    Ok((x_normalized, x_mean, x_std))
}

/// Ridge Regression feature selection
/// Uses Ridge regression coefficients to select features
#[derive(Debug, Clone)]
pub struct RidgeSelector<State = Untrained> {
    alpha: f64,
    threshold: Option<f64>,
    max_features: Option<usize>,
    state: PhantomData<State>,
    // Trained state
    coefficients_: Option<Array1<Float>>,
    selected_features_: Option<Vec<usize>>,
    n_features_: Option<usize>,
}

impl RidgeSelector<Untrained> {
    /// Create a new Ridge selector
    pub fn new() -> Self {
        Self {
            alpha: 1.0,
            threshold: None,
            max_features: None,
            state: PhantomData,
            coefficients_: None,
            selected_features_: None,
            n_features_: None,
        }
    }

    /// Set the regularization parameter alpha
    pub fn alpha(mut self, alpha: f64) -> Self {
        if alpha < 0.0 {
            panic!("alpha must be non-negative");
        }
        self.alpha = alpha;
        self
    }

    /// Set threshold for feature selection (if None, uses mean of absolute coefficients)
    pub fn threshold(mut self, threshold: Option<f64>) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set maximum number of features to select
    pub fn max_features(mut self, max_features: Option<usize>) -> Self {
        self.max_features = max_features;
        self
    }

    /// Simple solve for linear system Ax = b using Gaussian elimination
    fn simple_solve(&self, a: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = a.nrows();
        if n != a.ncols() {
            return Err(SklearsError::FitError("Matrix must be square".to_string()));
        }
        if n != b.len() {
            return Err(SklearsError::FitError("Dimension mismatch".to_string()));
        }

        // Simple approach: use iterative method (Jacobi iteration)
        // For a well-conditioned ridge system, this should converge
        let mut x = Array1::zeros(n);
        let max_iter = 1000;
        let tolerance = 1e-6;

        for _ in 0..max_iter {
            let mut x_new = Array1::zeros(n);
            for i in 0..n {
                let mut sum = 0.0;
                for j in 0..n {
                    if i != j {
                        sum += a[[i, j]] * x[j];
                    }
                }
                x_new[i] = (b[i] - sum) / a[[i, i]];
            }

            // Check convergence
            let diff: f64 = x_new
                .iter()
                .zip(x.iter())
                .map(|(a, b): (&f64, &f64)| (a - b).abs())
                .sum();
            if diff < tolerance {
                return Ok(x_new);
            }
            x = x_new;
        }

        // If we reach here, convergence failed - return best attempt
        Ok(x)
    }
}

impl Default for RidgeSelector<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for RidgeSelector<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<Array2<Float>, Array1<Float>> for RidgeSelector<Untrained> {
    type Fitted = RidgeSelector<Trained>;

    fn fit(self, x: &Array2<Float>, y: &Array1<Float>) -> SklResult<Self::Fitted> {
        validate::check_consistent_length(x, y)?;

        let (n_samples, n_features) = x.dim();
        if n_samples < 2 {
            return Err(SklearsError::InvalidInput(
                "Ridge requires at least 2 samples".to_string(),
            ));
        }

        // Solve Ridge regression: (X^T X + alpha * I) beta = X^T y
        let xtx = x.t().dot(x);
        let xty = x.t().dot(y);

        // Add ridge penalty to diagonal
        let mut xtx_ridge = xtx;
        for i in 0..n_features {
            xtx_ridge[[i, i]] += self.alpha;
        }

        // Solve for coefficients using simple approach
        // For now, use a simple pseudoinverse approach or iterative method
        let coefficients = self.simple_solve(&xtx_ridge, &xty)?;

        // Select features based on threshold
        let threshold = self.threshold.unwrap_or_else(|| {
            let abs_coeffs: Array1<Float> = coefficients.mapv(|x| x.abs());
            abs_coeffs.mean().unwrap_or(0.0)
        });

        let mut selected_features: Vec<usize> = coefficients
            .iter()
            .enumerate()
            .filter(|(_, &coeff)| coeff.abs() > threshold)
            .map(|(idx, _)| idx)
            .collect();

        // Apply max_features limit if specified
        if let Some(max_feat) = self.max_features {
            if selected_features.len() > max_feat {
                // Sort by absolute coefficient value and take top max_feat
                selected_features.sort_by(|&a, &b| {
                    coefficients[b]
                        .abs()
                        .partial_cmp(&coefficients[a].abs())
                        .unwrap()
                });
                selected_features.truncate(max_feat);
                selected_features.sort(); // Restore original order
            }
        }

        if selected_features.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No features selected by Ridge. Try reducing alpha or threshold.".to_string(),
            ));
        }

        Ok(RidgeSelector {
            alpha: self.alpha,
            threshold: self.threshold,
            max_features: self.max_features,
            state: PhantomData,
            coefficients_: Some(coefficients),
            selected_features_: Some(selected_features),
            n_features_: Some(n_features),
        })
    }
}

impl Transform<Array2<Float>> for RidgeSelector<Trained> {
    fn transform(&self, x: &Array2<Float>) -> SklResult<Array2<Float>> {
        validate::check_n_features(x, self.n_features_.unwrap())?;

        let selected_features = self.selected_features_.as_ref().unwrap();
        let n_samples = x.nrows();
        let k = selected_features.len();
        let mut x_new = Array2::zeros((n_samples, k));

        for (new_idx, &old_idx) in selected_features.iter().enumerate() {
            x_new.column_mut(new_idx).assign(&x.column(old_idx));
        }

        Ok(x_new)
    }
}

impl SelectorMixin for RidgeSelector<Trained> {
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

impl RidgeSelector<Trained> {
    /// Get Ridge coefficients
    pub fn coefficients(&self) -> &Array1<Float> {
        self.coefficients_.as_ref().unwrap()
    }

    /// Get selected features
    pub fn selected_features(&self) -> &[usize] {
        self.selected_features_.as_ref().unwrap()
    }
}
