//! Regularization techniques for cross-decomposition methods
//!
//! This module provides various regularization methods including elastic net,
//! group lasso, fused lasso, and other penalty functions that can be applied
//! to cross-decomposition algorithms.

use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result, SklearsError},
    types::Float,
};

/// Elastic Net regularization combining L1 and L2 penalties
///
/// The elastic net penalty is: alpha * (l1_ratio * |w|_1 + (1 - l1_ratio) * |w|_2^2)
/// where alpha controls overall regularization strength and l1_ratio controls
/// the balance between L1 and L2 penalties.
#[derive(Debug, Clone)]
pub struct ElasticNet {
    /// Regularization strength (alpha)
    pub alpha: Float,
    /// L1 ratio (0.0 = pure L2, 1.0 = pure L1)
    pub l1_ratio: Float,
    /// Maximum number of iterations for coordinate descent
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Whether to normalize features
    pub normalize: bool,
    /// Positive constraint on coefficients
    pub positive: bool,
    /// Random state for reproducibility
    pub random_state: Option<u64>,
}

impl ElasticNet {
    /// Create a new ElasticNet regularizer
    pub fn new(alpha: Float, l1_ratio: Float) -> Self {
        Self {
            alpha,
            l1_ratio: l1_ratio.clamp(0.0, 1.0),
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            normalize: false,
            positive: false,
            random_state: None,
        }
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set whether to normalize features
    pub fn normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Set positive constraint
    pub fn positive(mut self, positive: bool) -> Self {
        self.positive = positive;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit elastic net to data using coordinate descent
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "X and y have incompatible shapes: {} vs {}",
                n_samples,
                y.len()
            )));
        }

        // Normalize features if requested
        let (X_norm, feature_means, feature_stds) = if self.normalize {
            self.normalize_features(X)?
        } else {
            (
                X.clone(),
                Array1::zeros(n_features),
                Array1::ones(n_features),
            )
        };

        // Center target if fitting intercept
        let (y_centered, y_mean) = if self.fit_intercept {
            let mean = y.mean().unwrap();
            (y - mean, mean)
        } else {
            (y.clone(), 0.0)
        };

        // Initialize coefficients
        let mut coef = Array1::zeros(n_features);

        // Coordinate descent algorithm
        for iter in 0..self.max_iter {
            let old_coef = coef.clone();

            for j in 0..n_features {
                // Compute residual without j-th feature
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let col_k = X_norm.column(k);
                        residual = residual - coef[k] * &col_k;
                    }
                }

                // Compute correlation with j-th feature
                let col_j = X_norm.column(j);
                let rho = col_j.dot(&residual) / n_samples as Float;

                // Soft thresholding for L1 penalty
                let l1_penalty = self.alpha * self.l1_ratio;
                let l2_penalty = self.alpha * (1.0 - self.l1_ratio);

                // Denominator includes L2 penalty
                let denominator = col_j.dot(&col_j) / n_samples as Float + l2_penalty;

                let new_coef = if rho > l1_penalty {
                    (rho - l1_penalty) / denominator
                } else if rho < -l1_penalty {
                    (rho + l1_penalty) / denominator
                } else {
                    0.0
                };

                // Apply positive constraint if needed
                coef[j] = if self.positive {
                    new_coef.max(0.0)
                } else {
                    new_coef
                };
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        // Rescale coefficients if features were normalized
        if self.normalize {
            for j in 0..n_features {
                if feature_stds[j] > 1e-8 {
                    coef[j] /= feature_stds[j];
                }
            }
        }

        Ok(coef)
    }

    /// Normalize features to zero mean and unit variance
    fn normalize_features(
        &self,
        X: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array1<Float>, Array1<Float>)> {
        let means = X.mean_axis(Axis(0)).unwrap();
        let centered = X - &means.view().insert_axis(Axis(0));
        let stds = centered.var_axis(Axis(0), 1.0).mapv(|x| x.sqrt());

        let mut normalized = centered;
        for (j, &std) in stds.iter().enumerate() {
            if std > 1e-8 {
                normalized.column_mut(j).mapv_inplace(|x| x / std);
            }
        }

        Ok((normalized, means, stds))
    }

    /// Compute elastic net penalty value
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        let l1_penalty = self.l1_ratio * coef.mapv(|x| x.abs()).sum();
        let l2_penalty = (1.0 - self.l1_ratio) * 0.5 * coef.mapv(|x| x * x).sum();
        self.alpha * (l1_penalty + l2_penalty)
    }

    /// Compute elastic net regularization path for different alpha values
    pub fn path(
        &self,
        X: &Array2<Float>,
        y: &Array1<Float>,
        alphas: &Array1<Float>,
    ) -> Result<Array2<Float>> {
        let n_features = X.ncols();
        let n_alphas = alphas.len();
        let mut coef_path = Array2::zeros((n_features, n_alphas));

        for (alpha_idx, &alpha) in alphas.iter().enumerate() {
            let mut elastic_net = self.clone();
            elastic_net.alpha = alpha;

            let coef = elastic_net.fit(X, y)?;
            coef_path.column_mut(alpha_idx).assign(&coef);
        }

        Ok(coef_path)
    }
}

impl Default for ElasticNet {
    fn default() -> Self {
        Self::new(1.0, 0.5)
    }
}

/// Group Lasso regularization for structured sparsity
///
/// Group lasso applies L2 penalty within groups and L1 penalty between groups,
/// encouraging entire groups of features to be zeroed out together.
#[derive(Debug, Clone)]
pub struct GroupLasso {
    /// Regularization strength
    pub alpha: Float,
    /// Groups of features (indices for each group)
    pub groups: Vec<Vec<usize>>,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
}

impl GroupLasso {
    /// Create a new GroupLasso regularizer
    pub fn new(alpha: Float, groups: Vec<Vec<usize>>) -> Self {
        Self {
            alpha,
            groups,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
        }
    }

    /// Fit group lasso using block coordinate descent
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();
        let mut coef = Array1::zeros(n_features);

        // Validate groups
        let mut all_indices = Vec::new();
        for group in &self.groups {
            for &idx in group {
                if idx >= n_features {
                    return Err(SklearsError::InvalidInput(format!(
                        "Group index {} exceeds number of features {}",
                        idx, n_features
                    )));
                }
                all_indices.push(idx);
            }
        }
        all_indices.sort_unstable();
        all_indices.dedup();

        if all_indices.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Groups must cover all features exactly once".to_string(),
            ));
        }

        // Center target if fitting intercept
        let y_centered = if self.fit_intercept {
            let mean = y.mean().unwrap();
            y - mean
        } else {
            y.clone()
        };

        // Block coordinate descent
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            for group in &self.groups {
                if group.is_empty() {
                    continue;
                }

                // Extract group features
                let mut X_group = Array2::zeros((n_samples, group.len()));
                for (g_idx, &feat_idx) in group.iter().enumerate() {
                    X_group.column_mut(g_idx).assign(&X.column(feat_idx));
                }

                // Compute residual without current group
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if !group.contains(&k) {
                        residual = residual - coef[k] * &X.column(k);
                    }
                }

                // Compute group gradient
                let gradient = X_group.t().dot(&residual) / n_samples as Float;
                let gradient_norm = (gradient.dot(&gradient)).sqrt();

                // Group soft thresholding
                let group_penalty = self.alpha * (group.len() as Float).sqrt();

                if gradient_norm <= group_penalty {
                    // Zero out entire group
                    for &feat_idx in group {
                        coef[feat_idx] = 0.0;
                    }
                } else {
                    // Shrink group coefficients
                    let shrinkage = 1.0 - group_penalty / gradient_norm;

                    // Solve within-group problem (L2 regularized least squares)
                    let XtX = X_group.t().dot(&X_group) / n_samples as Float;
                    let group_coef = self.solve_within_group(&XtX, &gradient, shrinkage)?;

                    for (g_idx, &feat_idx) in group.iter().enumerate() {
                        coef[feat_idx] = group_coef[g_idx];
                    }
                }
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        Ok(coef)
    }

    /// Solve within-group optimization problem
    fn solve_within_group(
        &self,
        XtX: &Array2<Float>,
        gradient: &Array1<Float>,
        shrinkage: Float,
    ) -> Result<Array1<Float>> {
        // Simple solution: use gradient with shrinkage
        // In practice, this should solve the regularized least squares problem
        Ok(shrinkage * gradient)
    }

    /// Compute group lasso penalty
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        let mut penalty = 0.0;

        for group in &self.groups {
            let mut group_norm_sq = 0.0;
            for &idx in group {
                group_norm_sq += coef[idx] * coef[idx];
            }
            penalty += (group.len() as Float).sqrt() * group_norm_sq.sqrt();
        }

        self.alpha * penalty
    }
}

/// Fused Lasso regularization for sequential data
///
/// Fused lasso adds penalty on differences between adjacent coefficients,
/// encouraging piecewise constant solutions.
#[derive(Debug, Clone)]
pub struct FusedLasso {
    /// L1 penalty on coefficients
    pub alpha1: Float,
    /// L1 penalty on differences between adjacent coefficients
    pub alpha2: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
}

impl FusedLasso {
    /// Create a new FusedLasso regularizer
    pub fn new(alpha1: Float, alpha2: Float) -> Self {
        Self {
            alpha1,
            alpha2,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Fit fused lasso using proximal gradient method
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();
        let mut coef = Array1::zeros(n_features);

        // Learning rate (should be < 1 / largest eigenvalue of X^T X)
        let step_size = 0.001;

        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            // Compute gradient of least squares loss
            let residual = y - &X.dot(&coef);
            let gradient = -X.t().dot(&residual) / n_samples as Float;

            // Gradient step
            let mut updated_coef = &coef - step_size * &gradient;

            // Proximal operator for fused lasso penalty
            updated_coef = self.proximal_operator(&updated_coef, step_size)?;

            coef = updated_coef;

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        Ok(coef)
    }

    /// Proximal operator for fused lasso penalty
    fn proximal_operator(&self, coef: &Array1<Float>, step_size: Float) -> Result<Array1<Float>> {
        let n = coef.len();
        let mut result = coef.clone();

        // Apply soft thresholding for L1 penalty on coefficients
        let l1_threshold = step_size * self.alpha1;
        for i in 0..n {
            result[i] = if result[i] > l1_threshold {
                result[i] - l1_threshold
            } else if result[i] < -l1_threshold {
                result[i] + l1_threshold
            } else {
                0.0
            };
        }

        // Apply fusion penalty (simplified - in practice use dynamic programming)
        let fusion_threshold = step_size * self.alpha2;
        for i in 1..n {
            let diff = result[i] - result[i - 1];
            if diff.abs() < fusion_threshold {
                let avg = (result[i] + result[i - 1]) / 2.0;
                result[i] = avg;
                result[i - 1] = avg;
            }
        }

        Ok(result)
    }

    /// Compute fused lasso penalty
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        let l1_penalty = self.alpha1 * coef.mapv(|x| x.abs()).sum();

        let mut fusion_penalty = 0.0;
        for i in 1..coef.len() {
            fusion_penalty += (coef[i] - coef[i - 1]).abs();
        }
        fusion_penalty *= self.alpha2;

        l1_penalty + fusion_penalty
    }
}

/// Adaptive Lasso regularization with adaptive weights
///
/// Adaptive lasso uses data-dependent weights in the L1 penalty,
/// providing oracle properties under certain conditions.
#[derive(Debug, Clone)]
pub struct AdaptiveLasso {
    /// Regularization strength
    pub alpha: Float,
    /// Adaptive weights for each feature
    pub weights: Array1<Float>,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
}

impl AdaptiveLasso {
    /// Create adaptive lasso with weights based on initial OLS estimates
    pub fn from_ols(
        alpha: Float,
        X: &Array2<Float>,
        y: &Array1<Float>,
        gamma: Float,
    ) -> Result<Self> {
        // Compute OLS solution for weights
        let XtX = X.t().dot(X);
        let Xty = X.t().dot(y);

        // Solve normal equations (simplified - should use proper linear solver)
        let ols_coef = Self::solve_normal_equations(&XtX, &Xty)?;

        // Compute adaptive weights: w_j = 1 / |beta_ols_j|^gamma
        let weights = ols_coef.mapv(|x| 1.0 / (x.abs() + 1e-8).powf(gamma));

        Ok(Self {
            alpha,
            weights,
            max_iter: 1000,
            tol: 1e-4,
        })
    }

    /// Create adaptive lasso with custom weights
    pub fn with_weights(alpha: Float, weights: Array1<Float>) -> Self {
        Self {
            alpha,
            weights,
            max_iter: 1000,
            tol: 1e-4,
        }
    }

    /// Fit adaptive lasso using coordinate descent
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();
        let mut coef = Array1::zeros(n_features);

        if self.weights.len() != n_features {
            return Err(SklearsError::InvalidInput(
                "Weights length must match number of features".to_string(),
            ));
        }

        // Coordinate descent
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            for j in 0..n_features {
                // Compute residual without j-th feature
                let mut residual = y.clone();
                for k in 0..n_features {
                    if k != j {
                        residual = residual - coef[k] * &X.column(k);
                    }
                }

                // Correlation with j-th feature
                let col_j = X.column(j);
                let rho = col_j.dot(&residual) / n_samples as Float;

                // Weighted soft thresholding
                let threshold = self.alpha * self.weights[j];
                let denominator = col_j.dot(&col_j) / n_samples as Float;

                coef[j] = if rho > threshold {
                    (rho - threshold) / denominator
                } else if rho < -threshold {
                    (rho + threshold) / denominator
                } else {
                    0.0
                };
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        Ok(coef)
    }

    /// Simple normal equations solver (for weights computation)
    fn solve_normal_equations(XtX: &Array2<Float>, Xty: &Array1<Float>) -> Result<Array1<Float>> {
        // Simplified solver - in practice use proper matrix inversion/Cholesky
        let n = XtX.nrows();
        let mut result = Array1::zeros(n);

        // Diagonal approximation for simplicity
        for i in 0..n {
            if XtX[[i, i]].abs() > 1e-8 {
                result[i] = Xty[i] / XtX[[i, i]];
            }
        }

        Ok(result)
    }

    /// Compute adaptive lasso penalty
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        self.alpha
            * coef
                .iter()
                .zip(self.weights.iter())
                .map(|(&c, &w)| w * c.abs())
                .sum::<Float>()
    }
}

/// SCAD (Smoothly Clipped Absolute Deviation) regularization
///
/// SCAD provides a continuously differentiable penalty that applies less
/// penalty to large coefficients than L1, reducing bias in large coefficients
/// while maintaining sparsity for small ones.
///
/// The SCAD penalty function is:
/// - For |β| ≤ λ: λ|β|
/// - For λ < |β| ≤ aλ: (2aλ|β| - β² - λ²)/(2(a-1))  
/// - For |β| > aλ: λ²(a+1)/2
///
/// where λ is the regularization parameter and a > 2 is a shape parameter.
#[derive(Debug, Clone)]
pub struct SCAD {
    /// Regularization strength (lambda)
    pub lambda: Float,
    /// Shape parameter (a > 2)
    pub a: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Step size for coordinate descent
    pub step_size: Float,
}

impl SCAD {
    /// Create a new SCAD regularizer
    pub fn new(lambda: Float, a: Float) -> Result<Self> {
        if a <= 2.0 {
            return Err(SklearsError::InvalidInput(
                "SCAD parameter 'a' must be greater than 2.0".to_string(),
            ));
        }

        Ok(Self {
            lambda,
            a,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            step_size: 0.01,
        })
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set step size for optimization
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = step_size;
        self
    }

    /// Fit SCAD regularized regression using coordinate descent
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "X and y have incompatible shapes: {} vs {}",
                n_samples,
                y.len()
            )));
        }

        // Center target if fitting intercept
        let y_centered = if self.fit_intercept {
            let mean = y.mean().unwrap();
            y - mean
        } else {
            y.clone()
        };

        // Initialize coefficients
        let mut coef = Array1::zeros(n_features);

        // Coordinate descent with SCAD penalty
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            for j in 0..n_features {
                // Compute residual without j-th feature
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let col_k = X.column(k);
                        residual = residual - coef[k] * &col_k;
                    }
                }

                // Compute gradient and apply SCAD soft thresholding
                let col_j = X.column(j);
                let gradient = -col_j.dot(&residual) / n_samples as Float;
                let hessian = col_j.dot(&col_j) / n_samples as Float;

                if hessian > 0.0 {
                    let beta_old = coef[j];
                    let beta_unpenalized = beta_old - self.step_size * gradient / hessian;

                    // Apply SCAD soft thresholding
                    coef[j] = self.scad_soft_threshold(
                        beta_unpenalized,
                        self.lambda * self.step_size / hessian,
                    );
                }
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        Ok(coef)
    }

    /// SCAD soft thresholding operator
    fn scad_soft_threshold(&self, beta: Float, threshold: Float) -> Float {
        let abs_beta = beta.abs();
        let sign = if beta >= 0.0 { 1.0 } else { -1.0 };

        if abs_beta <= threshold {
            // L1 penalty region
            let shrunk = abs_beta - threshold;
            if shrunk > 0.0 {
                sign * shrunk
            } else {
                0.0
            }
        } else if abs_beta <= self.a * threshold {
            // SCAD penalty region
            let numerator = (self.a - 1.0) * beta - sign * self.a * threshold;
            let denominator = self.a - 2.0;
            numerator / denominator
        } else {
            // No penalty region
            beta
        }
    }

    /// Compute SCAD penalty value
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        coef.iter()
            .map(|&beta| self.scad_penalty_single(beta))
            .sum()
    }

    /// SCAD penalty for a single coefficient
    fn scad_penalty_single(&self, beta: Float) -> Float {
        let abs_beta = beta.abs();

        if abs_beta <= self.lambda {
            self.lambda * abs_beta
        } else if abs_beta <= self.a * self.lambda {
            (2.0 * self.a * self.lambda * abs_beta - beta * beta - self.lambda * self.lambda)
                / (2.0 * (self.a - 1.0))
        } else {
            self.lambda * self.lambda * (self.a + 1.0) / 2.0
        }
    }
}

impl Default for SCAD {
    fn default() -> Self {
        Self::new(1.0, 3.7).unwrap() // Standard choice a = 3.7
    }
}

/// MCP (Minimax Concave Penalty) regularization
///
/// MCP provides a concave penalty that applies decreasing marginal penalty
/// as coefficient magnitude increases, reducing bias more aggressively than SCAD
/// while maintaining variable selection properties.
///
/// The MCP penalty function is:
/// - For |β| ≤ γλ: λ|β| - β²/(2γ)
/// - For |β| > γλ: γλ²/2
///
/// where λ is the regularization parameter and γ > 1 is a shape parameter.
#[derive(Debug, Clone)]
pub struct MCP {
    /// Regularization strength (lambda)
    pub lambda: Float,
    /// Shape parameter (gamma > 1)
    pub gamma: Float,
    /// Maximum number of iterations
    pub max_iter: usize,
    /// Convergence tolerance
    pub tol: Float,
    /// Whether to fit intercept
    pub fit_intercept: bool,
    /// Step size for coordinate descent
    pub step_size: Float,
}

impl MCP {
    /// Create a new MCP regularizer
    pub fn new(lambda: Float, gamma: Float) -> Result<Self> {
        if gamma <= 1.0 {
            return Err(SklearsError::InvalidInput(
                "MCP parameter 'gamma' must be greater than 1.0".to_string(),
            ));
        }

        Ok(Self {
            lambda,
            gamma,
            max_iter: 1000,
            tol: 1e-4,
            fit_intercept: true,
            step_size: 0.01,
        })
    }

    /// Set maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set whether to fit intercept
    pub fn fit_intercept(mut self, fit_intercept: bool) -> Self {
        self.fit_intercept = fit_intercept;
        self
    }

    /// Set step size for optimization
    pub fn step_size(mut self, step_size: Float) -> Self {
        self.step_size = step_size;
        self
    }

    /// Fit MCP regularized regression using coordinate descent
    pub fn fit(&self, X: &Array2<Float>, y: &Array1<Float>) -> Result<Array1<Float>> {
        let (n_samples, n_features) = X.dim();

        if y.len() != n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "X and y have incompatible shapes: {} vs {}",
                n_samples,
                y.len()
            )));
        }

        // Center target if fitting intercept
        let y_centered = if self.fit_intercept {
            let mean = y.mean().unwrap();
            y - mean
        } else {
            y.clone()
        };

        // Initialize coefficients
        let mut coef = Array1::zeros(n_features);

        // Coordinate descent with MCP penalty
        for _iter in 0..self.max_iter {
            let old_coef = coef.clone();

            for j in 0..n_features {
                // Compute residual without j-th feature
                let mut residual = y_centered.clone();
                for k in 0..n_features {
                    if k != j {
                        let col_k = X.column(k);
                        residual = residual - coef[k] * &col_k;
                    }
                }

                // Compute gradient and apply MCP soft thresholding
                let col_j = X.column(j);
                let gradient = -col_j.dot(&residual) / n_samples as Float;
                let hessian = col_j.dot(&col_j) / n_samples as Float;

                if hessian > 0.0 {
                    let beta_old = coef[j];
                    let beta_unpenalized = beta_old - self.step_size * gradient / hessian;

                    // Apply MCP soft thresholding
                    coef[j] = self.mcp_soft_threshold(
                        beta_unpenalized,
                        self.lambda * self.step_size / hessian,
                    );
                }
            }

            // Check convergence
            let coef_change = (&coef - &old_coef).mapv(|x| x.abs()).sum();
            if coef_change < self.tol {
                break;
            }
        }

        Ok(coef)
    }

    /// MCP soft thresholding operator
    fn mcp_soft_threshold(&self, beta: Float, threshold: Float) -> Float {
        let abs_beta = beta.abs();
        let sign = if beta >= 0.0 { 1.0 } else { -1.0 };

        if abs_beta <= self.gamma * threshold {
            // MCP penalty region
            let shrunk = abs_beta - threshold;
            if shrunk > 0.0 {
                let denominator = 1.0 - 1.0 / self.gamma;
                sign * shrunk / denominator
            } else {
                0.0
            }
        } else {
            // No penalty region
            beta
        }
    }

    /// Compute MCP penalty value
    pub fn penalty(&self, coef: &Array1<Float>) -> Float {
        coef.iter().map(|&beta| self.mcp_penalty_single(beta)).sum()
    }

    /// MCP penalty for a single coefficient
    fn mcp_penalty_single(&self, beta: Float) -> Float {
        let abs_beta = beta.abs();

        if abs_beta <= self.gamma * self.lambda {
            self.lambda * abs_beta - beta * beta / (2.0 * self.gamma)
        } else {
            self.gamma * self.lambda * self.lambda / 2.0
        }
    }
}

impl Default for MCP {
    fn default() -> Self {
        Self::new(1.0, 3.0).unwrap() // Standard choice gamma = 3.0
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_elastic_net() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let elastic_net = ElasticNet::new(0.1, 0.5);
        let coef = elastic_net.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 2);

        // Test penalty computation
        let penalty = elastic_net.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_group_lasso() {
        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![1.0, 2.0, 3.0];
        let groups = vec![vec![0, 1], vec![2]];

        let group_lasso = GroupLasso::new(0.1, groups);
        let coef = group_lasso.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 3);

        let penalty = group_lasso.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_fused_lasso() {
        let X = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let y = array![1.0, 2.0, 3.0];

        let fused_lasso = FusedLasso::new(0.1, 0.1);
        let coef = fused_lasso.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 3);

        let penalty = fused_lasso.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_adaptive_lasso() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let adaptive_lasso = AdaptiveLasso::from_ols(0.1, &X, &y, 1.0).unwrap();
        let coef = adaptive_lasso.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 2);

        let penalty = adaptive_lasso.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_elastic_net_path() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];
        let alphas = array![0.001, 0.01, 0.1, 1.0];

        let elastic_net = ElasticNet::new(0.1, 0.5);
        let coef_path = elastic_net.path(&X, &y, &alphas).unwrap();

        assert_eq!(coef_path.shape(), &[2, 4]);
    }

    #[test]
    fn test_elastic_net_edge_cases() {
        let X = array![[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.5, 0.5]];
        let y = array![1.0, 1.0, 2.0, 1.0];

        // Pure L1 (Lasso) with weaker regularization
        let lasso = ElasticNet::new(0.01, 1.0);
        let coef_l1 = lasso.fit(&X, &y).unwrap();

        // Pure L2 (Ridge) with weaker regularization
        let ridge = ElasticNet::new(0.01, 0.0);
        let coef_l2 = ridge.fit(&X, &y).unwrap();

        // At least one coefficient should be non-zero
        assert!(coef_l1.iter().any(|&x| x.abs() > 1e-6) || coef_l2.iter().any(|&x| x.abs() > 1e-6));
    }

    #[test]
    fn test_scad() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let scad = SCAD::new(0.1, 3.7).unwrap();
        let coef = scad.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 2);

        let penalty = scad.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_scad_error_cases() {
        // Test invalid parameter a <= 2
        assert!(SCAD::new(0.1, 2.0).is_err());
        assert!(SCAD::new(0.1, 1.5).is_err());

        // Valid parameter should work
        assert!(SCAD::new(0.1, 2.1).is_ok());
        assert!(SCAD::new(0.1, 3.7).is_ok());
    }

    #[test]
    fn test_scad_penalty_function() {
        let scad = SCAD::new(1.0, 3.7).unwrap();

        // Test penalty for small coefficients (L1 region)
        let coef_small = array![0.5, 0.3];
        let penalty_small = scad.penalty(&coef_small);
        let expected_small = 0.5 + 0.3; // L1 penalty
        assert!((penalty_small - expected_small).abs() < 1e-6);

        // Test penalty for large coefficients (no penalty region)
        let coef_large = array![5.0, 4.0];
        let penalty_large = scad.penalty(&coef_large);
        let expected_large = 2.0 * (3.7 + 1.0) / 2.0; // Two coefficients in no-penalty region
        assert!((penalty_large - expected_large).abs() < 1e-6);
    }

    #[test]
    fn test_mcp() {
        let X = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let y = array![1.0, 2.0, 3.0, 4.0];

        let mcp = MCP::new(0.1, 3.0).unwrap();
        let coef = mcp.fit(&X, &y).unwrap();

        assert_eq!(coef.len(), 2);

        let penalty = mcp.penalty(&coef);
        assert!(penalty >= 0.0);
    }

    #[test]
    fn test_mcp_error_cases() {
        // Test invalid parameter gamma <= 1
        assert!(MCP::new(0.1, 1.0).is_err());
        assert!(MCP::new(0.1, 0.5).is_err());

        // Valid parameter should work
        assert!(MCP::new(0.1, 1.1).is_ok());
        assert!(MCP::new(0.1, 3.0).is_ok());
    }

    #[test]
    fn test_mcp_penalty_function() {
        let mcp = MCP::new(1.0, 3.0).unwrap();

        // Test penalty for small coefficients (MCP region)
        let coef_small = array![0.5, 0.3];
        let penalty_small = mcp.penalty(&coef_small);
        let expected_small = (0.5 - 0.5 * 0.5 / (2.0 * 3.0)) + (0.3 - 0.3 * 0.3 / (2.0 * 3.0));
        assert!((penalty_small - expected_small).abs() < 1e-6);

        // Test penalty for large coefficients (no penalty region)
        let coef_large = array![5.0, 4.0];
        let penalty_large = mcp.penalty(&coef_large);
        let expected_large = 2.0 * (3.0 * 1.0 * 1.0 / 2.0); // Two coefficients in no-penalty region
        assert!((penalty_large - expected_large).abs() < 1e-6);
    }

    #[test]
    fn test_scad_vs_mcp_penalties() {
        let scad = SCAD::new(0.1, 3.7).unwrap();
        let mcp = MCP::new(0.1, 3.0).unwrap();

        // For very small coefficients, both should behave like L1
        let coef_tiny = array![0.01, 0.02];
        let scad_penalty = scad.penalty(&coef_tiny);
        let mcp_penalty = mcp.penalty(&coef_tiny);
        let l1_penalty = 0.1 * (0.01 + 0.02);

        // SCAD should be approximately L1 for small coefficients
        assert!((scad_penalty - l1_penalty).abs() < 1e-3);

        // MCP should be less than L1 for small coefficients due to concavity
        assert!(mcp_penalty < l1_penalty);
    }

    #[test]
    fn test_regularization_builders() {
        // Test SCAD builder pattern
        let scad = SCAD::new(0.1, 3.7)
            .unwrap()
            .max_iter(500)
            .tol(1e-6)
            .fit_intercept(false)
            .step_size(0.02);

        assert_eq!(scad.max_iter, 500);
        assert_eq!(scad.tol, 1e-6);
        assert_eq!(scad.fit_intercept, false);
        assert_eq!(scad.step_size, 0.02);

        // Test MCP builder pattern
        let mcp = MCP::new(0.2, 2.5)
            .unwrap()
            .max_iter(800)
            .tol(1e-5)
            .fit_intercept(true)
            .step_size(0.05);

        assert_eq!(mcp.max_iter, 800);
        assert_eq!(mcp.tol, 1e-5);
        assert_eq!(mcp.fit_intercept, true);
        assert_eq!(mcp.step_size, 0.05);
    }
}
