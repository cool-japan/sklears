//! Gaussian Process Regression Imputer
//!
//! This module provides GP regression for missing value imputation with uncertainty quantification.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Gaussian Process Regression Imputer
///
/// Imputation using Gaussian Process Regression with Bayesian inference.
/// This method provides uncertainty quantification along with predictions
/// and can capture complex non-linear relationships with principled uncertainty.
///
/// # Parameters
///
/// * `kernel` - Kernel function type ('rbf', 'matern32', 'matern52', 'linear')
/// * `length_scale` - Length scale parameter for the kernel
/// * `length_scale_bounds` - Bounds for length scale optimization
/// * `nu` - Smoothness parameter for Matern kernels
/// * `alpha` - Noise regularization parameter
/// * `optimizer` - Method for hyperparameter optimization
/// * `n_restarts_optimizer` - Number of optimizer restarts
/// * `missing_values` - The placeholder for missing values
///
/// # Examples
///
/// ```
/// use sklears_impute::GaussianProcessImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = GaussianProcessImputer::new()
///     .kernel("rbf".to_string())
///     .length_scale(1.0)
///     .alpha(1e-6);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct GaussianProcessImputer<S = Untrained> {
    state: S,
    kernel: String,
    length_scale: f64,
    length_scale_bounds: (f64, f64),
    nu: f64,
    alpha: f64,
    optimizer: String,
    n_restarts_optimizer: usize,
    missing_values: f64,
    normalize_y: bool,
    random_state: Option<u64>,
}

/// Trained state for GaussianProcessImputer
#[derive(Debug, Clone)]
pub struct GaussianProcessImputerTrained {
    X_train_: Array2<f64>,
    y_train_: HashMap<usize, Array1<f64>>,
    L_: HashMap<usize, Array2<f64>>, // Cholesky decomposition of K
    alpha_: HashMap<usize, Array1<f64>>, // K^{-1} * y
    K_inv_: HashMap<usize, Array2<f64>>, // Inverse of kernel matrix
    log_marginal_likelihood_: HashMap<usize, f64>,
    n_features_in_: usize,
    optimized_kernel_params_: HashMap<usize, (f64, f64)>, // (length_scale, alpha)
}

/// Gaussian Process prediction result with uncertainty
#[derive(Debug, Clone)]
pub struct GPPredictionResult {
    /// mean
    pub mean: f64,
    /// std
    pub std: f64,
    /// confidence_interval_95
    pub confidence_interval_95: (f64, f64),
}

impl GaussianProcessImputer<Untrained> {
    /// Create a new GaussianProcessImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: "rbf".to_string(),
            length_scale: 1.0,
            length_scale_bounds: (1e-5, 1e5),
            nu: 1.5,
            alpha: 1e-10,
            optimizer: "fmin_l_bfgs_b".to_string(),
            n_restarts_optimizer: 0,
            missing_values: f64::NAN,
            normalize_y: false,
            random_state: None,
        }
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the length scale parameter
    pub fn length_scale(mut self, length_scale: f64) -> Self {
        self.length_scale = length_scale;
        self
    }

    /// Set the bounds for length scale optimization
    pub fn length_scale_bounds(mut self, bounds: (f64, f64)) -> Self {
        self.length_scale_bounds = bounds;
        self
    }

    /// Set the smoothness parameter for Matern kernels
    pub fn nu(mut self, nu: f64) -> Self {
        self.nu = nu;
        self
    }

    /// Set the noise regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the optimizer method
    pub fn optimizer(mut self, optimizer: String) -> Self {
        self.optimizer = optimizer;
        self
    }

    /// Set the number of optimizer restarts
    pub fn n_restarts_optimizer(mut self, n_restarts: usize) -> Self {
        self.n_restarts_optimizer = n_restarts;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set whether to normalize target values
    pub fn normalize_y(mut self, normalize_y: bool) -> Self {
        self.normalize_y = normalize_y;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn kernel_function(
        &self,
        x1: &ArrayView1<f64>,
        x2: &ArrayView1<f64>,
        length_scale: f64,
    ) -> f64 {
        match self.kernel.as_str() {
            "linear" => x1.dot(x2),
            "rbf" => {
                let diff = (x1 - x2).mapv(|x| x * x).sum();
                (-0.5 * diff / (length_scale * length_scale)).exp()
            }
            "matern32" => {
                let r = ((x1 - x2).mapv(|x| x * x).sum().sqrt()) / length_scale;
                (1.0 + (3.0_f64).sqrt() * r) * (-(3.0_f64).sqrt() * r).exp()
            }
            "matern52" => {
                let r = ((x1 - x2).mapv(|x| x * x).sum().sqrt()) / length_scale;
                (1.0 + (5.0_f64).sqrt() * r + 5.0 * r * r / 3.0) * (-(5.0_f64).sqrt() * r).exp()
            }
            _ => {
                // Default to RBF
                let diff = (x1 - x2).mapv(|x| x * x).sum();
                (-0.5 * diff / (length_scale * length_scale)).exp()
            }
        }
    }

    fn compute_kernel_matrix(&self, X: &Array2<f64>, length_scale: f64, alpha: f64) -> Array2<f64> {
        let n_samples = X.nrows();

        // Use optimized Gaussian process kernel matrix computation
        match self.kernel.as_str() {
            "rbf" | "squared_exponential" => {
                let mut K = Array2::zeros((n_samples, n_samples));

                // Optimized kernel matrix computation for GP
                for i in 0..n_samples {
                    let x1 = X.row(i);
                    for j in i..n_samples {
                        // Exploit symmetry
                        let x2 = X.row(j);
                        let dist_sq = x1
                            .iter()
                            .zip(x2.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        let kernel_value = (-0.5 * dist_sq / (length_scale * length_scale)).exp();
                        K[[i, j]] = kernel_value;
                        if i != j {
                            K[[j, i]] = kernel_value; // Exploit symmetry
                        }
                    }
                    K[[i, i]] += alpha; // Add noise to diagonal
                }
                K
            }
            _ => {
                // Fallback to original implementation for other kernels
                let mut K = Array2::zeros((n_samples, n_samples));
                for i in 0..n_samples {
                    for j in 0..n_samples {
                        K[[i, j]] = self.kernel_function(&X.row(i), &X.row(j), length_scale);
                        if i == j {
                            K[[i, j]] += alpha;
                        }
                    }
                }
                K
            }
        }
    }

    #[allow(non_snake_case)]
    fn log_marginal_likelihood(
        &self,
        X: &Array2<f64>,
        y: &Array1<f64>,
        length_scale: f64,
        alpha: f64,
    ) -> f64 {
        let K = self.compute_kernel_matrix(X, length_scale, alpha);

        // Compute Cholesky decomposition
        if let Ok(L) = self.cholesky_decomposition(&K) {
            // Solve L * alpha = y
            let alpha_vec = self.solve_triangular_lower(&L, y);

            // Log marginal likelihood = -0.5 * y^T * K^{-1} * y - 0.5 * log|K| - 0.5 * n * log(2π)
            let data_fit = -0.5 * y.dot(&alpha_vec);
            let complexity_penalty = -L.diag().mapv(|x| x.ln()).sum();
            let normalizing_constant = -0.5 * y.len() as f64 * (2.0 * std::f64::consts::PI).ln();

            data_fit + complexity_penalty + normalizing_constant
        } else {
            f64::NEG_INFINITY // Invalid kernel matrix
        }
    }

    fn optimize_hyperparameters(&self, X: &Array2<f64>, y: &Array1<f64>) -> (f64, f64) {
        let mut best_length_scale = self.length_scale;
        let mut best_alpha = self.alpha;
        let mut best_likelihood = f64::NEG_INFINITY;

        // Simple grid search optimization (in practice, you'd use L-BFGS-B)
        let length_scales = [0.1, 0.5, 1.0, 2.0, 5.0];
        let alphas = [1e-10, 1e-8, 1e-6, 1e-4, 1e-2];

        for &ls in &length_scales {
            for &alpha in &alphas {
                let likelihood = self.log_marginal_likelihood(X, y, ls, alpha);
                if likelihood > best_likelihood {
                    best_likelihood = likelihood;
                    best_length_scale = ls;
                    best_alpha = alpha;
                }
            }
        }

        (best_length_scale, best_alpha)
    }

    fn solve_triangular_lower(&self, L: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = L.nrows();
        let mut y = Array1::zeros(n);

        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += L[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / L[[i, i]];
        }

        y
    }

    fn solve_triangular_upper(&self, U: &Array2<f64>, b: &Array1<f64>) -> Array1<f64> {
        let n = U.nrows();
        let mut x = Array1::zeros(n);

        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += U[[i, j]] * x[j];
            }
            x[i] = (b[i] - sum) / U[[i, i]];
        }

        x
    }

    fn cholesky_decomposition(&self, A: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = A.nrows();
        let mut L = Array2::zeros((n, n));

        for i in 0..n {
            for j in 0..=i {
                if i == j {
                    // Diagonal elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += L[[j, k]] * L[[j, k]];
                    }
                    let val = A[[j, j]] - sum;
                    if val <= 0.0 {
                        return Err(SklearsError::InvalidInput(
                            "Matrix is not positive definite".to_string(),
                        ));
                    }
                    L[[j, j]] = val.sqrt();
                } else {
                    // Lower triangular elements
                    let mut sum = 0.0;
                    for k in 0..j {
                        sum += L[[i, k]] * L[[j, k]];
                    }
                    L[[i, j]] = (A[[i, j]] - sum) / L[[j, j]];
                }
            }
        }

        Ok(L)
    }
}

impl Default for GaussianProcessImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for GaussianProcessImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for GaussianProcessImputer<Untrained> {
    type Fitted = GaussianProcessImputer<GaussianProcessImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        // Collect complete cases for training
        let mut complete_rows = Vec::new();
        for i in 0..n_samples {
            let mut is_complete = true;
            for j in 0..n_features {
                if self.is_missing(X[[i, j]]) {
                    is_complete = false;
                    break;
                }
            }
            if is_complete {
                complete_rows.push(i);
            }
        }

        if complete_rows.is_empty() {
            return Err(SklearsError::InvalidInput(
                "No complete cases found for training".to_string(),
            ));
        }

        // Create training data from complete cases
        let mut X_train = Array2::zeros((complete_rows.len(), n_features));
        for (new_i, &orig_i) in complete_rows.iter().enumerate() {
            for j in 0..n_features {
                X_train[[new_i, j]] = X[[orig_i, j]];
            }
        }

        // Train GP for each feature
        let mut y_train = HashMap::new();
        let mut L_chol = HashMap::new();
        let mut alpha_coeffs = HashMap::new();
        let mut K_inv_matrices = HashMap::new();
        let mut log_likelihoods = HashMap::new();
        let mut optimized_params = HashMap::new();

        for target_feature in 0..n_features {
            // Prepare feature matrix (all features except target)
            let mut X_feat = Array2::zeros((complete_rows.len(), n_features - 1));
            let mut col_idx = 0;
            for j in 0..n_features {
                if j != target_feature {
                    for i in 0..complete_rows.len() {
                        X_feat[[i, col_idx]] = X_train[[i, j]];
                    }
                    col_idx += 1;
                }
            }

            // Target values for this feature
            let y_target = X_train.column(target_feature).to_owned();

            // Optimize hyperparameters
            let (opt_length_scale, opt_alpha) = self.optimize_hyperparameters(&X_feat, &y_target);

            // Compute kernel matrix with optimized parameters
            let K = self.compute_kernel_matrix(&X_feat, opt_length_scale, opt_alpha);

            // Compute Cholesky decomposition
            let L = self.cholesky_decomposition(&K)?;

            // Solve for alpha: K * alpha = y
            let alpha_vec = self.solve_triangular_lower(&L, &y_target);
            let alpha_final = self.solve_triangular_upper(&L.t().to_owned(), &alpha_vec);

            // Compute K^{-1} for predictions
            let I = Array2::eye(K.nrows());
            let mut K_inv = Array2::zeros((K.nrows(), K.ncols()));
            for i in 0..I.ncols() {
                let col = I.column(i);
                let y_temp = self.solve_triangular_lower(&L, &col.to_owned());
                let x_temp = self.solve_triangular_upper(&L.t().to_owned(), &y_temp);
                for j in 0..K.nrows() {
                    K_inv[[j, i]] = x_temp[j];
                }
            }

            // Compute log marginal likelihood
            let log_likelihood =
                self.log_marginal_likelihood(&X_feat, &y_target, opt_length_scale, opt_alpha);

            y_train.insert(target_feature, y_target);
            L_chol.insert(target_feature, L);
            alpha_coeffs.insert(target_feature, alpha_final);
            K_inv_matrices.insert(target_feature, K_inv);
            log_likelihoods.insert(target_feature, log_likelihood);
            optimized_params.insert(target_feature, (opt_length_scale, opt_alpha));
        }

        Ok(GaussianProcessImputer {
            state: GaussianProcessImputerTrained {
                X_train_: X_train,
                y_train_: y_train,
                L_: L_chol,
                alpha_: alpha_coeffs,
                K_inv_: K_inv_matrices,
                log_marginal_likelihood_: log_likelihoods,
                n_features_in_: n_features,
                optimized_kernel_params_: optimized_params,
            },
            kernel: self.kernel,
            length_scale: self.length_scale,
            length_scale_bounds: self.length_scale_bounds,
            nu: self.nu,
            alpha: self.alpha,
            optimizer: self.optimizer,
            n_restarts_optimizer: self.n_restarts_optimizer,
            missing_values: self.missing_values,
            normalize_y: self.normalize_y,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for GaussianProcessImputer<GaussianProcessImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut X_imputed = X.clone();

        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_imputed[[i, j]]) {
                    let prediction = self.predict_gp(&X_imputed, i, j)?;
                    X_imputed[[i, j]] = prediction.mean;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl GaussianProcessImputer<GaussianProcessImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn kernel_function(
        &self,
        x1: &ArrayView1<f64>,
        x2: &ArrayView1<f64>,
        length_scale: f64,
    ) -> f64 {
        match self.kernel.as_str() {
            "linear" => x1.dot(x2),
            "rbf" => {
                let diff = (x1 - x2).mapv(|x| x * x).sum();
                (-0.5 * diff / (length_scale * length_scale)).exp()
            }
            "matern32" => {
                let r = ((x1 - x2).mapv(|x| x * x).sum().sqrt()) / length_scale;
                (1.0 + (3.0_f64).sqrt() * r) * (-(3.0_f64).sqrt() * r).exp()
            }
            "matern52" => {
                let r = ((x1 - x2).mapv(|x| x * x).sum().sqrt()) / length_scale;
                (1.0 + (5.0_f64).sqrt() * r + 5.0 * r * r / 3.0) * (-(5.0_f64).sqrt() * r).exp()
            }
            _ => {
                // Default to RBF
                let diff = (x1 - x2).mapv(|x| x * x).sum();
                (-0.5 * diff / (length_scale * length_scale)).exp()
            }
        }
    }

    /// Predict with uncertainty quantification
    #[allow(non_snake_case)]
    pub fn predict_with_uncertainty(
        &self,
        X: &ArrayView2<'_, Float>,
    ) -> SklResult<Vec<Vec<GPPredictionResult>>> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if n_features != self.state.n_features_in_ {
            return Err(SklearsError::InvalidInput(format!(
                "Number of features {} does not match training features {}",
                n_features, self.state.n_features_in_
            )));
        }

        let mut predictions = Vec::new();

        for i in 0..n_samples {
            let mut sample_predictions = Vec::new();
            for j in 0..n_features {
                if self.is_missing(X[[i, j]]) {
                    let prediction = self.predict_gp(&X, i, j)?;
                    sample_predictions.push(prediction);
                } else {
                    // For observed values, return zero uncertainty
                    sample_predictions.push(GPPredictionResult {
                        mean: X[[i, j]],
                        std: 0.0,
                        confidence_interval_95: (X[[i, j]], X[[i, j]]),
                    });
                }
            }
            predictions.push(sample_predictions);
        }

        Ok(predictions)
    }

    #[allow(non_snake_case)]
    fn predict_gp(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        target_feature: usize,
    ) -> SklResult<GPPredictionResult> {
        // Get optimized parameters
        let (length_scale, alpha) = self
            .state
            .optimized_kernel_params_
            .get(&target_feature)
            .ok_or_else(|| {
                SklearsError::InvalidInput("Missing optimized parameters".to_string())
            })?;

        // Prepare feature vector (all features except target)
        let mut x_feat = Array1::zeros(self.state.n_features_in_ - 1);
        let mut col_idx = 0;
        for j in 0..self.state.n_features_in_ {
            if j != target_feature {
                x_feat[col_idx] = if self.is_missing(X[[sample_idx, j]]) {
                    // Use mean of training data for missing features
                    self.state.X_train_.column(j).mean().unwrap_or(0.0)
                } else {
                    X[[sample_idx, j]]
                };
                col_idx += 1;
            }
        }

        // Get training data for this feature
        let X_train_feat = self.get_training_features(target_feature);
        let alpha_vec =
            self.state.alpha_.get(&target_feature).ok_or_else(|| {
                SklearsError::InvalidInput("Missing alpha coefficients".to_string())
            })?;
        let y_train =
            self.state.y_train_.get(&target_feature).ok_or_else(|| {
                SklearsError::InvalidInput("Missing training targets".to_string())
            })?;
        let K_inv = self
            .state
            .K_inv_
            .get(&target_feature)
            .ok_or_else(|| SklearsError::InvalidInput("Missing K_inv matrix".to_string()))?;

        // Compute kernel vector k* = K(X*, X)
        let mut k_star = Array1::zeros(X_train_feat.nrows());
        for i in 0..X_train_feat.nrows() {
            k_star[i] = self.kernel_function(&x_feat.view(), &X_train_feat.row(i), *length_scale);
        }

        // Compute mean: μ* = k*^T * α
        let mut mean = k_star.dot(alpha_vec);

        // Compute basic statistics for fallback logic
        let mut y_min = f64::INFINITY;
        let mut y_max = f64::NEG_INFINITY;
        let mut y_sum = 0.0;
        for &val in y_train.iter() {
            if val.is_finite() {
                y_min = y_min.min(val);
                y_max = y_max.max(val);
                y_sum += val;
            }
        }
        let sample_count = y_train.len() as f64;
        let fallback_mean = if sample_count > 0.0 {
            y_sum / sample_count
        } else {
            0.0
        };

        if !mean.is_finite() {
            mean = fallback_mean;
        } else if y_min.is_finite() && y_max.is_finite() {
            let range = (y_max - y_min).abs().max(1.0);
            let lower_bound = y_min - 0.5 * range;
            let upper_bound = y_max + 0.5 * range;
            if mean < lower_bound {
                mean = lower_bound;
            } else if mean > upper_bound {
                mean = upper_bound;
            }
        }

        // Compute variance: σ*² = k** - k*^T * K^{-1} * k*
        let k_star_star =
            self.kernel_function(&x_feat.view(), &x_feat.view(), *length_scale) + alpha;
        let variance = k_star_star - k_star.dot(&K_inv.dot(&k_star));
        let mut std = variance.max(0.0).sqrt(); // Ensure non-negative variance
        if !std.is_finite() {
            let fallback_var = if sample_count > 1.0 {
                y_train
                    .iter()
                    .map(|&v| {
                        let diff = v - fallback_mean;
                        diff * diff
                    })
                    .sum::<f64>()
                    / (sample_count - 1.0)
            } else {
                0.0
            };
            std = fallback_var.max(0.0).sqrt();
        }

        // Compute 95% confidence interval
        let ci_width = 1.96 * std;
        let confidence_interval_95 = (mean - ci_width, mean + ci_width);

        Ok(GPPredictionResult {
            mean,
            std,
            confidence_interval_95,
        })
    }

    fn get_training_features(&self, target_feature: usize) -> Array2<f64> {
        let n_train = self.state.X_train_.nrows();
        let mut X_feat = Array2::zeros((n_train, self.state.n_features_in_ - 1));

        let mut col_idx = 0;
        for j in 0..self.state.n_features_in_ {
            if j != target_feature {
                for i in 0..n_train {
                    X_feat[[i, col_idx]] = self.state.X_train_[[i, j]];
                }
                col_idx += 1;
            }
        }

        X_feat
    }

    /// Get the log marginal likelihood for each feature
    pub fn log_marginal_likelihood(&self) -> &HashMap<usize, f64> {
        &self.state.log_marginal_likelihood_
    }

    /// Get the optimized kernel parameters for each feature
    pub fn optimized_kernel_params(&self) -> &HashMap<usize, (f64, f64)> {
        &self.state.optimized_kernel_params_
    }
}
