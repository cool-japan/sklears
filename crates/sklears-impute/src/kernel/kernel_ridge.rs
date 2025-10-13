//! Kernel Ridge Regression Imputer
//!
//! This module provides kernel ridge regression for missing value imputation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Kernel Ridge Regression Imputer
///
/// Imputation using kernel ridge regression with various kernel functions.
/// This method learns non-linear relationships between features using kernel tricks
/// and applies ridge regularization for numerical stability.
///
/// # Parameters
///
/// * `alpha` - Ridge regularization parameter
/// * `kernel` - Kernel function type ('rbf', 'linear', 'polynomial')
/// * `gamma` - Kernel coefficient for 'rbf' kernel
/// * `degree` - Degree for polynomial kernel
/// * `coef0` - Independent term for polynomial kernel
/// * `missing_values` - The placeholder for missing values
///
/// # Examples
///
/// ```
/// use sklears_impute::KernelRidgeImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = KernelRidgeImputer::new()
///     .alpha(1.0)
///     .kernel("rbf".to_string())
///     .gamma(1.0);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct KernelRidgeImputer<S = Untrained> {
    state: S,
    alpha: f64,
    kernel: String,
    gamma: f64,
    degree: usize,
    coef0: f64,
    missing_values: f64,
    max_iter: usize,
    tol: f64,
}

/// Trained state for KernelRidgeImputer
#[derive(Debug, Clone)]
pub struct KernelRidgeImputerTrained {
    X_train_: Array2<f64>,
    y_train_: HashMap<usize, Array1<f64>>, // Target values for each feature
    alpha_: HashMap<usize, Array1<f64>>,   // Dual coefficients for each feature
    n_features_in_: usize,
}

impl KernelRidgeImputer<Untrained> {
    /// Create a new KernelRidgeImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            alpha: 1.0,
            kernel: "rbf".to_string(),
            gamma: 1.0,
            degree: 3,
            coef0: 1.0,
            missing_values: f64::NAN,
            max_iter: 1000,
            tol: 1e-6,
        }
    }

    /// Set the ridge regularization parameter
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.alpha = alpha;
        self
    }

    /// Set the kernel function
    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    /// Set the kernel coefficient for RBF kernel
    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    /// Set the degree for polynomial kernel
    pub fn degree(mut self, degree: usize) -> Self {
        self.degree = degree;
        self
    }

    /// Set the independent term for polynomial kernel
    pub fn coef0(mut self, coef0: f64) -> Self {
        self.coef0 = coef0;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the maximum number of iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the tolerance for convergence
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn kernel_function(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        // Basic kernel computations (SIMD optimizations removed for compatibility)
        match self.kernel.as_str() {
            "linear" => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(),
            "polynomial" => {
                let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                (self.gamma * dot + self.coef0).powi(self.degree as i32)
            }
            "rbf" => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                (-self.gamma * dist_sq).exp()
            }
            "sigmoid" => {
                let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                (self.gamma * dot + self.coef0).tanh()
            }
            "laplacian" => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                (-self.gamma * dist).exp()
            }
            _ => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(), // Default to linear
        }
    }
}

impl Default for KernelRidgeImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for KernelRidgeImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for KernelRidgeImputer<Untrained> {
    type Fitted = KernelRidgeImputer<KernelRidgeImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x as f64);
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

        // Train a kernel ridge regressor for each feature
        let mut y_train = HashMap::new();
        let mut alpha_coeffs = HashMap::new();

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

            // Compute kernel matrix
            let K = self.compute_kernel_matrix(&X_feat)?;

            // Solve kernel ridge regression: (K + αI)α = y
            let mut K_reg = K.clone();
            for i in 0..K_reg.nrows() {
                K_reg[[i, i]] += self.alpha;
            }

            let alpha_vec = self.solve_linear_system(&K_reg, &y_target)?;

            y_train.insert(target_feature, y_target);
            alpha_coeffs.insert(target_feature, alpha_vec);
        }

        Ok(KernelRidgeImputer {
            state: KernelRidgeImputerTrained {
                X_train_: X_train,
                y_train_: y_train,
                alpha_: alpha_coeffs,
                n_features_in_: n_features,
            },
            alpha: self.alpha,
            kernel: self.kernel,
            gamma: self.gamma,
            degree: self.degree,
            coef0: self.coef0,
            missing_values: self.missing_values,
            max_iter: self.max_iter,
            tol: self.tol,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for KernelRidgeImputer<KernelRidgeImputerTrained>
{
    #[allow(non_snake_case)]
    fn transform(&self, X: &ArrayView2<'_, Float>) -> SklResult<Array2<Float>> {
        let X = X.mapv(|x| x as f64);
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
                    let imputed_value = self.predict_feature_value(&X_imputed, i, j)?;
                    X_imputed[[i, j]] = imputed_value;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl KernelRidgeImputer<Untrained> {
    fn compute_kernel_matrix(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = X.nrows();
        let mut K = Array2::<f64>::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in 0..n_samples {
                let x1 = X.row(i);
                let x2 = X.row(j);
                K[[i, j]] = match self.kernel.as_str() {
                    "linear" => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(),
                    "polynomial" => {
                        let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                        (self.gamma * dot + self.coef0).powi(self.degree as i32)
                    }
                    "rbf" => {
                        let dist_sq = x1
                            .iter()
                            .zip(x2.iter())
                            .map(|(a, b)| (a - b).powi(2))
                            .sum::<f64>();
                        (-self.gamma * dist_sq).exp()
                    }
                    "sigmoid" => {
                        let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                        (self.gamma * dot + self.coef0).tanh()
                    }
                    "laplacian" => {
                        let dist = x1
                            .iter()
                            .zip(x2.iter())
                            .map(|(a, b)| (a - b).abs())
                            .sum::<f64>();
                        (-self.gamma * dist).exp()
                    }
                    _ => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(),
                };
            }
        }
        Ok(K)
    }

    #[allow(non_snake_case)]
    fn solve_linear_system(&self, A: &Array2<f64>, b: &Array1<f64>) -> SklResult<Array1<f64>> {
        let n = A.nrows();
        if n != A.ncols() || n != b.len() {
            return Err(SklearsError::InvalidInput(
                "Matrix dimensions don't match".to_string(),
            ));
        }

        // Use Cholesky decomposition for positive definite matrices
        let L = self.cholesky_decomposition(A)?;

        // Solve Ly = b
        let mut y = Array1::zeros(n);
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..i {
                sum += L[[i, j]] * y[j];
            }
            y[i] = (b[i] - sum) / L[[i, i]];
        }

        // Solve L^T x = y
        let mut x = Array1::zeros(n);
        for i in (0..n).rev() {
            let mut sum = 0.0;
            for j in (i + 1)..n {
                sum += L[[j, i]] * x[j];
            }
            x[i] = (y[i] - sum) / L[[i, i]];
        }

        Ok(x)
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

impl KernelRidgeImputer<KernelRidgeImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    fn kernel_function(&self, x1: &ArrayView1<f64>, x2: &ArrayView1<f64>) -> f64 {
        // Basic kernel computations (SIMD optimizations removed for compatibility)
        match self.kernel.as_str() {
            "linear" => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(),
            "polynomial" => {
                let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                (self.gamma * dot + self.coef0).powi(self.degree as i32)
            }
            "rbf" => {
                let dist_sq = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>();
                (-self.gamma * dist_sq).exp()
            }
            "sigmoid" => {
                let dot = x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>();
                (self.gamma * dot + self.coef0).tanh()
            }
            "laplacian" => {
                let dist = x1
                    .iter()
                    .zip(x2.iter())
                    .map(|(a, b)| (a - b).abs())
                    .sum::<f64>();
                (-self.gamma * dist).exp()
            }
            _ => x1.iter().zip(x2.iter()).map(|(a, b)| a * b).sum::<f64>(), // Default to linear
        }
    }

    #[allow(non_snake_case)]
    fn predict_feature_value(
        &self,
        X: &Array2<f64>,
        sample_idx: usize,
        target_feature: usize,
    ) -> SklResult<f64> {
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

        // Compute kernel vector between x_feat and training data
        let X_train_feat = self.get_training_features(target_feature);
        let mut k_vec = Array1::zeros(X_train_feat.nrows());
        for i in 0..X_train_feat.nrows() {
            k_vec[i] = self.kernel_function(&x_feat.view(), &X_train_feat.row(i));
        }

        // Get dual coefficients for this feature
        let alpha =
            self.state.alpha_.get(&target_feature).ok_or_else(|| {
                SklearsError::InvalidInput("Missing dual coefficients".to_string())
            })?;

        // Predict: f(x) = Σ α_i K(x, x_i)
        let prediction = k_vec.dot(alpha);
        Ok(prediction)
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
}
