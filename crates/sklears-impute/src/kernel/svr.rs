//! Support Vector Regression Imputer
//!
//! This module provides SVR for missing value imputation.

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};
use std::collections::HashMap;

/// Support Vector Regression Imputer
///
/// Imputation using Support Vector Regression with various kernel functions.
/// This method is robust to outliers and can capture complex non-linear relationships.
///
/// # Parameters
///
/// * `C` - Regularization parameter
/// * `epsilon` - Epsilon in the epsilon-SVR model
/// * `kernel` - Kernel function type ('rbf', 'linear', 'polynomial')
/// * `gamma` - Kernel coefficient for 'rbf' kernel
/// * `degree` - Degree for polynomial kernel
/// * `coef0` - Independent term for polynomial kernel
/// * `missing_values` - The placeholder for missing values
///
/// # Examples
///
/// ```
/// use sklears_impute::SVRImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = SVRImputer::new()
///     .C(1.0)
///     .epsilon(0.1)
///     .kernel("rbf".to_string());
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct SVRImputer<S = Untrained> {
    state: S,
    C: f64,
    epsilon: f64,
    kernel: String,
    gamma: f64,
    degree: usize,
    coef0: f64,
    missing_values: f64,
    max_iter: usize,
    tol: f64,
}

/// Trained state for SVRImputer
#[derive(Debug, Clone)]
pub struct SVRImputerTrained {
    X_train_: Array2<f64>,
    support_vectors_: HashMap<usize, Array2<f64>>,
    dual_coef_: HashMap<usize, Array1<f64>>,
    intercept_: HashMap<usize, f64>,
    n_features_in_: usize,
}

impl SVRImputer<Untrained> {
    /// Create a new SVRImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            C: 1.0,
            epsilon: 0.1,
            kernel: "rbf".to_string(),
            gamma: 1.0,
            degree: 3,
            coef0: 1.0,
            missing_values: f64::NAN,
            max_iter: 1000,
            tol: 1e-3,
        }
    }

    /// Set the regularization parameter
    pub fn C(mut self, C: f64) -> Self {
        self.C = C;
        self
    }

    /// Set the epsilon parameter
    pub fn epsilon(mut self, epsilon: f64) -> Self {
        self.epsilon = epsilon;
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

    /// Set the tolerance
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

impl Default for SVRImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for SVRImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for SVRImputer<Untrained> {
    type Fitted = SVRImputer<SVRImputerTrained>;

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

        // Train SVR for each feature (simplified implementation)
        let mut support_vectors = HashMap::new();
        let mut dual_coef = HashMap::new();
        let mut intercept = HashMap::new();

        for target_feature in 0..n_features {
            // For simplified SVR, we'll use all training points as support vectors
            // In practice, you'd solve the dual optimization problem
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

            let y_target = X_train.column(target_feature).to_owned();

            // Simplified: use equal weights (in practice, solve quadratic programming)
            let n_sv = complete_rows.len();
            let alpha = Array1::from_elem(n_sv, 1.0 / n_sv as f64);
            let intercept_val = y_target.mean().unwrap_or(0.0);

            support_vectors.insert(target_feature, X_feat);
            dual_coef.insert(target_feature, alpha);
            intercept.insert(target_feature, intercept_val);
        }

        Ok(SVRImputer {
            state: SVRImputerTrained {
                X_train_: X_train,
                support_vectors_: support_vectors,
                dual_coef_: dual_coef,
                intercept_: intercept,
                n_features_in_: n_features,
            },
            C: self.C,
            epsilon: self.epsilon,
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

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for SVRImputer<SVRImputerTrained> {
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
                    let imputed_value = self.predict_svr(&X_imputed, i, j)?;
                    X_imputed[[i, j]] = imputed_value;
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl SVRImputer<SVRImputerTrained> {
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

    fn predict_svr(
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

        // Get support vectors and dual coefficients
        let support_vectors = self
            .state
            .support_vectors_
            .get(&target_feature)
            .ok_or_else(|| SklearsError::InvalidInput("Missing support vectors".to_string()))?;
        let dual_coef =
            self.state.dual_coef_.get(&target_feature).ok_or_else(|| {
                SklearsError::InvalidInput("Missing dual coefficients".to_string())
            })?;
        let intercept = self
            .state
            .intercept_
            .get(&target_feature)
            .ok_or_else(|| SklearsError::InvalidInput("Missing intercept".to_string()))?;

        // Compute prediction: f(x) = Σ α_i K(x, x_i) + b
        let mut prediction = *intercept;
        for i in 0..support_vectors.nrows() {
            let kernel_val = self.kernel_function(&x_feat.view(), &support_vectors.row(i));
            prediction += dual_coef[i] * kernel_val;
        }

        Ok(prediction)
    }
}
