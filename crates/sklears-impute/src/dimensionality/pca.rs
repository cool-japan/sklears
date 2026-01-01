//! Principal Component Analysis Imputer
//!
//! This module provides PCA-based imputation for missing value estimation.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView2, Axis};
use scirs2_core::random::Random;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Principal Component Analysis Imputer
///
/// Imputation using Principal Component Analysis to find low-dimensional representations
/// of the data. Missing values are imputed by projecting onto the principal component space
/// and reconstructing back to the original space.
///
/// # Parameters
///
/// * `n_components` - Number of principal components to use
/// * `max_iter` - Maximum number of iterations for iterative optimization
/// * `tol` - Tolerance for convergence
/// * `missing_values` - The placeholder for missing values
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::PCAImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0, 4.0], [f64::NAN, 3.0, 4.0, 5.0], [7.0, f64::NAN, 6.0, 8.0]];
///
/// let imputer = PCAImputer::new()
///     .n_components(2)
///     .max_iter(100);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct PCAImputer<S = Untrained> {
    state: S,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    missing_values: f64,
    random_state: Option<u64>,
}

/// Trained state for PCAImputer
#[derive(Debug, Clone)]
pub struct PCAImputerTrained {
    components_: Array2<f64>,
    mean_: Array1<f64>,
    explained_variance_: Array1<f64>,
    n_features_in_: usize,
    n_components_: usize,
}

impl PCAImputer<Untrained> {
    /// Create a new PCAImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            max_iter: 100,
            tol: 1e-6,
            missing_values: f64::NAN,
            random_state: None,
        }
    }

    /// Set the number of principal components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
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

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}

impl Default for PCAImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for PCAImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for PCAImputer<Untrained> {
    type Fitted = PCAImputer<PCAImputerTrained>;

    #[allow(non_snake_case)]
    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if self.n_components > n_features {
            return Err(SklearsError::InvalidInput(
                "Number of components cannot exceed number of features".to_string(),
            ));
        }

        // Initialize missing values with column means
        let mut X_filled = self.initialize_missing_values(&X)?;

        // Iterative PCA imputation
        for _iter in 0..self.max_iter {
            let X_old = X_filled.clone();

            // Compute PCA on current estimate
            let mean = X_filled.mean_axis(Axis(0)).unwrap();
            let X_centered = &X_filled - &mean.clone().insert_axis(Axis(0));

            // Compute covariance matrix
            let cov = X_centered.t().dot(&X_centered) / (n_samples as f64 - 1.0);

            // Eigenvalue decomposition (simplified)
            let (eigenvalues, eigenvectors) = self.eigen_decomposition(&cov)?;

            // Select top components
            let components = eigenvectors.slice(s![.., ..self.n_components]).to_owned();
            let _explained_variance = eigenvalues.slice(s![..self.n_components]).to_owned();

            // Project and reconstruct
            let X_transformed = X_centered.dot(&components);
            let X_reconstructed =
                X_transformed.dot(&components.t()) + &mean.clone().insert_axis(Axis(0));

            // Update missing values with reconstructed values
            for i in 0..n_samples {
                for j in 0..n_features {
                    if self.is_missing(X[[i, j]]) {
                        X_filled[[i, j]] = X_reconstructed[[i, j]];
                    }
                }
            }

            // Check convergence
            let max_change = (&X_filled - &X_old)
                .mapv(|x| x.abs())
                .fold(0.0f64, |acc, &x| acc.max(x));
            if max_change < self.tol {
                break;
            }
        }

        // Final PCA computation
        let mean = X_filled.mean_axis(Axis(0)).unwrap();
        let X_centered = &X_filled - &mean.clone().insert_axis(Axis(0));
        let cov = X_centered.t().dot(&X_centered) / (n_samples as f64 - 1.0);
        let (eigenvalues, eigenvectors) = self.eigen_decomposition(&cov)?;

        let components = eigenvectors.slice(s![.., ..self.n_components]).to_owned();
        let explained_variance = eigenvalues.slice(s![..self.n_components]).to_owned();

        Ok(PCAImputer {
            state: PCAImputerTrained {
                components_: components.t().to_owned(), // Store as (n_components, n_features)
                mean_: mean,
                explained_variance_: explained_variance,
                n_features_in_: n_features,
                n_components_: self.n_components,
            },
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            missing_values: self.missing_values,
            random_state: self.random_state,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>> for PCAImputer<PCAImputerTrained> {
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

        // Iterative imputation using trained PCA
        for _iter in 0..self.max_iter {
            let X_old = X_imputed.clone();

            // Center the data
            let X_centered = &X_imputed - &self.state.mean_.clone().insert_axis(Axis(0));

            // Project to PCA space
            let X_transformed = X_centered.dot(&self.state.components_.t());

            // Reconstruct from PCA space
            let X_reconstructed = X_transformed.dot(&self.state.components_)
                + &self.state.mean_.clone().insert_axis(Axis(0));

            // Update only missing values
            for i in 0..n_samples {
                for j in 0..n_features {
                    if self.is_missing(X[[i, j]]) {
                        X_imputed[[i, j]] = X_reconstructed[[i, j]];
                    }
                }
            }

            // Check convergence
            let max_change = (&X_imputed - &X_old)
                .mapv(|x| x.abs())
                .fold(0.0f64, |acc, &x| acc.max(x));
            if max_change < self.tol {
                break;
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl PCAImputer<Untrained> {
    fn initialize_missing_values(&self, X: &Array2<f64>) -> SklResult<Array2<f64>> {
        let (n_samples, n_features) = X.dim();
        let mut X_filled = X.clone();

        // Initialize missing values with column means
        for j in 0..n_features {
            let column = X.column(j);
            let observed_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if observed_values.is_empty() {
                return Err(SklearsError::InvalidInput(format!(
                    "Column {} has no observed values",
                    j
                )));
            }

            let mean_value = observed_values.iter().sum::<f64>() / observed_values.len() as f64;

            for i in 0..n_samples {
                if self.is_missing(X_filled[[i, j]]) {
                    X_filled[[i, j]] = mean_value;
                }
            }
        }

        Ok(X_filled)
    }

    fn eigen_decomposition(&self, A: &Array2<f64>) -> SklResult<(Array1<f64>, Array2<f64>)> {
        let n = A.nrows();
        if n != A.ncols() {
            return Err(SklearsError::InvalidInput(
                "Matrix must be square".to_string(),
            ));
        }

        // For simplicity, use power iteration for dominant eigenvalue/eigenvector
        // In practice, you'd use a proper eigenvalue decomposition library

        let mut rng = Random::default();
        let mut eigenvalues = Array1::zeros(n);
        let mut eigenvectors = Array2::zeros((n, n));

        for k in 0..n.min(self.n_components) {
            // Random initialization
            let mut v = Array1::from_shape_fn(n, |_| rng.random_range(-1.0..1.0));

            // Power iteration
            for _iter in 0..100 {
                let v_new = A.dot(&v);
                let norm = v_new.mapv(|x| x * x).sum().sqrt();
                if norm > 0.0 {
                    v = v_new / norm;
                } else {
                    break;
                }
            }

            // Compute eigenvalue
            let Av = A.dot(&v);
            let eigenvalue = v.dot(&Av);

            eigenvalues[k] = eigenvalue;
            for i in 0..n {
                eigenvectors[[i, k]] = v[i];
            }
        }

        // Sort by eigenvalue (descending)
        let mut pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        let mut sorted_eigenvalues = Array1::zeros(n);
        let mut sorted_eigenvectors = Array2::zeros((n, n));

        for (new_idx, &(eigenval, old_idx)) in pairs.iter().enumerate() {
            sorted_eigenvalues[new_idx] = eigenval;
            for i in 0..n {
                sorted_eigenvectors[[i, new_idx]] = eigenvectors[[i, old_idx]];
            }
        }

        Ok((sorted_eigenvalues, sorted_eigenvectors))
    }
}

impl PCAImputer<PCAImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }

    /// Get the principal components
    pub fn components(&self) -> &Array2<f64> {
        &self.state.components_
    }

    /// Get the explained variance
    pub fn explained_variance(&self) -> &Array1<f64> {
        &self.state.explained_variance_
    }

    /// Get the mean
    pub fn mean(&self) -> &Array1<f64> {
        &self.state.mean_
    }
}
