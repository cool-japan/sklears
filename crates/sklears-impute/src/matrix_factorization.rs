//! Matrix Factorization Imputer
//!
//! Imputation using matrix factorization techniques (SVD-based approach).
//! This method finds low-rank approximations of the data matrix to impute missing values.

use crate::utilities::solve_linear_system;
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::{Random};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Matrix Factorization Imputer
///
/// Imputation using matrix factorization techniques (SVD-based approach).
/// This method finds low-rank approximations of the data matrix to impute missing values.
///
/// # Parameters
///
/// * `n_components` - Number of components to use in the factorization
/// * `max_iter` - Maximum number of iterations for the optimization
/// * `tol` - Tolerance for convergence
/// * `regularization` - L2 regularization parameter
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```
/// use sklears_impute::MatrixFactorizationImputer;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let X = array![[1.0, 2.0, 3.0], [f64::NAN, 3.0, 4.0], [7.0, f64::NAN, 6.0]];
///
/// let imputer = MatrixFactorizationImputer::new()
///     .n_components(2)
///     .max_iter(100);
/// let fitted = imputer.fit(&X.view(), &()).unwrap();
/// let X_imputed = fitted.transform(&X.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MatrixFactorizationImputer<S = Untrained> {
    state: S,
    n_components: usize,
    max_iter: usize,
    tol: f64,
    regularization: f64,
    random_state: Option<u64>,
    missing_values: f64,
}

/// Trained state for MatrixFactorizationImputer
#[derive(Debug, Clone)]
pub struct MatrixFactorizationImputerTrained {
    U: Array2<f64>,
    V: Array2<f64>,
    mean_: Array1<f64>,
    n_features_in_: usize,
}

impl MatrixFactorizationImputer<Untrained> {
    /// Create a new MatrixFactorizationImputer instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 5,
            max_iter: 100,
            tol: 1e-4,
            regularization: 0.01,
            random_state: None,
            missing_values: f64::NAN,
        }
    }

    /// Set the number of components
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

    /// Set the regularization parameter
    pub fn regularization(mut self, regularization: f64) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set the random state for reproducibility
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the missing values placeholder
    pub fn missing_values(mut self, missing_values: f64) -> Self {
        self.missing_values = missing_values;
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

impl Default for MatrixFactorizationImputer<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MatrixFactorizationImputer<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MatrixFactorizationImputer<Untrained> {
    type Fitted = MatrixFactorizationImputer<MatrixFactorizationImputerTrained>;

    fn fit(self, X: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let X = X.mapv(|x| x);
        let (n_samples, n_features) = X.dim();

        if self.n_components > n_features.min(n_samples) {
            return Err(SklearsError::InvalidInput(format!(
                "n_components {} cannot be larger than min(n_samples, n_features) = {}",
                self.n_components,
                n_features.min(n_samples)
            )));
        }

        // Calculate column means for features with observed values
        let mut mean_ = Array1::zeros(n_features);
        for j in 0..n_features {
            let column = X.column(j);
            let valid_values: Vec<f64> = column
                .iter()
                .filter(|&&x| !self.is_missing(x))
                .cloned()
                .collect();

            if !valid_values.is_empty() {
                mean_[j] = valid_values.iter().sum::<f64>() / valid_values.len() as f64;
            }
        }

        // Initialize X with mean imputation
        let mut X_filled = X.clone();
        for i in 0..n_samples {
            for j in 0..n_features {
                if self.is_missing(X_filled[[i, j]]) {
                    X_filled[[i, j]] = mean_[j];
                }
            }
        }

        // Center the data
        for j in 0..n_features {
            for i in 0..n_samples {
                X_filled[[i, j]] -= mean_[j];
            }
        }

        // Initialize U and V randomly
        let mut rng = Random::default().seed(self.random_state.unwrap_or(42));
        let mut U = Array2::zeros((n_samples, self.n_components));
        let mut V = Array2::zeros((self.n_components, n_features));

        // Random initialization
        for i in 0..n_samples {
            for k in 0..self.n_components {
                U[[i, k]] = rng.normal(0.0, 0.1);
            }
        }

        for k in 0..self.n_components {
            for j in 0..n_features {
                V[[k, j]] = rng.normal(0.0, 0.1);
            }
        }

        // Create mask for observed values
        let mut mask = Array2::from_elem((n_samples, n_features), false);
        for i in 0..n_samples {
            for j in 0..n_features {
                mask[[i, j]] = !self.is_missing(X[[i, j]]);
            }
        }

        // Alternating least squares optimization
        let mut prev_loss = f64::INFINITY;

        for _iter in 0..self.max_iter {
            // Update V given U
            for j in 0..n_features {
                let mut AtA = Array2::zeros((self.n_components, self.n_components));
                let mut Atb = Array1::zeros(self.n_components);

                for i in 0..n_samples {
                    if mask[[i, j]] {
                        let u_i = U.row(i);
                        for k1 in 0..self.n_components {
                            Atb[k1] += u_i[k1] * (X[[i, j]] - mean_[j]);
                            for k2 in 0..self.n_components {
                                AtA[[k1, k2]] += u_i[k1] * u_i[k2];
                            }
                        }
                    }
                }

                // Add regularization
                for k in 0..self.n_components {
                    AtA[[k, k]] += self.regularization;
                }

                // Solve linear system (simplified approach)
                if let Ok(v_j) = solve_linear_system(&AtA, &Atb) {
                    for k in 0..self.n_components {
                        V[[k, j]] = v_j[k];
                    }
                }
            }

            // Update U given V
            for i in 0..n_samples {
                let mut AtA = Array2::zeros((self.n_components, self.n_components));
                let mut Atb = Array1::zeros(self.n_components);

                for j in 0..n_features {
                    if mask[[i, j]] {
                        let v_j = V.column(j);
                        for k1 in 0..self.n_components {
                            Atb[k1] += v_j[k1] * (X[[i, j]] - mean_[j]);
                            for k2 in 0..self.n_components {
                                AtA[[k1, k2]] += v_j[k1] * v_j[k2];
                            }
                        }
                    }
                }

                // Add regularization
                for k in 0..self.n_components {
                    AtA[[k, k]] += self.regularization;
                }

                // Solve linear system
                if let Ok(u_i) = solve_linear_system(&AtA, &Atb) {
                    for k in 0..self.n_components {
                        U[[i, k]] = u_i[k];
                    }
                }
            }

            // Compute loss
            let mut loss = 0.0;
            let mut count = 0;
            for i in 0..n_samples {
                for j in 0..n_features {
                    if mask[[i, j]] {
                        let prediction = U.row(i).dot(&V.column(j)) + mean_[j];
                        let residual = X[[i, j]] - prediction;
                        loss += residual * residual;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                loss /= count as f64;
            }

            // Add regularization to loss
            let u_reg: f64 = U.iter().map(|&x| x * x).sum();
            let v_reg: f64 = V.iter().map(|&x| x * x).sum();
            loss += self.regularization * (u_reg + v_reg);

            // Check convergence
            if (prev_loss - loss).abs() < self.tol {
                break;
            }
            prev_loss = loss;
        }

        Ok(MatrixFactorizationImputer {
            state: MatrixFactorizationImputerTrained {
                U,
                V,
                mean_,
                n_features_in_: n_features,
            },
            n_components: self.n_components,
            max_iter: self.max_iter,
            tol: self.tol,
            regularization: self.regularization,
            random_state: self.random_state,
            missing_values: self.missing_values,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<Float>>
    for MatrixFactorizationImputer<MatrixFactorizationImputerTrained>
{
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

        // For each sample, project onto the learned subspace
        for i in 0..n_samples {
            // Find the projection of this sample onto the learned subspace
            // by solving for u_i that minimizes ||x_i - u_i V||^2 for observed values

            let mut AtA = Array2::zeros((self.n_components, self.n_components));
            let mut Atb = Array1::zeros(self.n_components);
            let mut has_observed = false;

            for j in 0..n_features {
                if !self.is_missing(X[[i, j]]) {
                    has_observed = true;
                    let v_j = self.state.V.column(j);
                    for k1 in 0..self.n_components {
                        Atb[k1] += v_j[k1] * (X[[i, j]] - self.state.mean_[j]);
                        for k2 in 0..self.n_components {
                            AtA[[k1, k2]] += v_j[k1] * v_j[k2];
                        }
                    }
                }
            }

            if has_observed {
                // Add small regularization for numerical stability
                for k in 0..self.n_components {
                    AtA[[k, k]] += 1e-6;
                }

                if let Ok(u_i) = solve_linear_system(&AtA, &Atb) {
                    // Impute missing values using the projection
                    for j in 0..n_features {
                        if self.is_missing(X[[i, j]]) {
                            let prediction = u_i.dot(&self.state.V.column(j)) + self.state.mean_[j];
                            X_imputed[[i, j]] = prediction;
                        }
                    }
                } else {
                    // Fallback to mean imputation if linear system fails
                    for j in 0..n_features {
                        if self.is_missing(X[[i, j]]) {
                            X_imputed[[i, j]] = self.state.mean_[j];
                        }
                    }
                }
            } else {
                // No observed values, use mean imputation
                for j in 0..n_features {
                    if self.is_missing(X[[i, j]]) {
                        X_imputed[[i, j]] = self.state.mean_[j];
                    }
                }
            }
        }

        Ok(X_imputed.mapv(|x| x as Float))
    }
}

impl MatrixFactorizationImputer<MatrixFactorizationImputerTrained> {
    fn is_missing(&self, value: f64) -> bool {
        if self.missing_values.is_nan() {
            value.is_nan()
        } else {
            (value - self.missing_values).abs() < f64::EPSILON
        }
    }
}