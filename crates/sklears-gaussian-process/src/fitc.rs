//! Fully Independent Training Conditionals (FITC) approximation for Gaussian Processes
//!
//! FITC (Snelson & Ghahramani, 2006; see also Quiñonero-Candela & Rasmussen's
//! 2005 unifying view of sparse approximations) approximates a full Gaussian
//! Process posterior using a small set of `m` inducing points `Z` instead of
//! the full `n` training inputs. The key approximation is that, conditioned on
//! the inducing outputs `u = f(Z)`, the training outputs are taken to be
//! *independent*:
//!
//! ```text
//! p(f | u) ≈ N(K_nm K_mm⁻¹ u,  diag(K_nn - K_nm K_mm⁻¹ K_mn))
//! ```
//!
//! i.e. the exact conditional covariance `K_nn - K_nm K_mm⁻¹ K_mn` is replaced
//! by *its diagonal only* (this is what distinguishes FITC from the cruder
//! "Subset of Regressors"/deterministic training conditional approximation,
//! which drops that correction entirely). Marginalizing out `u` gives an
//! effective observation model
//!
//! ```text
//! y = f + ε,   Cov(y) = Q_nn + Λ,   Q_nn = K_nm K_mm⁻¹ K_mn,
//!                          Λ = diag(K_nn - Q_nn) + σ_n² I
//! ```
//!
//! Because `Λ` is diagonal and `Q_nn` has rank `m`, the Woodbury identity
//! turns the usual `O(n³)` GP computations into `O(n m²)`:
//!
//! ```text
//! B = K_mm + K_mn Λ⁻¹ K_nm                      (m x m)
//! β = B⁻¹ K_mn Λ⁻¹ y                             (m,)
//! ```
//!
//! and the predictive distribution at a test point `x*` (with `k_*m` the
//! kernel vector between `x*` and the inducing points) is
//!
//! ```text
//! E[f*]   = k_*m · β
//! Var[f*] = K** - k_*m^T K_mm⁻¹ k_*m + k_*m^T B⁻¹ k_*m
//! ```
//!
//! The `-k_*m^T K_mm⁻¹ k_*m + k_*m^T B⁻¹ k_*m` pairing is exactly
//! `-Q** + k_*m^T Σ_u K_mm⁻¹ ... ` reduced algebraically (`K_mm⁻¹ Σ_u K_mm⁻¹ = B⁻¹`
//! where `Σ_u = K_mm B⁻¹ K_mm` is the posterior covariance over the inducing
//! outputs `u`), which keeps everything expressed through Cholesky solves
//! against `K_mm` and `B` rather than explicit matrix inverses.
//!
//! A useful sanity check on this module's correctness (exercised by the tests
//! below): when the number of inducing points equals the number of training
//! points and they coincide with the training inputs, `Q_nn = K_nn` exactly,
//! `Λ` degenerates to a constant `σ_n² I`, and the whole construction reduces
//! algebraically to the predictions of an exact (non-sparse)
//! [`crate::gpr::GaussianProcessRegressor`].
use scirs2_core::ndarray::{Array1, Array2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
    types::Float,
};

use crate::kernels::Kernel;
use crate::utils::{robust_cholesky, triangular_solve};

// The sparse and FITC regressors share the exact same inducing-point
// placement strategies; re-exporting under the same local name means
// `lib.rs` can rename it to `FitcInducingPointInit` on the way out without
// this module having to duplicate the enum.
pub use crate::sparse_gpr::InducingPointInit;

/// Configuration for [`FitcGaussianProcessRegressor`].
#[derive(Debug, Clone)]
pub struct FitcGaussianProcessRegressorConfig {
    /// Number of inducing points used to approximate the full GP posterior.
    pub n_inducing: usize,
    /// Strategy used to place the inducing points.
    pub inducing_init: InducingPointInit,
    /// Observation noise variance `σ_n²`, folded into the FITC diagonal
    /// correction `Λ = diag(K_nn - Q_nn) + σ_n² I`.
    pub noise_variance: f64,
    /// Extra diagonal jitter added to `K_mm` before its Cholesky
    /// factorization, on top of the adaptive jitter `robust_cholesky`
    /// applies automatically if that turns out not to be enough.
    pub jitter: f64,
    /// Whether to retain a copy of the training inputs on the fitted model.
    pub copy_x_train: bool,
    /// Random seed controlling stochastic inducing-point initialization.
    pub random_state: Option<u64>,
}

impl Default for FitcGaussianProcessRegressorConfig {
    fn default() -> Self {
        Self {
            n_inducing: 10,
            inducing_init: InducingPointInit::Kmeans,
            noise_variance: 1e-6,
            jitter: 1e-6,
            copy_x_train: true,
            random_state: None,
        }
    }
}

/// FITC (Fully Independent Training Conditional) sparse Gaussian Process
/// regressor.
///
/// # Examples
///
/// ```ignore
/// let X = array![[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]];
/// let y = array![1.0, 4.0, 9.0, 16.0, 25.0, 36.0];
///
/// let kernel = RBF::new(1.0);
/// let fitc = FitcGaussianProcessRegressor::new()
///     .kernel(Box::new(kernel))
///     .n_inducing(3);
/// let fitted = fitc.fit(&X, &y).expect("fit should succeed with valid training data");
/// let (mean, std) = fitted.predict_with_std(&X).expect("predict_with_std should succeed on trained model");
/// ```
#[derive(Debug, Clone)]
pub struct FitcGaussianProcessRegressor<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    config: FitcGaussianProcessRegressorConfig,
}

/// Trained state for [`FitcGaussianProcessRegressor`].
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct FitcGprTrained {
    /// Training inputs (retained only if `copy_x_train` was set).
    pub X_train: Option<Array2<f64>>,
    /// Training targets.
    pub y_train: Array1<f64>,
    /// Inducing point locations.
    pub Z: Array2<f64>,
    /// Kernel function (with whatever hyperparameters it had at `fit` time).
    pub kernel: Box<dyn Kernel>,
    /// Cholesky factor of the (jittered) inducing-inducing kernel matrix `K_mm`.
    pub L_mm: Array2<f64>,
    /// Cholesky factor of `B = K_mm + K_mn Λ⁻¹ K_nm`.
    pub L_b: Array2<f64>,
    /// `β = B⁻¹ K_mn Λ⁻¹ y`; the only quantity needed (together with `Z` and
    /// the kernel) to compute the predictive mean at any test point.
    pub beta: Array1<f64>,
    /// Observation noise variance used during fitting.
    pub noise_variance: f64,
    /// FITC log marginal likelihood at the fitted hyperparameters.
    pub log_marginal_likelihood_value: f64,
}

impl FitcGaussianProcessRegressor<Untrained> {
    /// Create a new `FitcGaussianProcessRegressor` instance.
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            config: FitcGaussianProcessRegressorConfig::default(),
        }
    }

    /// Set the kernel function.
    pub fn kernel(mut self, kernel: Box<dyn Kernel>) -> Self {
        self.kernel = Some(kernel);
        self
    }

    /// Set the number of inducing points.
    pub fn n_inducing(mut self, n_inducing: usize) -> Self {
        self.config.n_inducing = n_inducing;
        self
    }

    /// Set the inducing point initialization method.
    pub fn inducing_init(mut self, inducing_init: InducingPointInit) -> Self {
        self.config.inducing_init = inducing_init;
        self
    }

    /// Set the observation noise variance `σ_n²`.
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.config.noise_variance = noise_variance;
        self
    }

    /// Set the diagonal jitter added to `K_mm` before factorization.
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.config.jitter = jitter;
        self
    }

    /// Set whether to copy `X` during training.
    pub fn copy_x_train(mut self, copy_x_train: bool) -> Self {
        self.config.copy_x_train = copy_x_train;
        self
    }

    /// Set the random state controlling inducing point initialization.
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Estimator for FitcGaussianProcessRegressor<Untrained> {
    type Config = FitcGaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for FitcGaussianProcessRegressor<FitcGprTrained> {
    type Config = FitcGaussianProcessRegressorConfig;
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
impl Fit<Array2<f64>, Array1<f64>> for FitcGaussianProcessRegressor<Untrained> {
    type Fitted = FitcGaussianProcessRegressor<FitcGprTrained>;

    fn fit(self, X: &Array2<f64>, y: &Array1<f64>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        let n = X.nrows();
        if n == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit FitcGaussianProcessRegressor on empty data".to_string(),
            ));
        }

        let kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        let n_inducing = self.config.n_inducing.clamp(1, n);

        // 1. Inducing point locations.
        let Z = match self.config.inducing_init {
            InducingPointInit::Random => crate::utils::random_inducing_points(
                &X.view(),
                n_inducing,
                self.config.random_state,
            )?,
            InducingPointInit::Uniform => crate::utils::uniform_inducing_points(
                &X.view(),
                n_inducing,
                self.config.random_state,
            )?,
            InducingPointInit::Kmeans => crate::utils::kmeans_inducing_points(
                &X.view(),
                n_inducing,
                self.config.random_state,
            )?,
        };
        let m = Z.nrows();

        // 2. K_mm (inducing-inducing), jittered, and its Cholesky factor.
        let mut Kmm = kernel.compute_kernel_matrix(&Z, None)?;
        for i in 0..m {
            Kmm[[i, i]] += self.config.jitter;
        }
        let L_mm = robust_cholesky(&Kmm)?;

        // 3. K_nm (data-inducing).
        let Knm = kernel.compute_kernel_matrix(X, Some(&Z))?;

        // 4. Q_nn diagonal = diag(K_nm K_mm⁻¹ K_mn), in O(n m²) without ever
        //    forming the full n x n matrix Q_nn. For each row i we solve
        //    K_mm w_i = k_i once (a full Cholesky-based solve, i.e. a single
        //    application of K_mm⁻¹), then take Q_nn[i] = k_i . w_i. Note this
        //    must dot the *original* k_i against w_i (not w_i against
        //    itself) to get a single power of K_mm⁻¹: w_i . w_i would give
        //    k_i^T K_mm⁻² k_i, which is a different (wrong) quantity.
        let mut Q_nn_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let k_i = Knm.row(i).to_owned();
            let w_i = triangular_solve(&L_mm, &k_i)?;
            Q_nn_diag[i] = k_i.dot(&w_i);
        }

        // 5. FITC diagonal correction: Λ = diag(K_nn - Q_nn) + σ_n².
        //    `K_nn - Q_nn` is the diagonal of a Schur complement of a PSD
        //    kernel matrix and is mathematically non-negative; floating
        //    point error can push it slightly below zero when an inducing
        //    point coincides with (or sits very close to) a training point,
        //    so it is clamped at zero before adding the observation noise.
        let mut Lambda_diag = Array1::<f64>::zeros(n);
        for i in 0..n {
            let x_i = X.row(i);
            let k_ii = kernel.kernel(&x_i, &x_i);
            let diag_correction = (k_ii - Q_nn_diag[i]).max(0.0);
            Lambda_diag[i] = diag_correction + self.config.noise_variance;
        }
        let inv_lambda = Lambda_diag.mapv(|v| 1.0 / v);

        // 6. B = K_mm + K_mn Λ⁻¹ K_nm  (m x m),  b = K_mn Λ⁻¹ y  (m,)
        let inv_lambda_col = inv_lambda.view().insert_axis(Axis(1));
        let Knm_scaled = &Knm * &inv_lambda_col;
        let B = Kmm + Knm.t().dot(&Knm_scaled);
        let y_scaled = y * &inv_lambda;
        let b = Knm.t().dot(&y_scaled);

        // 7. β = B⁻¹ b via the Cholesky factor of B.
        let L_b = robust_cholesky(&B)?;
        let beta = triangular_solve(&L_b, &b)?;

        // 8. FITC log marginal likelihood in closed form (matrix determinant
        //    lemma: log|Q_nn + Λ| = log|Λ| + log|B| - log|K_mm|), see module
        //    docs for the full derivation.
        let quad = {
            let mut s = 0.0;
            for i in 0..n {
                s += y[i] * y[i] * inv_lambda[i];
            }
            s - b.dot(&beta)
        };
        let log_det_lambda: f64 = Lambda_diag.mapv(|v| v.ln()).sum();
        let log_det_b = 2.0 * L_b.diag().mapv(|v| v.ln()).sum();
        let log_det_kmm = 2.0 * L_mm.diag().mapv(|v| v.ln()).sum();
        let log_marginal_likelihood_value = -0.5
            * (quad + log_det_lambda + log_det_b - log_det_kmm
                + n as f64 * (2.0 * std::f64::consts::PI).ln());

        let X_train = if self.config.copy_x_train {
            Some(X.to_owned())
        } else {
            None
        };

        Ok(FitcGaussianProcessRegressor {
            state: FitcGprTrained {
                X_train,
                y_train: y.to_owned(),
                Z,
                kernel,
                L_mm,
                L_b,
                beta,
                noise_variance: self.config.noise_variance,
                log_marginal_likelihood_value,
            },
            kernel: None,
            config: self.config,
        })
    }
}

#[allow(non_snake_case)]
impl Predict<Array2<f64>, Array1<f64>> for FitcGaussianProcessRegressor<FitcGprTrained> {
    fn predict(&self, X: &Array2<f64>) -> SklResult<Array1<f64>> {
        let (mean, _) = self.predict_with_std(X)?;
        Ok(mean)
    }
}

#[allow(non_snake_case)]
impl FitcGaussianProcessRegressor<FitcGprTrained> {
    /// Predict with uncertainty estimates.
    ///
    /// Mean: `k_*m · β`. Variance: `K** - k_*m^T K_mm⁻¹ k_*m + k_*m^T B⁻¹
    /// k_*m`, clamped at zero to absorb floating point noise (the exact
    /// quantity is a variance and therefore mathematically non-negative).
    pub fn predict_with_std(&self, X: &Array2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let K_star_m = self
            .state
            .kernel
            .compute_kernel_matrix(X, Some(&self.state.Z))?;
        let n_test = X.nrows();

        let mut mean = Array1::<f64>::zeros(n_test);
        let mut variance = Array1::<f64>::zeros(n_test);

        for i in 0..n_test {
            let k_star_i = K_star_m.row(i).to_owned();
            let x_i = X.row(i);
            let k_star_star = self.state.kernel.kernel(&x_i, &x_i);

            mean[i] = k_star_i.dot(&self.state.beta);

            // Q** = k_*m^T K_mm⁻¹ k_*m via the Cholesky factor of K_mm.
            let w_star = triangular_solve(&self.state.L_mm, &k_star_i)?;
            let q_star = k_star_i.dot(&w_star);

            // k_*m^T B⁻¹ k_*m via the Cholesky factor of B.
            let v_star = triangular_solve(&self.state.L_b, &k_star_i)?;
            let extra = k_star_i.dot(&v_star);

            variance[i] = (k_star_star - q_star + extra).max(0.0);
        }

        let std = variance.mapv(f64::sqrt);
        Ok((mean, std))
    }

    /// Get the FITC log marginal likelihood.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self.state.log_marginal_likelihood_value
    }

    /// Get the inducing points.
    pub fn inducing_points(&self) -> &Array2<f64> {
        &self.state.Z
    }
}

impl Default for FitcGaussianProcessRegressor<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::gpr::GaussianProcessRegressor;
    use crate::kernels::RBF;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    fn toy_1d_data() -> (Array2<f64>, Array1<f64>) {
        let X = array![
            [0.0],
            [0.5],
            [1.0],
            [1.5],
            [2.0],
            [2.5],
            [3.0],
            [3.5],
            [4.0],
            [4.5],
            [5.0],
            [5.5],
            [6.0],
            [6.5],
            [7.0]
        ];
        let y = X.column(0).mapv(|x: f64| x.sin() + 0.1 * (x * 3.7).cos());
        (X, y)
    }

    #[test]
    fn test_fitc_predictive_variance_nonnegative() {
        let (X, y) = toy_1d_data();
        let fitted = FitcGaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(1.0)))
            .n_inducing(5)
            .noise_variance(1e-3)
            .fit(&X, &y)
            .expect("FITC fit should succeed on well-conditioned toy data");

        let X_test = array![[0.25], [3.3], [7.0], [10.0], [-3.0]];
        let (_, std) = fitted
            .predict_with_std(&X_test)
            .expect("FITC predict_with_std should succeed after fitting");

        for &s in std.iter() {
            assert!(s.is_finite(), "predictive std must be finite");
            assert!(s >= 0.0, "predictive std must be non-negative");
        }
        // Extrapolating far outside the training range should not be more
        // confident than interpolating right next to a training point.
        assert!(
            std[3] >= std[0],
            "extrapolation std ({}) should not be smaller than interpolation std ({})",
            std[3],
            std[0]
        );
    }

    /// The key correctness check for FITC: when every training point is used
    /// as an inducing point, the sparse approximation has no information loss
    /// (`Q_nn = K_nn` exactly) and must reduce to the predictions of an exact
    /// `GaussianProcessRegressor` using the same kernel and noise level.
    #[test]
    fn test_fitc_approaches_full_gpr_as_inducing_grows() {
        let (X, y) = toy_1d_data();
        let n = X.nrows();
        let noise_variance = 1e-3;

        let full_gpr = GaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(1.0)))
            .alpha(noise_variance)
            .fit(&X, &y)
            .expect("full GPR fit should succeed on well-conditioned toy data");

        let fitc = FitcGaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(1.0)))
            .n_inducing(n)
            .noise_variance(noise_variance)
            .jitter(1e-10)
            .fit(&X, &y)
            .expect("FITC fit should succeed on well-conditioned toy data");

        let X_test = array![[0.25], [2.2], [3.9], [5.1], [6.6]];
        let (mean_full, std_full) = full_gpr
            .predict_with_std(&X_test)
            .expect("full GPR predict_with_std should succeed");
        let (mean_fitc, std_fitc) = fitc
            .predict_with_std(&X_test)
            .expect("FITC predict_with_std should succeed");

        for i in 0..X_test.nrows() {
            assert_abs_diff_eq!(mean_full[i], mean_fitc[i], epsilon = 1e-3);
            assert_abs_diff_eq!(std_full[i], std_fitc[i], epsilon = 1e-3);
        }
    }

    #[test]
    fn test_fitc_builder_defaults() {
        let cfg = FitcGaussianProcessRegressorConfig::default();
        assert_eq!(cfg.n_inducing, 10);
        assert!(cfg.noise_variance > 0.0);
        assert!(cfg.jitter > 0.0);
    }

    #[test]
    fn test_fitc_requires_kernel() {
        let (X, y) = toy_1d_data();
        let result = FitcGaussianProcessRegressor::new()
            .n_inducing(3)
            .fit(&X, &y);
        assert!(result.is_err());
    }

    #[test]
    fn test_fitc_rejects_mismatched_shapes() {
        let (X, _) = toy_1d_data();
        let y_short = array![1.0, 2.0, 3.0];
        let result = FitcGaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(1.0)))
            .fit(&X, &y_short);
        assert!(result.is_err());
    }

    #[test]
    fn test_fitc_log_marginal_likelihood_is_finite() {
        let (X, y) = toy_1d_data();
        let fitted = FitcGaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(1.0)))
            .n_inducing(6)
            .noise_variance(1e-3)
            .fit(&X, &y)
            .expect("FITC fit should succeed on well-conditioned toy data");

        assert!(fitted.log_marginal_likelihood().is_finite());
        assert_eq!(fitted.inducing_points().nrows(), 6);
    }
}
