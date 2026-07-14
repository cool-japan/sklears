//! Variational Gaussian Process classification
//!
//! This module implements Stochastic Variational Gaussian Process
//! Classification (SVGP-C; Hensman, Matthews & Ghahramani, 2015) with a
//! Bernoulli-logit likelihood. It follows the same sparse-variational
//! machinery as [`crate::regression::VariationalSparseGaussianProcessRegressor`]
//! (a set of inducing points `Z` with a variational posterior `q(u) = N(m, S)`
//! over the inducing outputs `u = f(Z)`), but swaps the Gaussian likelihood
//! for a non-conjugate Bernoulli one.
//!
//! # Model
//!
//! ```text
//! p(u)       = N(0, K_mm)
//! q(u)       = N(m, S)                                    (variational posterior)
//! p(f_i | u) = N(k_i^T K_mm⁻¹ u,  K_ii - k_i^T K_mm⁻¹ k_i) (GP conditional)
//! p(y_i|f_i) = Bernoulli(sigmoid(f_i))
//! ```
//!
//! Marginalizing `u` out of `p(f_i | u)` under `q(u)` gives
//! `q(f_i) = N(mu_i, sigma_i²)` with, writing `w_i = K_mm⁻¹ k_i`:
//!
//! ```text
//! mu_i     = w_i · m
//! sigma_i² = K_ii - k_i · w_i + w_i^T S w_i
//! ```
//!
//! The evidence lower bound maximized during training is
//!
//! ```text
//! ELBO = Σ_i E_{q(f_i)}[log p(y_i | f_i)]  -  KL(q(u) || p(u))
//! ```
//!
//! `KL(q(u) || p(u))` is the standard multivariate-Gaussian KL divergence
//! (computed via [`crate::utils::kl_divergence_gaussian`]). The likelihood
//! term is non-conjugate (`f_i` enters through a sigmoid), so its expectation
//! and the gradients needed to optimize `m` and `S` are estimated with
//! 10-point Gauss-Hermite quadrature, using two classical identities for
//! Gaussian expectations (Bonnet's and Price's theorems):
//!
//! ```text
//! d/dmu    E_{N(mu,sigma²)}[g(f)] = E[g'(f)]
//! d/dsigma² E_{N(mu,sigma²)}[g(f)] = 0.5 * E[g''(f)]
//! ```
//!
//! with `g(f) = log p(y|f)`, so `g'(f) = y - sigmoid(f)` and
//! `g''(f) = -sigmoid'(f)` (both already available from
//! [`crate::classification::sigmoid`] / [`crate::classification::sigmoid_derivative`]).
//! Chaining these through `mu_i = w_i · m` and `sigma_i² = ... + w_i^T S w_i`
//! gives closed-form (quadrature-estimated) gradients of the ELBO with
//! respect to `m` and `S`, which are optimized by plain gradient ascent
//! (mirroring `VariationalSparseGaussianProcessRegressor`'s optimizer). Kernel
//! hyperparameters are, optionally, also nudged by gradient ascent using a
//! finite-difference estimate of the ELBO gradient (the same technique
//! [`crate::kernel_trait::Kernel::compute_kernel_gradient`] and
//! [`crate::marginal_likelihood::MarginalLikelihoodOptimizer`] already use
//! elsewhere in this crate). Inducing point *locations* `Z` are fixed after
//! initialization, exactly as in `VariationalSparseGaussianProcessRegressor`.
use std::collections::BTreeSet;
use std::f64::consts::PI;

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, PredictProba, Untrained},
};

use crate::classification::{sigmoid, sigmoid_derivative};
use crate::kernels::Kernel;
use crate::sparse_gpr::InducingPointInit;
use crate::utils;

/// Physicists'-convention 10-point Gauss-Hermite nodes: for
/// `∫ e^{-x²} h(x) dx ≈ Σ weight_k * h(node_k)`.
const GH10_NODES: [f64; 10] = [
    -3.436_159_118_837_74,
    -2.532_731_674_232_79,
    -1.756_683_649_299_88,
    -1.036_610_829_789_51,
    -0.342_901_327_223_705,
    0.342_901_327_223_705,
    1.036_610_829_789_51,
    1.756_683_649_299_88,
    2.532_731_674_232_79,
    3.436_159_118_837_74,
];
/// Weights matching [`GH10_NODES`].
const GH10_WEIGHTS: [f64; 10] = [
    7.640_432_855_232_62e-6,
    1.343_645_746_781e-3,
    3.387_439_445_548_1e-2,
    2.401_386_110_823_14e-1,
    6.108_626_337_353_25e-1,
    6.108_626_337_353_25e-1,
    2.401_386_110_823_14e-1,
    3.387_439_445_548_1e-2,
    1.343_645_746_781e-3,
    7.640_432_855_232_62e-6,
];

/// Generous cap on the Euclidean norm of the raw `m` (and kernel-parameter)
/// gradients before they are scaled by the learning rate. `q(f_i)`'s mean and
/// variance can be far from calibrated early in training (`S` starts at the
/// identity), which can otherwise produce a very large first step; clipping
/// is a standard, correctness-preserving optimization safeguard (it does not
/// change any fixed point of the ascent, only how quickly early steps
/// approach one).
const GRADIENT_CLIP_NORM: f64 = 25.0;

/// Tighter Frobenius-norm cap for the raw `S` gradient. The KL term
/// contributes a `-0.5 S⁻¹` component whose magnitude grows without bound as
/// `S` approaches singularity, which makes *plain* (non-natural) gradient
/// ascent directly on the covariance matrix prone to a runaway
/// shrink-and-invert feedback loop; a tighter clip than `m`'s keeps each step
/// small enough that the PSD repair below stays within reach.
const GRADIENT_CLIP_NORM_S: f64 = 5.0;

/// `S` is updated with `learning_rate * S_LEARNING_RATE_SCALE` rather than
/// the raw `learning_rate`: the covariance update is intrinsically more
/// sensitive than the mean update (small negative moves compound through
/// `S⁻¹` in the next iteration's KL gradient), so damping it relative to `m`
/// keeps the ascent stable without changing the (m, S) fixed point that
/// maximizes the ELBO.
const S_LEARNING_RATE_SCALE: f64 = 0.1;

/// Numerically stable softplus: `log(1 + exp(x))`.
fn softplus(x: f64) -> f64 {
    x.max(0.0) + (1.0 + (-x.abs()).exp()).ln()
}

/// Numerically stable log-sigmoid: `log(sigmoid(x))`.
fn log_sigmoid(x: f64) -> f64 {
    -softplus(-x)
}

/// Gauss-Hermite estimates of `E_q[log p(y|f)]`, `E_q[d/df log p(y|f)]` and
/// `E_q[d²/df² log p(y|f)]` under `q(f) = N(mu, sigma²)`, for the
/// Bernoulli-logit likelihood `p(y=1|f) = sigmoid(f)` (`y` given as `0.0` or
/// `1.0`). These three moments are exactly what both the ELBO value and its
/// gradients w.r.t. `mu` and `sigma²` need (see module docs).
fn gauss_hermite_bernoulli_moments(mu: f64, sigma: f64, y: f64) -> (f64, f64, f64) {
    let norm = 1.0 / PI.sqrt();
    let mut e_log_lik = 0.0;
    let mut e_dlog = 0.0;
    let mut e_d2log = 0.0;

    for k in 0..10 {
        let f = mu + std::f64::consts::SQRT_2 * sigma * GH10_NODES[k];
        let weight = GH10_WEIGHTS[k] * norm;

        let log_lik = y * log_sigmoid(f) + (1.0 - y) * log_sigmoid(-f);
        let s = sigmoid(f);
        let dlog = y - s;
        let d2log = -sigmoid_derivative(f);

        e_log_lik += weight * log_lik;
        e_dlog += weight * dlog;
        e_d2log += weight * d2log;
    }

    (e_log_lik, e_dlog, e_d2log)
}

/// Gauss-Hermite estimate of `E_q[sigmoid(f)]` under `q(f) = N(mu, sigma²)`,
/// used for a calibrated `predict_proba` (integrating the sigmoid over the
/// posterior rather than evaluating it only at the mean).
fn gauss_hermite_sigmoid_expectation(mu: f64, sigma: f64) -> f64 {
    let norm = 1.0 / PI.sqrt();
    let mut acc = 0.0;
    for k in 0..10 {
        let f = mu + std::f64::consts::SQRT_2 * sigma * GH10_NODES[k];
        acc += GH10_WEIGHTS[k] * norm * sigmoid(f);
    }
    acc
}

fn clip_vector_norm(v: Array1<f64>, max_norm: f64) -> Array1<f64> {
    let norm = v.dot(&v).sqrt();
    if norm > max_norm && norm > 0.0 {
        v * (max_norm / norm)
    } else {
        v
    }
}

#[allow(non_snake_case)]
fn clip_matrix_norm(M: Array2<f64>, max_norm: f64) -> Array2<f64> {
    let norm = M.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > max_norm && norm > 0.0 {
        M * (max_norm / norm)
    } else {
        M
    }
}

/// Configuration for [`VariationalGaussianProcessClassifier`].
#[derive(Debug, Clone)]
pub struct VariationalGpcConfig {
    /// Number of inducing points.
    pub n_inducing: usize,
    /// Strategy used to place the inducing points (fixed after init).
    pub inducing_init: InducingPointInit,
    /// Maximum number of gradient-ascent iterations on the ELBO.
    pub max_iter: usize,
    /// Learning rate for the (plain) gradient ascent updates.
    pub learning_rate: f64,
    /// Convergence tolerance on successive ELBO values.
    pub tol: f64,
    /// Diagonal jitter added to `K_mm` before its Cholesky factorization.
    pub jitter: f64,
    /// Whether to also optimize kernel hyperparameters (via a
    /// finite-difference ELBO gradient) alongside the variational
    /// parameters.
    pub optimize_kernel: bool,
    /// Random seed controlling stochastic inducing-point initialization.
    pub random_state: Option<u64>,
}

impl Default for VariationalGpcConfig {
    fn default() -> Self {
        Self {
            n_inducing: 10,
            inducing_init: InducingPointInit::Kmeans,
            max_iter: 200,
            learning_rate: 0.01,
            tol: 1e-5,
            jitter: 1e-6,
            optimize_kernel: true,
            random_state: None,
        }
    }
}

/// Sparse variational Gaussian Process classifier with a Bernoulli-logit
/// likelihood (SVGP-C).
///
/// # Examples
///
/// ```ignore
/// let X = array![[-2.0, -2.0], [-2.5, -1.5], [2.0, 2.0], [2.5, 1.5]];
/// let y = array![0, 0, 1, 1];
///
/// let kernel = RBF::new(1.5);
/// let vgpc = VariationalGaussianProcessClassifier::new()
///     .kernel(Box::new(kernel))
///     .n_inducing(4);
/// let fitted = vgpc.fit(&X.view(), &y.view()).expect("fit should succeed with valid training data");
/// let predictions = fitted.predict(&X.view()).expect("predict should succeed on trained model");
/// ```
#[derive(Debug, Clone)]
pub struct VariationalGaussianProcessClassifier<S = Untrained> {
    state: S,
    kernel: Option<Box<dyn Kernel>>,
    config: VariationalGpcConfig,
}

/// Trained state for [`VariationalGaussianProcessClassifier`].
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct VariationalGpcTrained {
    /// Inducing point locations.
    pub Z: Array2<f64>,
    /// Variational mean over the inducing outputs `u`.
    pub m: Array1<f64>,
    /// Variational covariance over the inducing outputs `u`.
    pub S: Array2<f64>,
    /// Sorted unique class labels seen during training; `classes[1]` is the
    /// "positive" class used by `predict_proba`.
    pub classes: Array1<i32>,
    /// Kernel function (with whatever hyperparameters it had at the end of
    /// fitting).
    pub kernel: Box<dyn Kernel>,
    /// Diagonal jitter used for `K_mm` during fitting (reused at predict
    /// time for consistency).
    pub jitter: f64,
    /// ELBO value recorded at every iteration.
    pub elbo_history: Vec<f64>,
    /// Final ELBO value.
    pub final_elbo: f64,
}

impl VariationalGaussianProcessClassifier<Untrained> {
    /// Create a new `VariationalGaussianProcessClassifier` instance.
    pub fn new() -> Self {
        Self {
            state: Untrained,
            kernel: None,
            config: VariationalGpcConfig::default(),
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

    /// Set the maximum number of gradient-ascent iterations.
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.config.max_iter = max_iter;
        self
    }

    /// Set the learning rate.
    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.config.learning_rate = learning_rate;
        self
    }

    /// Set the convergence tolerance.
    pub fn tol(mut self, tol: f64) -> Self {
        self.config.tol = tol;
        self
    }

    /// Set the diagonal jitter added to `K_mm`.
    pub fn jitter(mut self, jitter: f64) -> Self {
        self.config.jitter = jitter;
        self
    }

    /// Set whether kernel hyperparameters are optimized alongside `m`/`S`.
    pub fn optimize_kernel(mut self, optimize_kernel: bool) -> Self {
        self.config.optimize_kernel = optimize_kernel;
        self
    }

    /// Set the random state.
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.config.random_state = random_state;
        self
    }
}

impl Estimator for VariationalGaussianProcessClassifier<Untrained> {
    type Config = VariationalGpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for VariationalGaussianProcessClassifier<VariationalGpcTrained> {
    type Config = VariationalGpcConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

/// Compute `K_mm` (jittered, with its Cholesky factor consumed internally),
/// the cross kernel matrix between `X` and `Z`, and, for every row `i`,
/// `w_i = K_mm⁻¹ k_i` (stacked as the rows of `W`). Used both for training
/// (with `X` = training inputs) and prediction (with `X` = test inputs).
#[allow(non_snake_case)]
fn kernel_workspace(
    kernel: &dyn Kernel,
    X: &Array2<f64>,
    Z: &Array2<f64>,
    jitter: f64,
) -> SklResult<(Array2<f64>, Array2<f64>, Array2<f64>)> {
    let m_ind = Z.nrows();
    let mut Kmm = kernel.compute_kernel_matrix(Z, None)?;
    for i in 0..m_ind {
        Kmm[[i, i]] += jitter;
    }
    let L_mm = utils::robust_cholesky(&Kmm)?;
    let Knm = kernel.compute_kernel_matrix(X, Some(Z))?;

    let n = X.nrows();
    let mut W = Array2::<f64>::zeros((n, m_ind));
    for i in 0..n {
        let k_i = Knm.row(i).to_owned();
        let w_i = utils::triangular_solve(&L_mm, &k_i)?;
        W.row_mut(i).assign(&w_i);
    }

    Ok((Kmm, Knm, W))
}

/// ELBO value and its gradients w.r.t. `m` and `S`, for the Bernoulli-logit
/// SVGP model described in the module docs.
#[allow(non_snake_case)]
fn elbo_and_gradients(
    y: &Array1<f64>,
    Knm: &Array2<f64>,
    Kxx_diag: &Array1<f64>,
    W: &Array2<f64>,
    Kmm: &Array2<f64>,
    m: &Array1<f64>,
    S: &Array2<f64>,
) -> SklResult<(f64, Array1<f64>, Array2<f64>)> {
    let n = y.len();
    let m_ind = m.len();

    let mut data_fit = 0.0;
    let mut grad_m = Array1::<f64>::zeros(m_ind);
    let mut grad_S = Array2::<f64>::zeros((m_ind, m_ind));

    for i in 0..n {
        let k_i = Knm.row(i);
        let w_i = W.row(i).to_owned();

        // q(f_i) = N(mu_i, sigma_i^2); see module docs for the derivation.
        let mu_i = w_i.dot(m);
        let q_ii = k_i.dot(&w_i); // = k_i^T Kmm^{-1} k_i, a single inverse power
        let sw_i = S.dot(&w_i);
        let extra_var = w_i.dot(&sw_i);
        let sigma2_i = (Kxx_diag[i] - q_ii + extra_var).max(1e-12);
        let sigma_i = sigma2_i.sqrt();

        let (e_log_lik, e_dloglik_df, e_d2loglik_df2) =
            gauss_hermite_bernoulli_moments(mu_i, sigma_i, y[i]);

        data_fit += e_log_lik;

        // Chain rule: mu_i = w_i . m  =>  d mu_i/dm = w_i.
        grad_m = &grad_m + e_dloglik_df * &w_i;

        // sigma_i^2 = ... + w_i^T S w_i  =>  d sigma_i^2/dS = w_i w_i^T, and
        // (Price's theorem) d E[g(f)]/d sigma^2 = 0.5 * E[g''(f)].
        let outer_w = Array2::from_shape_fn((m_ind, m_ind), |(a, b)| w_i[a] * w_i[b]);
        grad_S = &grad_S + (0.5 * e_d2loglik_df2) * &outer_w;
    }

    let zero_mean = Array1::<f64>::zeros(m_ind);
    let kl = utils::kl_divergence_gaussian(m, S, &zero_mean, Kmm)?;

    let kmm_inv = utils::matrix_inverse(Kmm)?;
    let S_inv = utils::matrix_inverse(S)?;
    let kmm_inv_m = kmm_inv.dot(m);

    // d KL/dm = Kmm^{-1} m ; d KL/dS = 0.5 (Kmm^{-1} - S^{-1})
    grad_m = &grad_m - &kmm_inv_m;
    grad_S = &grad_S - 0.5 * (&kmm_inv - &S_inv);

    let elbo = data_fit - kl;

    Ok((elbo, grad_m, grad_S))
}

#[allow(non_snake_case)]
fn elbo_value_for_kernel(
    kernel: &dyn Kernel,
    X: &Array2<f64>,
    y: &Array1<f64>,
    Z: &Array2<f64>,
    m: &Array1<f64>,
    S: &Array2<f64>,
    jitter: f64,
) -> SklResult<f64> {
    let (Kmm, Knm, W) = kernel_workspace(kernel, X, Z, jitter)?;
    let Kxx_diag = X
        .axis_iter(Axis(0))
        .map(|x| kernel.kernel(&x, &x))
        .collect::<Array1<f64>>();
    let (elbo, _, _) = elbo_and_gradients(y, &Knm, &Kxx_diag, &W, &Kmm, m, S)?;
    Ok(elbo)
}

/// Finite-difference estimate of `d ELBO / d kernel_params`, using the same
/// technique already used elsewhere in this crate for kernel gradients (see
/// [`crate::kernel_trait::Kernel::compute_kernel_gradient`] and
/// [`crate::marginal_likelihood::MarginalLikelihoodOptimizer`]).
#[allow(non_snake_case)]
fn kernel_param_gradient_fd(
    kernel: &dyn Kernel,
    X: &Array2<f64>,
    y: &Array1<f64>,
    Z: &Array2<f64>,
    m: &Array1<f64>,
    S: &Array2<f64>,
    jitter: f64,
) -> SklResult<Array1<f64>> {
    let params = kernel.get_params();
    if params.is_empty() {
        return Ok(Array1::zeros(0));
    }

    let base_elbo = elbo_value_for_kernel(kernel, X, y, Z, m, S, jitter)?;

    let mut grad = Array1::<f64>::zeros(params.len());
    for i in 0..params.len() {
        let step = 1e-6 * params[i].abs().max(1.0);
        let mut perturbed = params.clone();
        perturbed[i] += step;

        let mut kernel_plus = kernel.clone_box();
        kernel_plus.set_params(&perturbed)?;
        let elbo_plus = elbo_value_for_kernel(kernel_plus.as_ref(), X, y, Z, m, S, jitter)?;

        grad[i] = (elbo_plus - base_elbo) / step;
    }

    Ok(grad)
}

#[allow(non_snake_case)]
impl Fit<ArrayView2<'_, f64>, ArrayView1<'_, i32>>
    for VariationalGaussianProcessClassifier<Untrained>
{
    type Fitted = VariationalGaussianProcessClassifier<VariationalGpcTrained>;

    fn fit(self, X: &ArrayView2<f64>, y: &ArrayView1<i32>) -> SklResult<Self::Fitted> {
        if X.nrows() != y.len() {
            return Err(SklearsError::InvalidInput(
                "X and y must have the same number of samples".to_string(),
            ));
        }
        if X.nrows() == 0 {
            return Err(SklearsError::InvalidInput(
                "Cannot fit VariationalGaussianProcessClassifier on empty data".to_string(),
            ));
        }

        let mut kernel = self
            .kernel
            .ok_or_else(|| SklearsError::InvalidInput("Kernel must be specified".to_string()))?;

        let mut classes_set: BTreeSet<i32> = BTreeSet::new();
        for &label in y.iter() {
            classes_set.insert(label);
        }
        if classes_set.len() != 2 {
            return Err(SklearsError::InvalidInput(
                "Variational GP classification requires exactly 2 classes".to_string(),
            ));
        }
        let classes: Vec<i32> = classes_set.into_iter().collect();
        let classes_arr = Array1::from(classes.clone());
        let y_bernoulli: Array1<f64> = y.mapv(|label| if label == classes[0] { 0.0 } else { 1.0 });

        let n = X.nrows();
        let n_inducing = self.config.n_inducing.clamp(1, n);
        let X_owned = X.to_owned();

        let Z = match self.config.inducing_init {
            InducingPointInit::Random => {
                utils::random_inducing_points(X, n_inducing, self.config.random_state)?
            }
            InducingPointInit::Uniform => {
                utils::uniform_inducing_points(X, n_inducing, self.config.random_state)?
            }
            InducingPointInit::Kmeans => {
                utils::kmeans_inducing_points(X, n_inducing, self.config.random_state)?
            }
        };
        let m_ind = Z.nrows();

        let Kxx_diag = X_owned
            .axis_iter(Axis(0))
            .map(|x| kernel.kernel(&x, &x))
            .collect::<Array1<f64>>();

        let mut m = Array1::<f64>::zeros(m_ind);
        let mut S = Array2::<f64>::eye(m_ind);

        let mut elbo_history: Vec<f64> = Vec::with_capacity(self.config.max_iter);

        for _iter in 0..self.config.max_iter {
            let (Kmm, Knm, W) =
                kernel_workspace(kernel.as_ref(), &X_owned, &Z, self.config.jitter)?;

            let (elbo, grad_m, grad_S) =
                elbo_and_gradients(&y_bernoulli, &Knm, &Kxx_diag, &W, &Kmm, &m, &S)?;

            let converged = matches!(
                elbo_history.last(),
                Some(&prev) if (elbo - prev).abs() < self.config.tol
            );
            elbo_history.push(elbo);
            if converged {
                break;
            }

            if self.config.optimize_kernel {
                let grad_kernel = kernel_param_gradient_fd(
                    kernel.as_ref(),
                    &X_owned,
                    &y_bernoulli,
                    &Z,
                    &m,
                    &S,
                    self.config.jitter,
                )?;
                if !grad_kernel.is_empty() {
                    let clipped = clip_vector_norm(grad_kernel, GRADIENT_CLIP_NORM);
                    let mut params = kernel.get_params();
                    for (p, g) in params.iter_mut().zip(clipped.iter()) {
                        *p += self.config.learning_rate * g;
                        // Kernel hyperparameters (length scales, variances, ...)
                        // must stay strictly positive.
                        if *p <= 1e-8 {
                            *p = 1e-8;
                        }
                    }
                    kernel.set_params(&params)?;
                }
            }

            let grad_m_clipped = clip_vector_norm(grad_m, GRADIENT_CLIP_NORM);
            let grad_S_clipped = clip_matrix_norm(grad_S, GRADIENT_CLIP_NORM_S);

            m = &m + self.config.learning_rate * &grad_m_clipped;
            let S_updated =
                &S + (self.config.learning_rate * S_LEARNING_RATE_SCALE) * &grad_S_clipped;
            let S_sym = 0.5 * (&S_updated + &S_updated.t());
            // Repair to a strictly PSD matrix via `robust_cholesky`'s
            // escalating-jitter search (far more robust than a single fixed
            // regularization attempt), then reconstruct S = L L^T so the
            // result is *exactly* PSD by construction rather than merely
            // "hopefully regularized enough".
            let L_s_repair = utils::robust_cholesky(&S_sym)?;
            S = L_s_repair.dot(&L_s_repair.t());
        }

        let final_elbo = elbo_history.last().copied().unwrap_or(f64::NEG_INFINITY);

        Ok(VariationalGaussianProcessClassifier {
            state: VariationalGpcTrained {
                Z,
                m,
                S,
                classes: classes_arr,
                kernel,
                jitter: self.config.jitter,
                elbo_history,
                final_elbo,
            },
            kernel: None,
            config: self.config,
        })
    }
}

#[allow(non_snake_case)]
impl VariationalGaussianProcessClassifier<VariationalGpcTrained> {
    /// `q(f*)` mean/variance at test points, computed the same way as the
    /// per-datum `q(f_i)` used during training.
    fn latent_moments(&self, X: &ArrayView2<f64>) -> SklResult<(Array1<f64>, Array1<f64>)> {
        let X_owned = X.to_owned();
        let (_, K_star_m, W) = kernel_workspace(
            self.state.kernel.as_ref(),
            &X_owned,
            &self.state.Z,
            self.state.jitter,
        )?;

        let n_test = X.nrows();
        let mut mean = Array1::<f64>::zeros(n_test);
        let mut var = Array1::<f64>::zeros(n_test);

        for i in 0..n_test {
            let k_i = K_star_m.row(i);
            let w_i = W.row(i).to_owned();
            let x_i = X_owned.row(i);

            let mu_i = w_i.dot(&self.state.m);
            let q_ii = k_i.dot(&w_i);
            let k_ii = self.state.kernel.kernel(&x_i, &x_i);
            let sw_i = self.state.S.dot(&w_i);
            let extra = w_i.dot(&sw_i);

            mean[i] = mu_i;
            var[i] = (k_ii - q_ii + extra).max(1e-12);
        }

        Ok((mean, var))
    }
}

impl VariationalGaussianProcessClassifier<VariationalGpcTrained> {
    /// Get the ELBO history recorded during optimization.
    pub fn elbo_history(&self) -> &[f64] {
        &self.state.elbo_history
    }

    /// Get the final ELBO value.
    pub fn elbo(&self) -> f64 {
        self.state.final_elbo
    }

    /// Get the inducing points.
    pub fn inducing_points(&self) -> &Array2<f64> {
        &self.state.Z
    }

    /// Get the sorted unique class labels seen during training.
    pub fn classes(&self) -> &Array1<i32> {
        &self.state.classes
    }
}

#[allow(non_snake_case)]
impl PredictProba<ArrayView2<'_, f64>, Array1<f64>>
    for VariationalGaussianProcessClassifier<VariationalGpcTrained>
{
    /// Predict `P(y = classes()[1] | x)`, estimated by Gauss-Hermite
    /// integration of `sigmoid` over the latent posterior `q(f*)` (a more
    /// calibrated estimate than evaluating `sigmoid` at the posterior mean
    /// alone).
    fn predict_proba(&self, X: &ArrayView2<f64>) -> SklResult<Array1<f64>> {
        let (mean, var) = self.latent_moments(X)?;
        let probs = mean
            .iter()
            .zip(var.iter())
            .map(|(&mu, &v)| gauss_hermite_sigmoid_expectation(mu, v.sqrt()).clamp(0.0, 1.0))
            .collect::<Array1<f64>>();
        Ok(probs)
    }
}

#[allow(non_snake_case)]
impl Predict<ArrayView2<'_, f64>, Array1<i32>>
    for VariationalGaussianProcessClassifier<VariationalGpcTrained>
{
    fn predict(&self, X: &ArrayView2<f64>) -> SklResult<Array1<i32>> {
        let proba = self.predict_proba(X)?;
        let preds = proba
            .iter()
            .map(|&p| {
                if p >= 0.5 {
                    self.state.classes[1]
                } else {
                    self.state.classes[0]
                }
            })
            .collect::<Vec<i32>>();
        Ok(Array1::from(preds))
    }
}

impl Default for VariationalGaussianProcessClassifier<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

/// `SparseGaussianProcessClassifier` is the same sparse-variational-GP
/// classifier as [`VariationalGaussianProcessClassifier`]: the SVGP
/// formulation implemented in this module already performs the full
/// variational treatment needed for sparse GP classification (an
/// inducing-point posterior `q(u)` with a non-conjugate likelihood handled
/// via Gauss-Hermite quadrature), so a separate, simpler approximation would
/// not add anything — this is a thin alias rather than duplicated logic.
pub type SparseGaussianProcessClassifier<S = Untrained> = VariationalGaussianProcessClassifier<S>;
/// Trained state for [`SparseGaussianProcessClassifier`].
pub type SgpcTrained = VariationalGpcTrained;

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::kernels::RBF;
    use scirs2_core::ndarray::array;

    fn linearly_separable_2class_data() -> (Array2<f64>, Array1<i32>) {
        let X = array![
            [-2.0, -2.0],
            [-2.5, -1.5],
            [-1.5, -2.5],
            [-3.0, -2.0],
            [-2.0, -3.0],
            [-2.2, -1.8],
            [-1.8, -2.2],
            [-2.7, -2.3],
            [-1.6, -1.9],
            [-2.4, -2.6],
            [2.0, 2.0],
            [2.5, 1.5],
            [1.5, 2.5],
            [3.0, 2.0],
            [2.0, 3.0],
            [2.2, 1.8],
            [1.8, 2.2],
            [2.7, 2.3],
            [1.6, 1.9],
            [2.4, 2.6],
        ];
        let y = array![0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1];
        (X, y)
    }

    #[test]
    fn test_variational_gpc_elbo_overall_increases() {
        let (X, y) = linearly_separable_2class_data();
        let fitted = VariationalGaussianProcessClassifier::new()
            .kernel(Box::new(RBF::new(1.5)))
            .n_inducing(8)
            .max_iter(200)
            .learning_rate(0.01)
            .optimize_kernel(false)
            .random_state(Some(0))
            .fit(&X.view(), &y.view())
            .expect("variational GPC fit should succeed on separable toy data");

        let history = fitted.elbo_history();
        assert!(
            history.len() >= 2,
            "expected multiple ELBO iterations to be recorded, got {}",
            history.len()
        );
        let first = history[0];
        let last = *history.last().expect("history is non-empty");
        assert!(
            first.is_finite() && last.is_finite(),
            "ELBO values must be finite"
        );
        assert!(
            last > first,
            "ELBO should increase overall across optimization: first={first}, last={last}"
        );
    }

    #[test]
    fn test_variational_gpc_training_accuracy() {
        let (X, y) = linearly_separable_2class_data();
        let fitted = VariationalGaussianProcessClassifier::new()
            .kernel(Box::new(RBF::new(1.5)))
            .n_inducing(8)
            .max_iter(300)
            .learning_rate(0.01)
            .random_state(Some(0))
            .fit(&X.view(), &y.view())
            .expect("variational GPC fit should succeed on separable toy data");

        let preds = fitted
            .predict(&X.view())
            .expect("predict should succeed after fitting");

        let correct = preds.iter().zip(y.iter()).filter(|(&p, &t)| p == t).count();
        let accuracy = correct as f64 / y.len() as f64;
        assert!(
            accuracy > 0.9,
            "expected training accuracy > 0.9 on a well-separated toy set, got {accuracy}"
        );
    }

    #[test]
    fn test_variational_gpc_predict_proba_range() {
        let (X, y) = linearly_separable_2class_data();
        let fitted = VariationalGaussianProcessClassifier::new()
            .kernel(Box::new(RBF::new(1.5)))
            .n_inducing(8)
            .max_iter(50)
            .random_state(Some(0))
            .fit(&X.view(), &y.view())
            .expect("variational GPC fit should succeed on separable toy data");

        let proba = fitted
            .predict_proba(&X.view())
            .expect("predict_proba should succeed after fitting");
        for &p in proba.iter() {
            assert!((0.0..=1.0).contains(&p), "probability out of range: {p}");
        }
    }

    #[test]
    fn test_variational_gpc_requires_two_classes() {
        let X = array![[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]];
        let y = array![0, 0, 0];
        let result = VariationalGaussianProcessClassifier::new()
            .kernel(Box::new(RBF::new(1.0)))
            .fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_variational_gpc_requires_kernel() {
        let (X, y) = linearly_separable_2class_data();
        let result = VariationalGaussianProcessClassifier::new().fit(&X.view(), &y.view());
        assert!(result.is_err());
    }

    #[test]
    fn test_sparse_gpc_alias_matches_variational() {
        let (X, y) = linearly_separable_2class_data();
        let fitted = SparseGaussianProcessClassifier::new()
            .kernel(Box::new(RBF::new(1.5)))
            .n_inducing(8)
            .max_iter(20)
            .random_state(Some(0))
            .fit(&X.view(), &y.view())
            .expect("alias type should fit identically to VariationalGaussianProcessClassifier");
        assert!(fitted.elbo().is_finite());
    }

    #[test]
    fn test_gauss_hermite_bernoulli_moments_sane() {
        // At mu=0, sigma->0, E[log p(y|f)] should approach log(0.5).
        let (e_log_lik, _, _) = gauss_hermite_bernoulli_moments(0.0, 1e-6, 1.0);
        assert!((e_log_lik - (0.5_f64).ln()).abs() < 1e-3);
    }
}
