//! Convolution Processes for multi-output Gaussian processes
//!
//! This module implements the *Convolution Process* (also called "process
//! convolution" or "convolved multi-output GP") construction of Boyle & Frean
//! (2005, "Dependent Gaussian Processes", NeurIPS) and Álvarez & Lawrence (2009,
//! "Convolved Gaussian Process Models for Multi-output Regression", NeurIPS; see
//! also Álvarez, Rosasco & Lawrence 2012, "Kernels for Vector-Valued Functions: A
//! Review"). Several correlated output functions are modelled as convolutions of a
//! small, *shared* set of latent white-noise processes with per-output *smoothing
//! kernels*:
//!
//! ```text
//! f_d(x) = sum_{q=1}^{Q}  ∫  G_{d,q}(x - z) u_q(z) dz ,   d = 1..D_outputs
//! ```
//!
//! where `u_1, ..., u_Q` (`Q = n_latent`) are independent latent white-noise
//! processes, `cov[u_q(z), u_q'(z')] = delta_{q,q'} * delta(z - z')`, and each
//! output's smoothing kernel `G_{d,q}` is chosen Gaussian-shaped:
//!
//! ```text
//! G_{d,q}(t) = S_{d,q} * N(t; 0, P_{d,q}) ,   P_{d,q} = diag(ell_{d,q,1}^2, ..., ell_{d,q,K}^2)
//! ```
//!
//! with `N(.; 0, P)` the *normalized* zero-mean Gaussian density with covariance
//! `P`, and `K` the number of input dimensions.
//!
//! # Closed-form induced covariance
//!
//! A convolution of two Gaussian densities is again Gaussian:
//! `∫ N(x - z; 0, A) * N(x' - z; 0, B) dz = N(x - x'; 0, A + B)`. Applying this
//! identity to `cov[f_d(x), f_d'(x')]` (the white-noise covariance collapses the
//! double integral over `u_q` to a single integral over `z`) gives, summing the `Q`
//! independent latent contributions:
//!
//! ```text
//! k_{d,d'}(x, x') = sum_{q=1}^{Q} S_{d,q} S_{d',q} * N(x - x'; 0, P_{d,q} + P_{d',q})
//! ```
//!
//! This is the "smoothing-kernel-width" parameterization: `P_{d,q}` (equivalently
//! `ell_{d,q}`) is the covariance/width of output `d`'s smoothing kernel for latent
//! process `q`. This module exposes, instead of the raw width `ell_{d,q}`, the
//! *effective output length scale* `L_{d,q} in R^K` -- interpreted exactly like
//! [`crate::kernels::ARDRBF`]'s `length_scales`, i.e. the length scale output `d`'s
//! *own* auto-covariance would have -- together with a *signal variance*
//! `v_{d,q} >= 0` (latent `q`'s contribution to `k_{d,d}(x,x)`), because those are
//! the quantities a user actually reasons about. The two parameterizations are
//! related by
//!
//! ```text
//! ell_{d,q} = L_{d,q} / sqrt(2)   and   S_{d,q} = sqrt(v_{d,q}) * (2*pi)^(K/4) * sqrt(prod_k L_{d,q,k})
//! ```
//!
//! chosen precisely so that substituting back collapses the induced covariance to:
//!
//! ```text
//! k_{d,d'}(x,x') = sum_{q=1}^{Q} sqrt(v_{d,q} * v_{d',q})
//!     * [ prod_{k=1}^{K} sqrt( 2 L_{d,q,k} L_{d',q,k} / (L_{d,q,k}^2 + L_{d',q,k}^2) ) ]
//!     * exp( - sum_{k=1}^{K} (x_k - x'_k)^2 / (L_{d,q,k}^2 + L_{d',q,k}^2) )
//! ```
//!
//! (implemented by [`latent_covariance`] / [`induced_covariance_sum`] below, in
//! log-space for numerical stability). This has two properties that make it a
//! trustworthy building block for a multi-output GP:
//!
//! * **Auto-covariance reduces to ARD-RBF.** For `d == d'` the bracketed prefactor
//!   collapses to exactly `1` (each factor becomes `sqrt(2 L^2 / (2 L^2)) = 1`) and
//!   the formula reduces *exactly* to a variance-scaled ARD-RBF kernel:
//!   `k_{d,d}(x,x') = sum_q v_{d,q} * exp(-0.5 * sum_k (x_k - x'_k)^2 / L_{d,q,k}^2)`.
//!   So with `n_latent = 1`, a single output, and the default `v_{0,0} = 1`, this
//!   model's induced kernel is *identical* to [`crate::kernels::RBF`] /
//!   [`crate::kernels::ARDRBF`] with the same length scale -- see
//!   `test_degenerate_single_output_matches_standard_gpr` and
//!   `test_auto_covariance_matches_ard_rbf`.
//! * **Guaranteed positive semi-definiteness.** The full multi-output Gram matrix
//!   (stacking every output's auto- and cross-covariance blocks) is PSD in exact
//!   arithmetic for *any* choice of positive length scales/variances, because it is,
//!   by construction, the covariance matrix of a finite collection of linear
//!   functionals of the well-defined latent white-noise processes `u_q` -- it is not
//!   an ad hoc matrix that merely happens to pass a PSD check after the fact. (Finite
//!   precision can still make the matrix numerically indefinite, which is why
//!   [`crate::utils::robust_cholesky`] -- with its adaptive nugget -- is used rather
//!   than a bare Cholesky factorization.)
//!
//! # Data layout
//!
//! Unlike [`crate::linear_model_coregionalization`] and
//! [`crate::intrinsic_coregionalization`] (which require every output to be observed
//! at the *same* input locations, via `Y: (n_samples, n_outputs)`), a convolution
//! process is fit from one `(X_d, y_d)` pair *per output*
//! (`Fit<Vec<Array2<f64>>, Vec<Array1<f64>>>`): each output's smoothing kernel is
//! independent of the others, so different outputs may be sampled at different
//! locations, or in different numbers. This is one of the main practical advantages
//! of the convolution-process construction over LMC/ICM, and is directly exercised
//! by `test_two_correlated_outputs_information_sharing` (output 2 has far fewer
//! observations than output 1, at different input locations, yet its predictions
//! benefit from output 1's data through the induced cross-covariance).

use scirs2_core::ndarray::{Array1, Array2, ArrayView1};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Predict, Untrained},
};

use crate::utils;

/// Configuration for a [`ConvolutionProcess`].
///
/// See the module documentation for the mathematical role of each parameter.
#[derive(Debug, Clone)]
pub struct ConvolutionProcessConfig {
    /// Number of shared latent white-noise processes `Q`. Each output is a sum of
    /// `Q` independent convolutions against these shared latents; `n_latent = 1`
    /// (the default) is the simplest well-defined case, and is already sufficient
    /// to induce full cross-output correlation.
    pub n_latent: usize,
    /// Default per-dimension effective output length scale `L_{d,q,k}`, used to
    /// initialize every output/latent/dimension's length scale when an explicit
    /// override is not supplied via [`ConvolutionProcess::length_scales`].
    pub default_length_scale: f64,
    /// Default noise variance added to the diagonal of every output's covariance
    /// block, used when an explicit override is not supplied via
    /// [`ConvolutionProcess::noise_variances`].
    pub noise_variance: f64,
}

impl Default for ConvolutionProcessConfig {
    fn default() -> Self {
        Self {
            n_latent: 1,
            default_length_scale: 1.0,
            noise_variance: 1e-10,
        }
    }
}

/// Convolution Process (a.k.a. Convolved / Dependent Gaussian Process) for
/// multi-output regression.
///
/// See the module documentation for the generative model and the closed-form
/// induced covariance this type trains against. Use the builder methods below to
/// configure it, then call [`Fit::fit`] with one input matrix and one target vector
/// *per output*.
///
/// # Examples
///
/// ```
/// use sklears_gaussian_process::ConvolutionProcess;
/// use sklears_core::traits::{Fit, Predict};
/// use scirs2_core::ndarray::array;
///
/// let x1 = array![[0.0], [1.0], [2.0], [3.0]];
/// let y1 = array![0.0, 1.0, 4.0, 9.0];
/// let x2 = array![[0.5], [2.5]];
/// let y2 = array![2.0, 17.0];
///
/// let cp = ConvolutionProcess::new().n_latent(1).noise_variance(1e-6);
/// let fitted = cp
///     .fit(&vec![x1, x2], &vec![y1, y2])
///     .expect("fit should succeed with valid training data");
/// let predictions = fitted
///     .predict(&vec![array![[1.5]], array![[1.0]]])
///     .expect("predict should succeed on a trained model");
/// assert_eq!(predictions.len(), 2);
/// ```
#[derive(Debug, Clone)]
pub struct ConvolutionProcess<S = Untrained> {
    config: ConvolutionProcessConfig,
    /// Optional explicit `[output][latent] -> per-dimension length scale` override.
    /// When `None`, every output/latent/dimension defaults to
    /// `config.default_length_scale` (resolved once the number of outputs and the
    /// input dimensionality are known, at `fit` time).
    length_scales: Option<Vec<Vec<Array1<f64>>>>,
    /// Optional explicit `[output][latent] -> signal variance` override. When
    /// `None`, defaults to `1 / n_latent` for every output/latent pair, so each
    /// output's total prior variance `k_{d,d}(x,x)` is `1.0` by default (matching
    /// the usual convention of a normalized kernel).
    output_variances: Option<Vec<Vec<f64>>>,
    /// Optional explicit per-output noise variance override. When `None`, every
    /// output defaults to `config.noise_variance`.
    noise_variances: Option<Vec<f64>>,
    _state: S,
}

/// Trained state of a [`ConvolutionProcess`].
#[derive(Debug, Clone)]
#[allow(non_snake_case)]
pub struct ConvolutionProcessTrained {
    pub(crate) n_outputs: usize,
    pub(crate) n_latent: usize,
    pub(crate) n_features: usize,
    /// Per-output training inputs, `X_train[d]` has shape `(n_samples_d, n_features)`.
    pub(crate) X_train: Vec<Array2<f64>>,
    /// Per-output training targets, `y_train[d]` has length `n_samples_d`.
    #[allow(dead_code)]
    pub(crate) y_train: Vec<Array1<f64>>,
    /// Row offset of output `d`'s block within the stacked covariance / alpha
    /// vector: output `d`'s rows occupy `offsets[d] .. offsets[d] + n_samples_d`.
    pub(crate) offsets: Vec<usize>,
    /// Total number of stacked training rows across all outputs.
    pub(crate) n_total: usize,
    /// Resolved `[output][latent] -> length scale` table actually used for training.
    pub(crate) length_scales: Vec<Vec<Array1<f64>>>,
    /// Resolved `[output][latent] -> signal variance` table actually used for training.
    pub(crate) output_variances: Vec<Vec<f64>>,
    /// Resolved per-output noise variance actually used for training.
    pub(crate) noise_variances: Vec<f64>,
    /// Cholesky factor of the stacked (noise-regularized) covariance matrix.
    pub(crate) L: Array2<f64>,
    /// Solved coefficients `alpha = K_reg^{-1} y` (stacked across outputs).
    pub(crate) alpha: Array1<f64>,
    pub(crate) log_marginal_likelihood_value: f64,
}

/// Bundle of fully-resolved (defaults-applied, validated) hyperparameters, computed
/// once the number of outputs and input dimensionality are known at `fit` time.
struct ResolvedHyperparameters {
    n_features: usize,
    length_scales: Vec<Vec<Array1<f64>>>,
    output_variances: Vec<Vec<f64>>,
    noise_variances: Vec<f64>,
}

impl Default for ConvolutionProcess<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

#[allow(non_snake_case)]
impl ConvolutionProcess<Untrained> {
    /// Create a new, unconfigured Convolution Process (`n_latent = 1` by default).
    pub fn new() -> Self {
        Self {
            config: ConvolutionProcessConfig::default(),
            length_scales: None,
            output_variances: None,
            noise_variances: None,
            _state: Untrained,
        }
    }

    /// Set the number of shared latent processes `Q`.
    pub fn n_latent(mut self, n_latent: usize) -> Self {
        self.config.n_latent = n_latent;
        self
    }

    /// Set the default per-dimension length scale used when [`Self::length_scales`]
    /// is not supplied explicitly.
    pub fn default_length_scale(mut self, length_scale: f64) -> Self {
        self.config.default_length_scale = length_scale;
        self
    }

    /// Set the default (uniform, per-output) noise variance used when
    /// [`Self::noise_variances`] is not supplied explicitly.
    pub fn noise_variance(mut self, noise_variance: f64) -> Self {
        self.config.noise_variance = noise_variance;
        self
    }

    /// Explicitly set the `[output][latent] -> per-dimension length scale` table.
    /// The outer `Vec` must have one entry per output (matching the length of the
    /// `X` passed to [`Fit::fit`]), each inner `Vec` one entry per latent process
    /// (matching [`Self::n_latent`]), and each [`Array1`] one length scale per input
    /// dimension. Validated (and defaulted, if omitted) at `fit` time.
    pub fn length_scales(mut self, length_scales: Vec<Vec<Array1<f64>>>) -> Self {
        self.length_scales = Some(length_scales);
        self
    }

    /// Explicitly set the `[output][latent] -> signal variance` table (same outer/
    /// inner shape as [`Self::length_scales`], minus the per-dimension axis).
    pub fn output_variances(mut self, output_variances: Vec<Vec<f64>>) -> Self {
        self.output_variances = Some(output_variances);
        self
    }

    /// Explicitly set a per-output noise variance (one entry per output).
    pub fn noise_variances(mut self, noise_variances: Vec<f64>) -> Self {
        self.noise_variances = Some(noise_variances);
        self
    }

    /// Resolve (validating or defaulting) the hyperparameters needed to fit against
    /// `X`, an owning `(n_outputs)`-length collection of per-output design matrices.
    fn resolve_hyperparameters(&self, X: &[Array2<f64>]) -> SklResult<ResolvedHyperparameters> {
        let n_outputs = X.len();
        if n_outputs == 0 {
            return Err(SklearsError::InvalidInput(
                "ConvolutionProcess requires at least one output".to_string(),
            ));
        }
        if self.config.n_latent == 0 {
            return Err(SklearsError::InvalidInput(
                "n_latent must be at least 1".to_string(),
            ));
        }
        let n_features = X[0].ncols();
        if n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "Input data must have at least one feature".to_string(),
            ));
        }
        for (d, Xd) in X.iter().enumerate() {
            if Xd.ncols() != n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Output {} has {} input features, but output 0 has {}; every \
                     output must share the same input dimensionality",
                    d,
                    Xd.ncols(),
                    n_features
                )));
            }
            if Xd.nrows() == 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "Output {} has no training samples",
                    d
                )));
            }
        }

        let n_latent = self.config.n_latent;

        let length_scales = if let Some(ref ls) = self.length_scales {
            if ls.len() != n_outputs {
                return Err(SklearsError::InvalidInput(format!(
                    "length_scales has {} outputs, expected {}",
                    ls.len(),
                    n_outputs
                )));
            }
            for (d, ls_d) in ls.iter().enumerate() {
                if ls_d.len() != n_latent {
                    return Err(SklearsError::InvalidInput(format!(
                        "length_scales[{}] has {} latent entries, expected n_latent={}",
                        d,
                        ls_d.len(),
                        n_latent
                    )));
                }
                for (q, ls_dq) in ls_d.iter().enumerate() {
                    if ls_dq.len() != n_features {
                        return Err(SklearsError::InvalidInput(format!(
                            "length_scales[{}][{}] has {} dimensions, expected {}",
                            d,
                            q,
                            ls_dq.len(),
                            n_features
                        )));
                    }
                    if ls_dq.iter().any(|&v| v <= 0.0 || !v.is_finite()) {
                        return Err(SklearsError::InvalidInput(format!(
                            "length_scales[{}][{}] must be strictly positive and finite",
                            d, q
                        )));
                    }
                }
            }
            ls.clone()
        } else {
            if self.config.default_length_scale <= 0.0
                || !self.config.default_length_scale.is_finite()
            {
                return Err(SklearsError::InvalidInput(
                    "default_length_scale must be strictly positive and finite".to_string(),
                ));
            }
            (0..n_outputs)
                .map(|_| {
                    (0..n_latent)
                        .map(|_| Array1::from_elem(n_features, self.config.default_length_scale))
                        .collect()
                })
                .collect()
        };

        let output_variances = if let Some(ref ov) = self.output_variances {
            if ov.len() != n_outputs {
                return Err(SklearsError::InvalidInput(format!(
                    "output_variances has {} outputs, expected {}",
                    ov.len(),
                    n_outputs
                )));
            }
            for (d, ov_d) in ov.iter().enumerate() {
                if ov_d.len() != n_latent {
                    return Err(SklearsError::InvalidInput(format!(
                        "output_variances[{}] has {} latent entries, expected n_latent={}",
                        d,
                        ov_d.len(),
                        n_latent
                    )));
                }
                if ov_d.iter().any(|&v| v < 0.0 || !v.is_finite()) {
                    return Err(SklearsError::InvalidInput(format!(
                        "output_variances[{}] must be non-negative and finite",
                        d
                    )));
                }
            }
            ov.clone()
        } else {
            let default_variance = 1.0 / n_latent as f64;
            vec![vec![default_variance; n_latent]; n_outputs]
        };

        let noise_variances = if let Some(ref nv) = self.noise_variances {
            if nv.len() != n_outputs {
                return Err(SklearsError::InvalidInput(format!(
                    "noise_variances has {} entries, expected {}",
                    nv.len(),
                    n_outputs
                )));
            }
            if nv.iter().any(|&v| v < 0.0 || !v.is_finite()) {
                return Err(SklearsError::InvalidInput(
                    "noise_variances must be non-negative and finite".to_string(),
                ));
            }
            nv.clone()
        } else {
            if self.config.noise_variance < 0.0 || !self.config.noise_variance.is_finite() {
                return Err(SklearsError::InvalidInput(
                    "noise_variance must be non-negative and finite".to_string(),
                ));
            }
            vec![self.config.noise_variance; n_outputs]
        };

        Ok(ResolvedHyperparameters {
            n_features,
            length_scales,
            output_variances,
            noise_variances,
        })
    }
}

/// Closed-form induced covariance contributed by a *single* latent process between
/// two (output, point) pairs. See the module documentation for the derivation; `l1`,
/// `l2` are the two outputs' per-dimension effective length scales `L_{d,q,:}` and
/// `L_{d',q,:}` for this latent process, `v1`, `v2` their signal variances.
///
/// Computed in log-space (`log_shape - exponent`, exponentiated once at the end)
/// since both the prefactor (a product over dimensions) and the exponential decay
/// can otherwise under/overflow for higher-dimensional inputs.
fn latent_covariance(
    x1: &ArrayView1<f64>,
    x2: &ArrayView1<f64>,
    l1: &Array1<f64>,
    l2: &Array1<f64>,
    v1: f64,
    v2: f64,
) -> f64 {
    if v1 <= 0.0 || v2 <= 0.0 {
        return 0.0;
    }
    let n_dims = l1.len();
    let mut log_shape = 0.0_f64;
    let mut exponent = 0.0_f64;
    for k in 0..n_dims {
        let l1k = l1[k];
        let l2k = l2[k];
        let sum_sq = l1k * l1k + l2k * l2k;
        // By AM-GM, 2*l1k*l2k <= sum_sq, so this ratio is in (0, 1] and the log is <= 0:
        // mismatched length scales strictly attenuate the cross-covariance amplitude.
        log_shape += 0.5 * (2.0 * l1k * l2k / sum_sq).ln();
        let diff = x1[k] - x2[k];
        exponent += diff * diff / sum_sq;
    }
    let amplitude = (v1 * v2).sqrt();
    amplitude * (log_shape - exponent).exp()
}

/// Sum of [`latent_covariance`] over all `n_latent` shared latent processes, i.e.
/// the full induced covariance `k_{d,d'}(x1, x2)` between output `d` (parameterized
/// by `ls_d`/`ov_d`) and output `d'` (parameterized by `ls_dp`/`ov_dp`).
fn induced_covariance_sum(
    x1: &ArrayView1<f64>,
    x2: &ArrayView1<f64>,
    ls_d: &[Array1<f64>],
    ls_dp: &[Array1<f64>],
    ov_d: &[f64],
    ov_dp: &[f64],
) -> f64 {
    let n_latent = ls_d.len();
    let mut total = 0.0;
    for q in 0..n_latent {
        total += latent_covariance(x1, x2, &ls_d[q], &ls_dp[q], ov_d[q], ov_dp[q]);
    }
    total
}

/// Compute the row offset of each output's block within the stacked training
/// vector/matrix, plus the total stacked size.
#[allow(non_snake_case)]
fn compute_offsets(X: &[Array2<f64>]) -> (Vec<usize>, usize) {
    let mut offsets = Vec::with_capacity(X.len());
    let mut total = 0usize;
    for Xd in X {
        offsets.push(total);
        total += Xd.nrows();
    }
    (offsets, total)
}

/// Build the full stacked multi-output covariance matrix (no noise added) between
/// every training point of every output, using the given (already-resolved)
/// hyperparameters. Block `(d, d')` of the result is the induced cross-covariance
/// `k_{d,d'}(X[d], X[d'])`; the diagonal blocks (`d == d'`) are each output's
/// auto-covariance.
#[allow(non_snake_case)]
fn build_stacked_covariance(
    X: &[Array2<f64>],
    length_scales: &[Vec<Array1<f64>>],
    output_variances: &[Vec<f64>],
    offsets: &[usize],
    n_total: usize,
) -> Array2<f64> {
    let n_outputs = X.len();
    let mut K = Array2::<f64>::zeros((n_total, n_total));
    for d in 0..n_outputs {
        for i in 0..X[d].nrows() {
            let row_idx = offsets[d] + i;
            let xi = X[d].row(i);
            for dp in 0..n_outputs {
                for j in 0..X[dp].nrows() {
                    let col_idx = offsets[dp] + j;
                    let xj = X[dp].row(j);
                    K[[row_idx, col_idx]] = induced_covariance_sum(
                        &xi,
                        &xj,
                        &length_scales[d],
                        &length_scales[dp],
                        &output_variances[d],
                        &output_variances[dp],
                    );
                }
            }
        }
    }
    K
}

impl Estimator for ConvolutionProcess<Untrained> {
    type Config = ConvolutionProcessConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

impl Estimator for ConvolutionProcess<ConvolutionProcessTrained> {
    type Config = ConvolutionProcessConfig;
    type Error = SklearsError;
    type Float = f64;

    fn config(&self) -> &Self::Config {
        &self.config
    }
}

#[allow(non_snake_case)]
impl Fit<Vec<Array2<f64>>, Vec<Array1<f64>>> for ConvolutionProcess<Untrained> {
    type Fitted = ConvolutionProcess<ConvolutionProcessTrained>;

    fn fit(self, X: &Vec<Array2<f64>>, Y: &Vec<Array1<f64>>) -> SklResult<Self::Fitted> {
        if X.len() != Y.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of outputs in X ({}) must match Y ({})",
                X.len(),
                Y.len()
            )));
        }
        for (d, (Xd, Yd)) in X.iter().zip(Y.iter()).enumerate() {
            if Xd.nrows() != Yd.len() {
                return Err(SklearsError::InvalidInput(format!(
                    "Output {}: X has {} samples but y has {}",
                    d,
                    Xd.nrows(),
                    Yd.len()
                )));
            }
        }

        let ResolvedHyperparameters {
            n_features,
            length_scales,
            output_variances,
            noise_variances,
        } = self.resolve_hyperparameters(X)?;

        let n_outputs = X.len();
        let n_latent = self.config.n_latent;
        let (offsets, n_total) = compute_offsets(X);

        let mut K =
            build_stacked_covariance(X, &length_scales, &output_variances, &offsets, n_total);
        for d in 0..n_outputs {
            for i in 0..X[d].nrows() {
                K[[offsets[d] + i, offsets[d] + i]] += noise_variances[d];
            }
        }

        let L = utils::robust_cholesky(&K)?;

        let mut y_stacked = Array1::<f64>::zeros(n_total);
        for d in 0..n_outputs {
            for i in 0..Y[d].len() {
                y_stacked[offsets[d] + i] = Y[d][i];
            }
        }

        let alpha = utils::triangular_solve(&L, &y_stacked)?;
        let log_marginal_likelihood_value = utils::log_marginal_likelihood(&L, &alpha, &y_stacked);

        let trained = ConvolutionProcessTrained {
            n_outputs,
            n_latent,
            n_features,
            X_train: X.clone(),
            y_train: Y.clone(),
            offsets,
            n_total,
            length_scales,
            output_variances,
            noise_variances,
            L,
            alpha,
            log_marginal_likelihood_value,
        };

        Ok(ConvolutionProcess {
            config: self.config,
            length_scales: self.length_scales,
            output_variances: self.output_variances,
            noise_variances: self.noise_variances,
            _state: trained,
        })
    }
}

#[allow(non_snake_case)]
impl ConvolutionProcess<ConvolutionProcessTrained> {
    /// Access the full trained state.
    pub fn trained_state(&self) -> &ConvolutionProcessTrained {
        &self._state
    }

    /// Number of outputs the model was fitted on.
    pub fn n_outputs(&self) -> usize {
        self._state.n_outputs
    }

    /// Number of shared latent processes `Q`.
    pub fn n_latent(&self) -> usize {
        self._state.n_latent
    }

    /// Log marginal likelihood of the training data under the fitted model.
    pub fn log_marginal_likelihood(&self) -> f64 {
        self._state.log_marginal_likelihood_value
    }

    /// Resolved per-output noise variance actually used for training (one entry per
    /// output, in fit order).
    pub fn noise_variances(&self) -> &[f64] {
        &self._state.noise_variances
    }

    /// Evaluate the model's induced covariance `k_{d,d'}(x, x')` between output
    /// `output_d` and output `output_dp` (which may be equal) at arbitrary points,
    /// using the fitted (or defaulted) hyperparameters. See the module
    /// documentation for the closed-form formula being evaluated.
    ///
    /// This is the direct, low-level way to inspect whether -- and how strongly --
    /// the model represents correlation between two outputs: for two genuinely
    /// independent single-output GPs this would be identically zero everywhere,
    /// whereas a convolution process sharing at least one latent process between
    /// `output_d` and `output_dp` gives a nontrivial value.
    pub fn induced_covariance(
        &self,
        output_d: usize,
        output_dp: usize,
        x: &ArrayView1<f64>,
        xp: &ArrayView1<f64>,
    ) -> SklResult<f64> {
        let state = &self._state;
        if output_d >= state.n_outputs || output_dp >= state.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "Output index out of range: got ({}, {}), model has {} outputs",
                output_d, output_dp, state.n_outputs
            )));
        }
        if x.len() != state.n_features || xp.len() != state.n_features {
            return Err(SklearsError::InvalidInput(format!(
                "Expected points with {} features",
                state.n_features
            )));
        }
        Ok(induced_covariance_sum(
            x,
            xp,
            &state.length_scales[output_d],
            &state.length_scales[output_dp],
            &state.output_variances[output_d],
            &state.output_variances[output_dp],
        ))
    }

    /// Rebuild the full (noise-free) stacked multi-output covariance matrix over
    /// the training points, using the fitted hyperparameters. This is the matrix
    /// whose positive semi-definiteness is the key theoretical guarantee of the
    /// convolution-process construction (see module docs); exposed for diagnostics
    /// and testing.
    pub fn training_covariance_matrix(&self) -> Array2<f64> {
        let state = &self._state;
        build_stacked_covariance(
            &state.X_train,
            &state.length_scales,
            &state.output_variances,
            &state.offsets,
            state.n_total,
        )
    }

    /// Predict the posterior mean and standard deviation for each output at the
    /// given per-output test points. `X_test` must have exactly `n_outputs()`
    /// entries (one design matrix per output, in the same order as at `fit` time);
    /// pass a zero-row matrix for any output you are not interested in predicting.
    pub fn predict_with_std(
        &self,
        X_test: &[Array2<f64>],
    ) -> SklResult<Vec<(Array1<f64>, Array1<f64>)>> {
        let state = &self._state;
        if X_test.len() != state.n_outputs {
            return Err(SklearsError::InvalidInput(format!(
                "predict_with_std expected {} outputs, got {}",
                state.n_outputs,
                X_test.len()
            )));
        }

        let mut results = Vec::with_capacity(state.n_outputs);
        for (d, Xd) in X_test.iter().enumerate() {
            if Xd.nrows() > 0 && Xd.ncols() != state.n_features {
                return Err(SklearsError::InvalidInput(format!(
                    "Output {} test inputs have {} features, expected {}",
                    d,
                    Xd.ncols(),
                    state.n_features
                )));
            }
            let n_star = Xd.nrows();
            let mut mean_d = Array1::<f64>::zeros(n_star);
            let mut std_d = Array1::<f64>::zeros(n_star);

            for t in 0..n_star {
                let x_star = Xd.row(t);

                let mut k_star = Array1::<f64>::zeros(state.n_total);
                for dp in 0..state.n_outputs {
                    for j in 0..state.X_train[dp].nrows() {
                        let x_train_j = state.X_train[dp].row(j);
                        k_star[state.offsets[dp] + j] = induced_covariance_sum(
                            &x_star,
                            &x_train_j,
                            &state.length_scales[d],
                            &state.length_scales[dp],
                            &state.output_variances[d],
                            &state.output_variances[dp],
                        );
                    }
                }

                let mean = k_star.dot(&state.alpha);

                // Predictive variance: k_** - k_*^T K_reg^{-1} k_*, mirroring
                // gpr.rs::predict_with_std -- `triangular_solve` performs a *full*
                // solve, so the quadratic form dots the *original* k_star against
                // the solved vector `v`, not `v` against itself.
                let v = utils::triangular_solve(&state.L, &k_star)?;
                let k_star_star = induced_covariance_sum(
                    &x_star,
                    &x_star,
                    &state.length_scales[d],
                    &state.length_scales[d],
                    &state.output_variances[d],
                    &state.output_variances[d],
                );
                let var = (k_star_star - k_star.dot(&v)).max(0.0);

                mean_d[t] = mean;
                std_d[t] = var.sqrt();
            }

            results.push((mean_d, std_d));
        }
        Ok(results)
    }
}

#[allow(non_snake_case)]
impl Predict<Vec<Array2<f64>>, Vec<Array1<f64>>> for ConvolutionProcess<ConvolutionProcessTrained> {
    fn predict(&self, X: &Vec<Array2<f64>>) -> SklResult<Vec<Array1<f64>>> {
        let with_std = self.predict_with_std(X)?;
        Ok(with_std.into_iter().map(|(mean, _std)| mean).collect())
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpr::GaussianProcessRegressor;
    use crate::kernels::{Kernel, ARDRBF, RBF};
    use approx::assert_abs_diff_eq;
    // SciRS2 Policy - Use scirs2-core for array! macro and types
    use scirs2_core::ndarray::array;

    #[test]
    fn test_convolution_process_creation() {
        let cp = ConvolutionProcess::new().n_latent(3).noise_variance(1e-4);
        assert_eq!(cp.config.n_latent, 3);
        assert_abs_diff_eq!(cp.config.noise_variance, 1e-4, epsilon = 1e-16);
        assert_abs_diff_eq!(cp.config.default_length_scale, 1.0, epsilon = 1e-16);
    }

    #[test]
    fn test_convolution_process_fit_predict_shapes() {
        let x1 = array![[0.0], [1.0], [2.0]];
        let y1 = array![1.0, 2.0, 1.5];
        let x2 = array![[0.5], [1.5]];
        let y2 = array![0.5, 1.0];

        let cp = ConvolutionProcess::new().n_latent(1).noise_variance(1e-6);
        let fitted = cp
            .fit(&vec![x1, x2], &vec![y1, y2])
            .expect("fit should succeed");

        assert_eq!(fitted.n_outputs(), 2);
        assert_eq!(fitted.n_latent(), 1);
        assert!(fitted.log_marginal_likelihood().is_finite());

        let x1_test = array![[0.25], [1.25]];
        let x2_test = array![[0.75]];
        let preds = fitted
            .predict(&vec![x1_test.clone(), x2_test.clone()])
            .expect("predict should succeed");

        assert_eq!(preds.len(), 2);
        assert_eq!(preds[0].len(), 2);
        assert_eq!(preds[1].len(), 1);

        let with_std = fitted
            .predict_with_std(&[x1_test, x2_test])
            .expect("predict_with_std should succeed");
        assert_eq!(with_std.len(), 2);
        for (mean, std) in &with_std {
            for &s in std.iter() {
                assert!(s >= 0.0, "std must be non-negative, got {}", s);
            }
            for &m in mean.iter() {
                assert!(m.is_finite());
            }
        }
    }

    #[test]
    fn test_convolution_process_errors() {
        let x1 = array![[0.0], [1.0]];
        let y1 = array![1.0, 2.0];

        // Mismatched number of outputs between X and y.
        let cp = ConvolutionProcess::new();
        let result = cp.fit(&vec![x1.clone()], &vec![y1.clone(), array![0.0]]);
        assert!(result.is_err());

        // Mismatched sample count within an output.
        let cp = ConvolutionProcess::new();
        let bad_y = array![1.0, 2.0, 3.0];
        let result = cp.fit(&vec![x1.clone()], &vec![bad_y]);
        assert!(result.is_err());

        // n_latent = 0 is invalid.
        let cp = ConvolutionProcess::new().n_latent(0);
        let result = cp.fit(&vec![x1.clone()], &vec![y1.clone()]);
        assert!(result.is_err());

        // Non-positive default length scale is invalid.
        let cp = ConvolutionProcess::new().default_length_scale(0.0);
        let result = cp.fit(&vec![x1.clone()], &vec![y1.clone()]);
        assert!(result.is_err());

        // Empty output list.
        let cp = ConvolutionProcess::new();
        let result = cp.fit(&Vec::<Array2<f64>>::new(), &Vec::<Array1<f64>>::new());
        assert!(result.is_err());

        // Mismatched feature dimensionality across outputs.
        let cp = ConvolutionProcess::new();
        let x2_wrong_dim = array![[0.0, 1.0], [1.0, 2.0]];
        let y2 = array![1.0, 2.0];
        let result = cp.fit(&vec![x1.clone(), x2_wrong_dim], &vec![y1.clone(), y2]);
        assert!(result.is_err());
    }

    #[test]
    fn test_log_marginal_likelihood_finite() {
        let x1 = array![[0.0], [1.0], [2.0], [3.0]];
        let y1 = array![1.0, 2.0, 3.0, 4.0];

        let cp = ConvolutionProcess::new().n_latent(1).noise_variance(1e-6);
        let fitted = cp.fit(&vec![x1], &vec![y1]).expect("fit should succeed");

        assert!(fitted.log_marginal_likelihood().is_finite());
    }

    /// **Required test 1: PSD check.** The induced multi-output covariance matrix
    /// (over a small two-output toy dataset, with a nontrivial `n_latent = 2`
    /// structure using distinct length scales per output/latent) must be positive
    /// semi-definite: `fit()` succeeding demonstrates a successful Cholesky
    /// factorization of `K + noise*I`, and we additionally verify every eigenvalue
    /// of the raw (noise-free) induced covariance matrix is non-negative.
    #[test]
    fn test_psd_multi_output_covariance() {
        let x1 = array![[0.0], [1.0], [2.0]];
        let y1 = array![0.1, 0.4, 0.2];
        let x2 = array![[0.2], [0.9], [2.1], [3.0]];
        let y2 = array![0.3, 0.5, 0.1, 0.6];

        let length_scales = vec![
            vec![Array1::from_elem(1, 0.5), Array1::from_elem(1, 2.0)],
            vec![Array1::from_elem(1, 0.8), Array1::from_elem(1, 1.5)],
        ];
        let output_variances = vec![vec![0.6, 0.4], vec![0.3, 0.7]];

        let cp = ConvolutionProcess::new()
            .n_latent(2)
            .length_scales(length_scales)
            .output_variances(output_variances)
            .noise_variance(1e-6);

        let fitted = cp.fit(&vec![x1, x2], &vec![y1, y2]).expect(
            "fit should succeed: the stacked covariance must be PSD for robust_cholesky to succeed",
        );

        let k = fitted.training_covariance_matrix();
        assert_eq!(k.shape(), &[7, 7]);

        // Symmetric.
        for i in 0..k.nrows() {
            for j in 0..k.ncols() {
                assert_abs_diff_eq!(k[[i, j]], k[[j, i]], epsilon = 1e-10);
            }
        }

        // Every eigenvalue is non-negative (up to numerical tolerance).
        use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
        let (eigenvalues, _eigenvectors) = k
            .eigh(UPLO::Lower)
            .expect("eigendecomposition should succeed");
        for &ev in eigenvalues.iter() {
            assert!(
                ev >= -1e-8,
                "covariance matrix has a negative eigenvalue: {}",
                ev
            );
        }
    }

    /// **Required test 2: degenerate case.** With `n_latent = 1` and a single
    /// output, the model's induced kernel is mathematically identical to a
    /// standard RBF kernel (see module docs), so its posterior mean/std should
    /// match [`GaussianProcessRegressor`] with an [`RBF`] kernel of the same
    /// length scale, essentially exactly (both ultimately call the same
    /// `robust_cholesky` / `triangular_solve` routines over the same matrix).
    #[test]
    fn test_degenerate_single_output_matches_standard_gpr() {
        let x = array![[0.0], [1.0], [2.0], [3.0], [4.0]];
        let y = array![0.5, 2.0, 1.5, 3.0, 2.5];
        let length_scale = 1.3;
        let noise = 1e-6;

        let cp = ConvolutionProcess::new()
            .n_latent(1)
            .default_length_scale(length_scale)
            .noise_variance(noise);
        let cp_fitted = cp
            .fit(&vec![x.clone()], &vec![y.clone()])
            .expect("fit should succeed");

        let gpr = GaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(length_scale)))
            .alpha(noise);
        let gpr_fitted = gpr.fit(&x, &y).expect("fit should succeed");

        let x_test = array![[0.5], [1.5], [2.5], [3.5], [5.0]];
        let cp_result = cp_fitted
            .predict_with_std(std::slice::from_ref(&x_test))
            .expect("predict should succeed");
        let (cp_mean, cp_std) = &cp_result[0];
        let (gpr_mean, gpr_std) = gpr_fitted
            .predict_with_std(&x_test)
            .expect("predict should succeed");

        for i in 0..x_test.nrows() {
            assert_abs_diff_eq!(cp_mean[i], gpr_mean[i], epsilon = 1e-6);
            assert_abs_diff_eq!(cp_std[i], gpr_std[i], epsilon = 1e-6);
        }

        // Also cross-check the log marginal likelihoods, computed independently by
        // the two implementations over what should be the same K + noise*I matrix.
        assert_abs_diff_eq!(
            cp_fitted.log_marginal_likelihood(),
            gpr_fitted.log_marginal_likelihood(),
            epsilon = 1e-6
        );
    }

    /// The single-output/single-latent auto-covariance must equal
    /// `variance * ARDRBF(length_scales)` exactly, directly checking the closed-form
    /// formula (isolated from the rest of the GP machinery) against a known-correct
    /// reference kernel implementation.
    #[test]
    fn test_auto_covariance_matches_ard_rbf() {
        let length_scales = Array1::from_vec(vec![0.7, 1.3]);
        let variance = 2.5;

        let cp = ConvolutionProcess::new()
            .n_latent(1)
            .length_scales(vec![vec![length_scales.clone()]])
            .output_variances(vec![vec![variance]])
            .noise_variance(1e-8);

        let x = array![[0.0, 0.0], [1.0, 2.0]];
        let y = array![0.0, 0.0];
        let fitted = cp.fit(&vec![x], &vec![y]).expect("fit should succeed");

        let ard = ARDRBF::new(length_scales);
        let x1 = array![0.3, -0.2];
        let x2 = array![1.1, 0.4];

        let cp_val = fitted
            .induced_covariance(0, 0, &x1.view(), &x2.view())
            .expect("induced_covariance should succeed");
        let ard_val = variance * ard.kernel(&x1.view(), &x2.view());

        assert_abs_diff_eq!(cp_val, ard_val, epsilon = 1e-10);

        // And at x1 == x2, the auto-covariance must equal the variance exactly
        // (ARD-RBF kernel is 1 at zero distance).
        let self_cov = fitted
            .induced_covariance(0, 0, &x1.view(), &x1.view())
            .expect("induced_covariance should succeed");
        assert_abs_diff_eq!(self_cov, variance, epsilon = 1e-10);
    }

    /// The induced cross-covariance must be symmetric under swapping (output,
    /// point) pairs, and must respect the Cauchy-Schwarz bound
    /// `|k_{d,d'}(x,x)| <= sqrt(k_{d,d}(x,x) * k_{d',d'}(x,x))` that any valid
    /// cross-covariance of jointly well-defined processes has to satisfy.
    #[test]
    fn test_induced_covariance_symmetry_and_cauchy_schwarz() {
        let length_scales = vec![
            vec![Array1::from_elem(1, 0.6)],
            vec![Array1::from_elem(1, 1.4)],
        ];
        let output_variances = vec![vec![1.2], vec![0.8]];

        let x1 = array![[0.0], [1.0]];
        let y1 = array![0.0, 0.0];
        let x2 = array![[0.5], [1.5]];
        let y2 = array![0.0, 0.0];

        let cp = ConvolutionProcess::new()
            .n_latent(1)
            .length_scales(length_scales)
            .output_variances(output_variances)
            .noise_variance(1e-8);
        let fitted = cp
            .fit(&vec![x1, x2], &vec![y1, y2])
            .expect("fit should succeed");

        let a = array![0.3];
        let b = array![1.7];

        let k01 = fitted
            .induced_covariance(0, 1, &a.view(), &b.view())
            .expect("induced_covariance should succeed");
        let k10 = fitted
            .induced_covariance(1, 0, &b.view(), &a.view())
            .expect("induced_covariance should succeed");
        assert_abs_diff_eq!(k01, k10, epsilon = 1e-12);

        let k00 = fitted
            .induced_covariance(0, 0, &a.view(), &a.view())
            .expect("induced_covariance should succeed");
        let k11 = fitted
            .induced_covariance(1, 1, &a.view(), &a.view())
            .expect("induced_covariance should succeed");
        let k01_same = fitted
            .induced_covariance(0, 1, &a.view(), &a.view())
            .expect("induced_covariance should succeed");
        assert!(
            k01_same.abs() <= (k00 * k11).sqrt() + 1e-12,
            "cross-covariance {} violates Cauchy-Schwarz bound {}",
            k01_same,
            (k00 * k11).sqrt()
        );
        // With distinct length scales the bound is strict (not attained).
        assert!(k01_same.abs() < (k00 * k11).sqrt());
    }

    /// **Required test 3: two correlated outputs.** Output 2 is a scaled version of
    /// output 1 (`y2 = 2 * y1`), driven by the same shared latent process, but is
    /// only sparsely observed at locations *disjoint* from output 1's. We verify
    /// both (a) the fitted model's cross-covariance between the two outputs is
    /// nontrivially nonzero, and (b) that this cross-covariance actually gets used
    /// during inference: predicting output 2 far from output 2's own data is
    /// substantially more accurate under the joint convolution-process fit than
    /// under a standalone single-output GP fit only on output 2's sparse data --
    /// i.e. the model is not equivalent to two independent single-output GPs.
    #[test]
    fn test_two_correlated_outputs_information_sharing() {
        // Output 1: densely sampled sin(x).
        let x1 = array![[0.0], [0.5], [1.0], [1.5], [2.0], [2.5], [3.0]];
        let y1: Array1<f64> = x1.column(0).mapv(|v: f64| v.sin());

        // Output 2: y2 = 2 * sin(x), sparsely sampled at locations disjoint from x1.
        let x2 = array![[0.25], [1.25]];
        let y2: Array1<f64> = x2.column(0).mapv(|v: f64| 2.0 * v.sin());

        let length_scale = 0.8;
        let noise = 1e-4;

        // Signal variances scaled 1:4, matching Var[2*f] = 4*Var[f] -- i.e. the
        // model's prior belief about the two outputs' relationship matches the
        // true generative scaling of 2x.
        let cp = ConvolutionProcess::new()
            .n_latent(1)
            .default_length_scale(length_scale)
            .output_variances(vec![vec![1.0], vec![4.0]])
            .noise_variance(noise);

        let fitted = cp
            .fit(&vec![x1, x2.clone()], &vec![y1, y2.clone()])
            .expect("fit should succeed");

        // (a) Structural check: nontrivially nonzero cross-covariance. Two
        // independent single-output GPs would have cross-covariance identically
        // zero everywhere.
        let probe = array![1.0];
        let cross = fitted
            .induced_covariance(0, 1, &probe.view(), &probe.view())
            .expect("induced_covariance should succeed");
        assert!(
            cross.abs() > 0.5,
            "cross-covariance between correlated outputs should be substantial, got {}",
            cross
        );

        // (b) Quantitative check: information sharing improves the sparse output's
        // predictions far from its own data.
        let x_star = array![[2.6]];
        let true_value = 2.0 * 2.6_f64.sin();

        let empty_output_0 = Array2::<f64>::zeros((0, 1));
        let joint_pred = fitted
            .predict_with_std(&[empty_output_0, x_star.clone()])
            .expect("predict should succeed");
        let cp_mean = joint_pred[1].0[0];

        let standalone_gpr = GaussianProcessRegressor::new()
            .kernel(Box::new(RBF::new(length_scale)))
            .alpha(noise);
        let standalone_fitted = standalone_gpr.fit(&x2, &y2).expect("fit should succeed");
        let standalone_mean = standalone_fitted
            .predict(&x_star)
            .expect("predict should succeed")[0];

        let cp_error = (cp_mean - true_value).abs();
        let standalone_error = (standalone_mean - true_value).abs();

        assert!(
            cp_error < standalone_error,
            "convolution process should predict the sparse output more accurately \
             by borrowing strength from the correlated dense output (cp_error = {}, \
             standalone_error = {}, true_value = {}, cp_mean = {}, standalone_mean = {})",
            cp_error,
            standalone_error,
            true_value,
            cp_mean,
            standalone_mean
        );
    }
}
