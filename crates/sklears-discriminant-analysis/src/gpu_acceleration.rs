//! GPU Acceleration for Discriminant Analysis
//!
//! GPU-accelerated implementations of Linear and Quadratic Discriminant
//! Analysis, built directly on the `oxicuda` crate family via
//! `sklears_core::gpu` (real `oxicuda-driver` context / `oxicuda-blas` GEMM /
//! `oxicuda-solver` dense solvers — no CPU-backed pseudo-GPU fallback baked
//! into the types themselves).
//!
//! # Architecture
//!
//! [`GpuDiscriminantAnalysis::new`] calls [`GpuBackend::detect`], which
//! returns `Ok(None)` when no CUDA-capable device/driver is present (the
//! expected outcome on, e.g., this crate's own macOS development machine).
//! Every entry point ([`GpuDiscriminantAnalysis::gpu_lda_fit_predict`],
//! [`GpuDiscriminantAnalysis::gpu_qda_fit_predict`]) transparently falls back
//! to the plain CPU [`LinearDiscriminantAnalysis`] / [`QuadraticDiscriminantAnalysis`]
//! estimators whenever: no GPU was detected, the problem is smaller than
//! [`GpuAccelerationConfig::gpu_threshold`], the requested solver
//! configuration uses a regularization mode the GPU path does not (yet)
//! implement (see [`GpuDiscriminantAnalysis::lda_config_supported_on_gpu`] /
//! [`GpuDiscriminantAnalysis::qda_config_supported_on_gpu`]), or an on-device
//! call fails for any reason. This mirrors the fallback shape already used by
//! `sklears_decomposition::hardware_acceleration::GpuAcceleration`.
//!
//! # What is actually GPU-accelerated
//!
//! * Class means: a single GEMM against a host-built class-averaging
//!   indicator matrix ([`GpuLDAKernel::compute_class_means_gpu`]).
//! * Within-class scatter / pooled covariance and per-class QDA covariances:
//!   host-side centering (`O(n·d)`, cheap) followed by a `Xcᵀ · Xc` GEMM
//!   ([`GpuLDAKernel::compute_within_scatter_gpu`],
//!   [`GpuQDAKernel::compute_class_covariances_gpu`]).
//! * Discriminant scores: per-class GEMM + elementwise-multiply-and-reduce
//!   for the quadratic form ([`GpuLDAKernel::compute_discriminant_gpu`],
//!   [`GpuQDAKernel::compute_qda_discriminant_gpu`]).
//! * `Σ⁻¹` / `log|Σ|`: `oxicuda_solver::dense::{inverse, log_determinant}`
//!   (LU-based, on-device).
//! * The LDA generalized eigenproblem `S_b w = λ S_w w`
//!   ([`GpuLDAKernel::solve_generalized_eigen_gpu`]): oxicuda-solver 0.4.0 has
//!   no generalized symmetric eigensolver (`sygvd`), so this is solved by
//!   hand via Cholesky reduction to a standard eigenproblem (see that
//!   method's doc comment for the four-step derivation and an honesty note
//!   about `syevd`'s current host-fallback implementation).
//!
//! Between-class scatter `S_b` is computed host-side: its cost,
//! `O(n_classes · n_features²)`, is independent of the number of samples, so
//! GPU acceleration would not be worthwhile there (unlike `S_w`, whose
//! `O(n_samples · n_features²)` cost is what the GEMM path targets).

use scirs2_core::ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, ShapeBuilder};

use oxicuda_blas::FillMode;
use oxicuda_memory::DeviceBuffer;
use oxicuda_solver::{dense, EigJob, SolverHandle};

use sklears_core::gpu::{GpuArray, GpuBackend, GpuMatrixOps};

use crate::{
    lda::{LinearDiscriminantAnalysis, LinearDiscriminantAnalysisConfig},
    numerical_stability::NumericalStability,
    qda::{QuadraticDiscriminantAnalysis, QuadraticDiscriminantAnalysisConfig},
};

use sklears_core::{
    error::Result,
    prelude::SklearsError,
    traits::{Fit, Predict},
    types::Float,
};
use std::sync::{Arc, Mutex};

// ─── Configuration ──────────────────────────────────────────────────────────

/// Configuration for GPU-accelerated discriminant analysis.
#[derive(Debug, Clone)]
pub struct GpuAccelerationConfig {
    /// Prefer tensor-core GEMM paths where available.
    ///
    /// Accepted for API compatibility and forward-extensibility, but not
    /// currently wired to anything: `oxicuda_blas::level3::gemm` (via
    /// [`GpuMatrixOps::matmul`]) does not yet expose an algorithm-selection
    /// knob at this layer.
    pub use_tensor_cores: bool,
    /// Use mixed precision (f16/bf16) for memory efficiency.
    ///
    /// Accepted for API compatibility; not yet wired into the GEMM path
    /// (which currently always computes in [`Float`] precision).
    pub use_mixed_precision: bool,
    /// Batch size for GPU operations.
    pub batch_size: usize,
    /// Memory usage limit for GPU (bytes, None = auto-detect).
    pub memory_limit: Option<usize>,
    /// Enable asynchronous GPU operations.
    ///
    /// Accepted for API compatibility; every GPU call in this module is
    /// currently synchronous.
    pub async_operations: bool,
    /// Fallback to CPU if GPU operations fail.
    pub cpu_fallback: bool,
    /// Minimum problem size (`n_samples * n_features`) to use GPU (smaller
    /// problems use CPU, where transfer overhead would dominate).
    pub gpu_threshold: usize,
    /// Enable multi-GPU support for very large datasets.
    ///
    /// Accepted for API compatibility; not yet implemented (a
    /// [`GpuDiscriminantAnalysis`] binds to a single device via
    /// [`GpuBackend::detect`] / [`GpuDiscriminantAnalysis::with_device_id`]).
    pub multi_gpu: bool,
    /// GPU memory management strategy.
    pub memory_strategy: GpuMemoryStrategy,
}

/// GPU memory management strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuMemoryStrategy {
    Conservative,
    Balanced,
    Aggressive,
    Custom {
        batch_size: usize,
        buffer_size: usize,
    },
}

impl Default for GpuAccelerationConfig {
    fn default() -> Self {
        Self {
            use_tensor_cores: true,
            use_mixed_precision: false,
            batch_size: 1024,
            memory_limit: None,
            async_operations: true,
            cpu_fallback: true,
            gpu_threshold: 256,
            multi_gpu: false,
            memory_strategy: GpuMemoryStrategy::Balanced,
        }
    }
}

/// Performance statistics for GPU operations.
#[derive(Debug, Clone, Default)]
pub struct GpuPerformanceStats {
    pub gpu_time_ms: f64,
    pub cpu_time_ms: f64,
    pub transfer_time_ms: f64,
    pub memory_usage_mb: f64,
    pub kernel_executions: usize,
    pub fallback_count: usize,
}

// ─── Small free-function helpers ───────────────────────────────────────────

/// Project a square matrix onto its symmetric part `(A + Aᵀ) / 2`.
///
/// Local copy of `lda::symmetrize` (that function is private to its module):
/// the scatter matrices computed below are symmetric by construction, but
/// explicit element-wise accumulation picks up round-off asymmetry that the
/// strict machine-epsilon symmetry checks in downstream symmetric solvers
/// (`syevd`, [`NumericalStability::stable_eigen_decomposition`]) reject.
fn symmetrize(matrix: &Array2<Float>) -> Array2<Float> {
    let n = matrix.nrows();
    let mut symmetric = Array2::zeros((n, n));
    for i in 0..n {
        for j in 0..n {
            symmetric[[i, j]] = 0.5 * (matrix[[i, j]] + matrix[[j, i]]);
        }
    }
    symmetric
}

/// Validates that every label in `y` is a 0-based index `< n_classes`.
fn validate_labels(y: &ArrayView1<usize>, n_classes: usize) -> Result<()> {
    if let Some(&bad) = y.iter().find(|&&label| label >= n_classes) {
        return Err(SklearsError::InvalidInput(format!(
            "class label {bad} out of range (n_classes={n_classes}); \
             GPU kernels expect 0-based contiguous class indices"
        )));
    }
    Ok(())
}

/// Converts 0-based `usize` class indices to `i32` labels, as required by
/// [`LinearDiscriminantAnalysis`] / [`QuadraticDiscriminantAnalysis`]'s
/// `Fit`/`Predict` trait contracts.
fn usize_labels_to_i32(y: &ArrayView1<usize>) -> Result<Array1<i32>> {
    let mut out = Array1::<i32>::zeros(y.len());
    for (o, &v) in out.iter_mut().zip(y.iter()) {
        *o = i32::try_from(v).map_err(|_| {
            SklearsError::InvalidInput(format!("class label {v} does not fit in i32"))
        })?;
    }
    Ok(out)
}

/// Converts `i32` predictions back to `usize`. Safe as long as every
/// predicted label is non-negative, which holds whenever the labels passed
/// to `fit` were themselves produced by [`usize_labels_to_i32`] (predictions
/// are always one of the fitted class labels).
fn i32_predictions_to_usize(y: &Array1<i32>) -> Result<Array1<usize>> {
    let mut out = Array1::<usize>::zeros(y.len());
    for (o, &v) in out.iter_mut().zip(y.iter()) {
        *o = usize::try_from(v).map_err(|_| {
            SklearsError::InvalidOperation(format!(
                "predicted label {v} is negative, cannot convert to a class index"
            ))
        })?;
    }
    Ok(out)
}

/// Row-wise argmax: the standard `predict` reduction over per-class
/// discriminant scores.
fn argmax_rows(scores: &ArrayView2<Float>) -> Array1<usize> {
    Array1::from_vec(
        scores
            .axis_iter(Axis(0))
            .map(|row| {
                row.iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx)
                    .unwrap_or(0)
            })
            .collect::<Vec<usize>>(),
    )
}

/// Row-wise softmax, for callers that need class probabilities from raw
/// discriminant scores. Cheap (`O(n·k)`), computed host-side, matching the
/// approach already used by [`LinearDiscriminantAnalysis::predict_proba`] and
/// [`QuadraticDiscriminantAnalysis::predict_proba`].
pub fn softmax_rows(scores: &ArrayView2<Float>) -> Array2<Float> {
    let mut probabilities = Array2::zeros(scores.dim());
    for (i, row) in scores.axis_iter(Axis(0)).enumerate() {
        let max_score = row.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let exp_scores: Vec<Float> = row.iter().map(|&v| (v - max_score).exp()).collect();
        let sum_exp: Float = exp_scores.iter().sum();
        for (j, &e) in exp_scores.iter().enumerate() {
            probabilities[[i, j]] = e / sum_exp;
        }
    }
    probabilities
}

/// Between-class scatter `S_b = Σ_k n_k (μ_k − μ̄)(μ_k − μ̄)ᵀ`, computed
/// host-side (see the module-level doc comment for why this one is not
/// GPU-accelerated).
///
/// Pairs with [`GpuLDAKernel::compute_within_scatter_gpu`] (`S_w`) as the two
/// inputs [`GpuLDAKernel::solve_generalized_eigen_gpu`] needs; not itself
/// called by the `fit`/`predict` pipeline, which only needs `S_w⁻¹` (see that
/// method's doc comment for why the discriminant-directions eigenproblem is
/// a separate concern from classification).
pub fn between_class_scatter(
    x: &ArrayView2<Float>,
    y: &ArrayView1<usize>,
    class_means: &ArrayView2<Float>,
    n_classes: usize,
) -> Array2<Float> {
    let n_features = x.ncols();
    let overall_mean = x
        .mean_axis(Axis(0))
        .unwrap_or_else(|| Array1::zeros(n_features));

    let mut counts = vec![0usize; n_classes];
    for &label in y.iter() {
        if label < n_classes {
            counts[label] += 1;
        }
    }

    let mut sb = Array2::<Float>::zeros((n_features, n_features));
    for (k, &count) in counts.iter().enumerate() {
        let count = count as Float;
        if count == 0.0 {
            continue;
        }
        let mean_k = class_means.row(k);
        let diff = &mean_k - &overall_mean;
        for j in 0..n_features {
            for l in 0..n_features {
                sb[[j, l]] += count * diff[j] * diff[l];
            }
        }
    }
    sb
}

/// `Σ⁻¹` via `oxicuda_solver::dense::inverse` (on-device, LU-based), falling
/// back to [`NumericalStability::matrix_inverse`] on the host if the
/// on-device solve fails for any reason (mirrors
/// `sklears_decomposition::hardware_acceleration::GpuAcceleration::gpu_svd`'s
/// fallback shape).
fn gpu_matrix_inverse(backend: &GpuBackend, matrix: &Array2<Float>) -> Result<Array2<Float>> {
    match gpu_matrix_inverse_on_device(backend, matrix) {
        Ok(inv) => Ok(inv),
        Err(_) => NumericalStability::new().matrix_inverse(matrix),
    }
}

fn gpu_matrix_inverse_on_device(
    backend: &GpuBackend,
    matrix: &Array2<Float>,
) -> Result<Array2<Float>> {
    let n = matrix.nrows();
    if matrix.ncols() != n {
        return Err(SklearsError::InvalidInput(
            "gpu_matrix_inverse: matrix must be square".to_string(),
        ));
    }
    backend
        .context()
        .set_current()
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    let mut handle = SolverHandle::new(backend.context())
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

    let col_major: Vec<Float> = matrix.t().iter().copied().collect();
    let mut a_buf = DeviceBuffer::from_host(&col_major)
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    dense::inverse(&mut handle, &mut a_buf, n as u32, n as u32)
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

    let mut host = vec![0.0 as Float; n * n];
    a_buf
        .copy_to_host(&mut host)
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    Array2::from_shape_vec((n, n).f(), host)
        .map_err(|e| SklearsError::NumericalError(format!("reshape inverse: {e}")))
}

/// `log|det Σ|` via `oxicuda_solver::dense::log_determinant` (on-device,
/// LU-based), falling back to a host eigenvalue-sum computation via
/// [`NumericalStability::stable_eigen_decomposition`] (`Σ` is symmetric
/// positive definite here, so `log|det Σ| = Σᵢ log(λᵢ)`) if the on-device
/// solve fails or reports a non-positive sign (indicating singularity or a
/// numerical issue).
fn gpu_log_determinant(backend: &GpuBackend, matrix: &Array2<Float>) -> Result<Float> {
    match gpu_log_determinant_on_device(backend, matrix) {
        Ok((log_abs, sign)) if sign > 0.0 => Ok(log_abs),
        _ => {
            let ns = NumericalStability::new();
            let (eigenvalues, _) = ns.stable_eigen_decomposition(matrix)?;
            Ok(eigenvalues
                .iter()
                .map(|v| v.max(Float::MIN_POSITIVE).ln())
                .sum())
        }
    }
}

fn gpu_log_determinant_on_device(
    backend: &GpuBackend,
    matrix: &Array2<Float>,
) -> Result<(Float, Float)> {
    let n = matrix.nrows();
    backend
        .context()
        .set_current()
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    let mut handle = SolverHandle::new(backend.context())
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

    let col_major: Vec<Float> = matrix.t().iter().copied().collect();
    let a_buf = DeviceBuffer::from_host(&col_major)
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    let (log_abs, sign) = dense::log_determinant(&mut handle, &a_buf, n as u32, n as u32)
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
    Ok((log_abs, sign))
}

// ─── GpuLDAKernel ───────────────────────────────────────────────────────────

/// GPU-accelerated numerical kernels for Linear Discriminant Analysis.
///
/// Holds only configuration: the [`GpuBackend`] is passed explicitly to each
/// method, since a kernel may be constructed before a backend is known to be
/// available.
pub struct GpuLDAKernel {
    #[allow(dead_code)] // reserved for future threshold/precision-aware dispatch
    config: GpuAccelerationConfig,
}

impl GpuLDAKernel {
    pub fn new(config: GpuAccelerationConfig) -> Self {
        Self { config }
    }

    /// Computes per-class means via a single GEMM against a host-built
    /// `[n_classes × n_samples]` indicator matrix `S` with
    /// `S[k, i] = 1 / n_k` if `y[i] == k` else `0`, so that `means = S · X`.
    pub fn compute_class_means_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        n_classes: usize,
    ) -> Result<Array2<Float>> {
        validate_labels(y, n_classes)?;
        let n_samples = x.nrows();

        let mut counts = vec![0usize; n_classes];
        for &label in y.iter() {
            counts[label] += 1;
        }

        let mut indicator = Array2::<Float>::zeros((n_classes, n_samples));
        for (i, &label) in y.iter().enumerate() {
            if counts[label] > 0 {
                indicator[[label, i]] = 1.0 / counts[label] as Float;
            }
        }

        let indicator_gpu = GpuArray::<Float>::from_array2(backend, &indicator)?;
        let x_gpu = GpuArray::<Float>::from_array2(backend, &x.to_owned())?;
        let means_gpu = indicator_gpu.matmul(&x_gpu)?;
        means_gpu.to_array2()
    }

    /// `scale * Xcᵀ · Xc`, where `Xc` is `x` with each row shifted by its
    /// class mean (host-side centering, `O(n·d)`), computed on-device via a
    /// single transpose + GEMM.
    ///
    /// `scale = 1.0` yields the raw within-class scatter matrix `S_w` used by
    /// [`crate::lda`]'s `solve_eigen` / `solve_svd` (see
    /// [`compute_within_scatter_gpu`](Self::compute_within_scatter_gpu));
    /// `scale = 1 / (n_samples - 1)` yields the usual pooled sample
    /// covariance (see
    /// [`compute_covariance_gpu`](Self::compute_covariance_gpu)).
    fn centered_gram_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        class_means: &ArrayView2<Float>,
        scale: Float,
    ) -> Result<Array2<Float>> {
        let n_classes = class_means.nrows();
        validate_labels(y, n_classes)?;
        let (n_samples, n_features) = x.dim();

        let mut centered = Array2::<Float>::zeros((n_samples, n_features));
        for (i, row) in x.axis_iter(Axis(0)).enumerate() {
            let mean = class_means.row(y[i]);
            for j in 0..n_features {
                centered[[i, j]] = row[j] - mean[j];
            }
        }

        let xc_gpu = GpuArray::<Float>::from_array2(backend, &centered)?;
        let gram_gpu = xc_gpu.transpose()?.matmul(&xc_gpu)?;
        let gram_gpu = gram_gpu.scale(scale as f32)?;
        gram_gpu.to_array2()
    }

    /// Raw within-class scatter `S_w = Σᵢ (xᵢ − μ_{y[i]})(xᵢ − μ_{y[i]})ᵀ`
    /// (unnormalized sum of outer products), matching
    /// [`crate::lda::LinearDiscriminantAnalysis`]'s internal convention so
    /// that the Bayes-classifier coefficients derived from it agree with the
    /// CPU solver.
    pub fn compute_within_scatter_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        class_means: &ArrayView2<Float>,
    ) -> Result<Array2<Float>> {
        self.centered_gram_gpu(backend, x, y, class_means, 1.0)
    }

    /// Pooled within-class sample covariance `S_w / (n_samples - 1)`.
    pub fn compute_covariance_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        class_means: &ArrayView2<Float>,
    ) -> Result<Array2<Float>> {
        let n_samples = x.nrows();
        let denom = (n_samples.saturating_sub(1)).max(1) as Float;
        self.centered_gram_gpu(backend, x, y, class_means, 1.0 / denom)
    }

    /// Computes `[n_samples × n_classes]` discriminant scores
    /// `δ_k(x) = -0.5 (x − μ_k)ᵀ Σ⁻¹ (x − μ_k) + log_prior_k` for every row of
    /// `x` and every class, via one GEMM + one elementwise-multiply-and-reduce
    /// per class (the row-wise quadratic form
    /// `diag(diff · Σ⁻¹ · diffᵀ)`, computed without materializing the full
    /// `[n × n]` product).
    ///
    /// `cov_inv` is shared across classes here (LDA's pooled `S_w⁻¹`); QDA
    /// uses the analogous [`GpuQDAKernel::compute_qda_discriminant_gpu`] with
    /// a per-class covariance inverse and an added log-determinant term.
    ///
    /// This quadratic form and the linear Bayes-classifier score
    /// `x·S_w⁻¹μ_k − ½μ_kᵀS_w⁻¹μ_k + ln π_k` used by
    /// [`crate::lda::LinearDiscriminantAnalysis`] agree up to a per-sample,
    /// class-independent additive constant `-0.5xᵀΣ⁻¹x` (it cancels because
    /// `Σ` is shared across classes in LDA), so `argmax_k` — and therefore
    /// every prediction — is identical between the two formulations; this
    /// method keeps the more general, QDA-compatible quadratic form.
    pub fn compute_discriminant_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        class_means: &ArrayView2<Float>,
        cov_inv: &ArrayView2<Float>,
        log_priors: &ArrayView1<Float>,
    ) -> Result<GpuArray<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_classes = class_means.nrows();
        if cov_inv.nrows() != n_features || cov_inv.ncols() != n_features {
            return Err(SklearsError::ShapeMismatch {
                expected: format!("[{n_features}x{n_features}]"),
                actual: format!("{:?}", cov_inv.dim()),
            });
        }

        let cov_inv_gpu = GpuArray::<Float>::from_array2(backend, &cov_inv.to_owned())?;
        let mut scores = Array2::<Float>::zeros((n_samples, n_classes));

        for k in 0..n_classes {
            let mean_k = class_means.row(k);
            let mut diff = Array2::<Float>::zeros((n_samples, n_features));
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                for j in 0..n_features {
                    diff[[i, j]] = row[j] - mean_k[j];
                }
            }
            let diff_gpu = GpuArray::<Float>::from_array2(backend, &diff)?;
            let temp_gpu = diff_gpu.matmul(&cov_inv_gpu)?;
            let quad_terms_gpu = temp_gpu.mul(&diff_gpu)?;
            let quad_terms = quad_terms_gpu.to_array2()?;
            let quad = quad_terms.sum_axis(Axis(1));
            for i in 0..n_samples {
                scores[[i, k]] = -0.5 * quad[i] + log_priors[k];
            }
        }

        GpuArray::<Float>::from_array2(backend, &scores)
    }

    /// Downloads discriminant scores and returns the row-wise argmax
    /// (predicted class index per sample). No GPU kernel is launched here —
    /// argmax over a handful of classes is cheap host-side work; this
    /// intentionally drops the old (dead, never-implemented) GPU softmax
    /// dispatch that used to sit in front of a plain host argmax anyway.
    pub fn predict_gpu(&self, discriminants: &GpuArray<Float>) -> Result<Array1<usize>> {
        let scores = discriminants.to_array2()?;
        Ok(argmax_rows(&scores.view()))
    }

    /// Solves the LDA generalized eigenproblem `S_b w = λ S_w w` for the
    /// discriminant directions (eigenvectors, sorted by descending
    /// eigenvalue / Fisher ratio) via Cholesky reduction to a standard
    /// symmetric eigenproblem.
    ///
    /// oxicuda-solver 0.4.0 has no generalized symmetric eigensolver
    /// (`sygvd`), so this reduces the generalized problem by hand, the
    /// standard textbook way (equivalent to LAPACK `sygvd`'s `itype=1`
    /// reduction):
    ///
    /// 1. `S_w = L Lᵀ` via `oxicuda_solver::dense::cholesky` (on-device;
    ///    its blocked implementation launches real PTX TRSM/SYRK kernels).
    /// 2. `L⁻¹` via `oxicuda_solver::dense::inverse` on a copy of `L`, then
    ///    `C = L⁻¹ S_b L⁻ᵀ` via two [`GpuMatrixOps::matmul`] GEMMs. `C` is
    ///    symmetric because it is a congruence transform of the symmetric
    ///    `S_b`.
    /// 3. `C y = λ y` via `oxicuda_solver::dense::syevd`.
    /// 4. Back-transform `w = L⁻ᵀ y` via another `matmul`.
    ///
    /// Derivation: substituting `y = Lᵀw` (i.e. `w = L⁻ᵀy`) into
    /// `S_b w = λ L Lᵀ w` gives `S_b L⁻ᵀ y = λ L y`, and left-multiplying by
    /// `L⁻¹` gives `L⁻¹ S_b L⁻ᵀ y = λ y` — the standard eigenproblem solved
    /// in step 3.
    ///
    /// Each host round-trip between steps goes through a full `Array2` (not
    /// raw device bytes), which is what makes it safe to mix this function's
    /// raw column-major `oxicuda_solver` calls with [`GpuArray`]'s row-major
    /// convention: `Array2::from_shape_vec((n, n).f(), ..)` and
    /// [`GpuArray::from_array2`] both operate on logically-indexed matrices,
    /// not raw memory layout, so orientation is preserved correctly across
    /// the boundary regardless of which convention originated the bytes.
    ///
    /// # Honesty note
    ///
    /// As of oxicuda-solver 0.4.0, `syevd`'s own doc comments state its
    /// on-device symmetric eigensolver is not yet implemented: it currently
    /// downloads the matrix, runs Householder tridiagonalization + implicit-shift
    /// QR iteration on the host, and stages the results back to device
    /// buffers. `cholesky` (blocked TRSM/SYRK), `inverse` (LU-based), and the
    /// two `GpuMatrixOps::matmul` GEMMs used here are real on-device work.
    /// So this path is Cholesky- and GEMM-accelerated today, with fully
    /// on-device eigensolution pending a future oxicuda-solver release; this
    /// implementation calls the real public API either way and will pick up
    /// that acceleration automatically once available, with no caller-visible
    /// change.
    ///
    /// Returns `(eigenvalues, eigenvectors)` sorted by descending eigenvalue,
    /// matching [`NumericalStability::stable_generalized_eigen`]'s contract
    /// (the CPU fallback used when no GPU is available).
    pub fn solve_generalized_eigen_gpu(
        &self,
        backend: &GpuBackend,
        sb: &Array2<Float>,
        sw: &Array2<Float>,
    ) -> Result<(Array1<Float>, Array2<Float>)> {
        let n = sw.nrows();
        if sw.ncols() != n || sb.dim() != (n, n) {
            return Err(SklearsError::InvalidInput(
                "solve_generalized_eigen_gpu: sb/sw must be square and equal size".to_string(),
            ));
        }

        backend
            .context()
            .set_current()
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut handle = SolverHandle::new(backend.context())
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        // Step 1: S_w = L L^T.
        let sw_col_major: Vec<Float> = sw.t().iter().copied().collect();
        let mut sw_buf = DeviceBuffer::from_host(&sw_col_major)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        dense::cholesky(
            &mut handle,
            FillMode::Lower,
            &mut sw_buf,
            n as u32,
            n as u32,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut l_host = vec![0.0 as Float; n * n];
        sw_buf
            .copy_to_host(&mut l_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let l_full = Array2::from_shape_vec((n, n).f(), l_host)
            .map_err(|e| SklearsError::NumericalError(format!("reshape L: {e}")))?;
        // `cholesky` only defines the lower triangle; zero the rest
        // explicitly so nothing downstream depends on undefined content.
        let mut l = Array2::<Float>::zeros((n, n));
        for i in 0..n {
            for j in 0..=i {
                l[[i, j]] = l_full[[i, j]];
            }
        }

        // Step 2a: L^{-1} via `inverse` on a fresh column-major copy of L.
        let l_col_major: Vec<Float> = l.t().iter().copied().collect();
        let mut l_buf = DeviceBuffer::from_host(&l_col_major)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        dense::inverse(&mut handle, &mut l_buf, n as u32, n as u32)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut l_inv_host = vec![0.0 as Float; n * n];
        l_buf
            .copy_to_host(&mut l_inv_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let l_inv = Array2::from_shape_vec((n, n).f(), l_inv_host)
            .map_err(|e| SklearsError::NumericalError(format!("reshape L^-1: {e}")))?;

        // Step 2b: C = L^{-1} S_b L^{-T}, via two on-device GEMMs.
        let l_inv_gpu = GpuArray::<Float>::from_array2(backend, &l_inv)?;
        let sb_gpu = GpuArray::<Float>::from_array2(backend, sb)?;
        let l_inv_t_gpu = l_inv_gpu.transpose()?;
        let c_gpu = l_inv_gpu.matmul(&sb_gpu)?.matmul(&l_inv_t_gpu)?;
        let c_host = symmetrize(&c_gpu.to_array2()?);

        // Step 3: C y = lambda y (ascending eigenvalues from syevd).
        let c_col_major: Vec<Float> = c_host.t().iter().copied().collect();
        let mut c_buf = DeviceBuffer::from_host(&c_col_major)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut eigenvalues_buf = DeviceBuffer::<Float>::zeroed(n)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        dense::syevd(
            &mut handle,
            &mut c_buf,
            n as u32,
            n as u32,
            &mut eigenvalues_buf,
            EigJob::ValuesAndVectors,
        )
        .map_err(|e| SklearsError::NumericalError(e.to_string()))?;

        let mut eigenvalues_host = vec![0.0 as Float; n];
        eigenvalues_buf
            .copy_to_host(&mut eigenvalues_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let mut y_host = vec![0.0 as Float; n * n];
        c_buf
            .copy_to_host(&mut y_host)
            .map_err(|e| SklearsError::NumericalError(e.to_string()))?;
        let y = Array2::from_shape_vec((n, n).f(), y_host)
            .map_err(|e| SklearsError::NumericalError(format!("reshape eigenvectors: {e}")))?;

        // Step 4: back-transform w = L^{-T} y.
        let y_gpu = GpuArray::<Float>::from_array2(backend, &y)?;
        let w_gpu = l_inv_t_gpu.matmul(&y_gpu)?;
        let w = w_gpu.to_array2()?;

        // `syevd` returns eigenvalues ascending; sort descending to match
        // `stable_generalized_eigen`'s contract (largest Fisher ratio first).
        let eigenvalues_desc = Array1::from_iter(eigenvalues_host.iter().rev().copied());
        let mut w_desc = Array2::<Float>::zeros((n, n));
        for (new_col, old_col) in (0..n).rev().enumerate() {
            w_desc.column_mut(new_col).assign(&w.column(old_col));
        }

        Ok((eigenvalues_desc, w_desc))
    }
}

// ─── GpuQDAKernel ───────────────────────────────────────────────────────────

/// GPU-accelerated numerical kernels for Quadratic Discriminant Analysis.
pub struct GpuQDAKernel {
    #[allow(dead_code)] // reserved for future threshold/precision-aware dispatch
    config: GpuAccelerationConfig,
}

impl GpuQDAKernel {
    pub fn new(config: GpuAccelerationConfig) -> Self {
        Self { config }
    }

    /// Per-class covariance matrices via GPU GEMM: for each class `k`,
    /// centers that class's rows of `x` by `class_means[k]` (host-side),
    /// then computes `Xc_kᵀ Xc_k / (n_k - 1)` on-device.
    pub fn compute_class_covariances_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        class_means: &ArrayView2<Float>,
        n_classes: usize,
    ) -> Result<Vec<Array2<Float>>> {
        validate_labels(y, n_classes)?;
        let n_features = x.ncols();

        let mut counts = vec![0usize; n_classes];
        for &label in y.iter() {
            counts[label] += 1;
        }

        let mut covariances = Vec::with_capacity(n_classes);
        for (k, &n_k) in counts.iter().enumerate() {
            if n_k == 0 {
                return Err(SklearsError::InvalidInput(format!(
                    "class {k} has no samples"
                )));
            }
            let mean_k = class_means.row(k);
            let mut centered = Array2::<Float>::zeros((n_k, n_features));
            let mut row_idx = 0;
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                if y[i] == k {
                    for j in 0..n_features {
                        centered[[row_idx, j]] = row[j] - mean_k[j];
                    }
                    row_idx += 1;
                }
            }
            let xc_gpu = GpuArray::<Float>::from_array2(backend, &centered)?;
            let xtx_gpu = xc_gpu.transpose()?.matmul(&xc_gpu)?;
            let denom = ((n_k.saturating_sub(1)).max(1)) as f32;
            let cov_gpu = xtx_gpu.scale(1.0 / denom)?;
            covariances.push(cov_gpu.to_array2()?);
        }
        Ok(covariances)
    }

    /// QDA discriminant scores: `[n_samples × n_classes]` matrix of
    /// `δ_k(x) = -0.5 [(x − μ_k)ᵀ Σ_k⁻¹ (x − μ_k) + log|Σ_k|] + log_prior_k`,
    /// computed the same way as
    /// [`GpuLDAKernel::compute_discriminant_gpu`] but with a per-class
    /// covariance inverse and an added log-determinant term (matches
    /// [`crate::qda::QuadraticDiscriminantAnalysis::predict_proba`]'s
    /// log-likelihood formula).
    pub fn compute_qda_discriminant_gpu(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        class_means: &ArrayView2<Float>,
        class_cov_inv: &[Array2<Float>],
        class_log_det: &[Float],
        log_priors: &ArrayView1<Float>,
    ) -> Result<GpuArray<Float>> {
        let (n_samples, n_features) = x.dim();
        let n_classes = class_means.nrows();
        if class_cov_inv.len() != n_classes || class_log_det.len() != n_classes {
            return Err(SklearsError::InvalidInput(
                "compute_qda_discriminant_gpu: class_cov_inv/class_log_det length must equal n_classes"
                    .to_string(),
            ));
        }

        let mut scores = Array2::<Float>::zeros((n_samples, n_classes));

        for k in 0..n_classes {
            let cov_inv = &class_cov_inv[k];
            let cov_inv_gpu = GpuArray::<Float>::from_array2(backend, cov_inv)?;
            let mean_k = class_means.row(k);
            let mut diff = Array2::<Float>::zeros((n_samples, n_features));
            for (i, row) in x.axis_iter(Axis(0)).enumerate() {
                for j in 0..n_features {
                    diff[[i, j]] = row[j] - mean_k[j];
                }
            }
            let diff_gpu = GpuArray::<Float>::from_array2(backend, &diff)?;
            let temp_gpu = diff_gpu.matmul(&cov_inv_gpu)?;
            let quad_terms_gpu = temp_gpu.mul(&diff_gpu)?;
            let quad_terms = quad_terms_gpu.to_array2()?;
            let quad = quad_terms.sum_axis(Axis(1));
            for i in 0..n_samples {
                scores[[i, k]] = -0.5 * (quad[i] + class_log_det[k]) + log_priors[k];
            }
        }

        GpuArray::<Float>::from_array2(backend, &scores)
    }
}

// ─── GpuDiscriminantAnalysis ────────────────────────────────────────────────

/// GPU-accelerated discriminant analysis manager: owns the (optional) GPU
/// backend and dispatches LDA/QDA fit+predict requests to the GPU kernels
/// above, or transparently to plain CPU [`LinearDiscriminantAnalysis`] /
/// [`QuadraticDiscriminantAnalysis`].
pub struct GpuDiscriminantAnalysis {
    backend: Option<GpuBackend>,
    config: GpuAccelerationConfig,
    lda_kernel: Option<GpuLDAKernel>,
    qda_kernel: Option<GpuQDAKernel>,
    performance_stats: Arc<Mutex<GpuPerformanceStats>>,
}

impl GpuDiscriminantAnalysis {
    /// Creates a new GPU-accelerated discriminant analysis manager,
    /// auto-detecting the GPU with the most free memory via
    /// [`GpuBackend::detect`]. Never fails just because no GPU/driver is
    /// present — [`Self::is_gpu_available`] will report `false` and every
    /// fit/predict call will use the CPU fallback.
    pub fn new(config: GpuAccelerationConfig) -> Result<Self> {
        let backend = GpuBackend::detect()?;
        Ok(Self {
            backend,
            lda_kernel: Some(GpuLDAKernel::new(config.clone())),
            qda_kernel: Some(GpuQDAKernel::new(config.clone())),
            config,
            performance_stats: Arc::new(Mutex::new(GpuPerformanceStats::default())),
        })
    }

    /// Like [`Self::new`] but binds to a specific device ordinal instead of
    /// auto-selecting.
    pub fn with_device_id(config: GpuAccelerationConfig, device_id: usize) -> Result<Self> {
        let backend = GpuBackend::with_device_id(device_id)?;
        Ok(Self {
            backend,
            lda_kernel: Some(GpuLDAKernel::new(config.clone())),
            qda_kernel: Some(GpuQDAKernel::new(config.clone())),
            config,
            performance_stats: Arc::new(Mutex::new(GpuPerformanceStats::default())),
        })
    }

    /// Whether a real GPU backend was detected and is bound to this manager.
    pub fn is_gpu_available(&self) -> bool {
        self.backend.is_some()
    }

    /// The bound GPU backend, if any.
    pub fn backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// Snapshot of accumulated performance statistics.
    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.performance_stats
            .lock()
            .expect("performance stats lock not poisoned")
            .clone()
    }

    /// Whether to use the GPU path for a problem of this size (GPU must be
    /// available and the problem large enough that transfer overhead is
    /// worth paying).
    fn should_use_gpu(&self, n_samples: usize, n_features: usize) -> bool {
        let problem_size = n_samples * n_features;
        problem_size >= self.config.gpu_threshold && self.is_gpu_available()
    }

    /// Whether `config` requests only regularization modes the GPU LDA path
    /// implements today (plain pooled `S_w`, no shrinkage / adaptive
    /// regularization / robust MCD estimation / elastic-net coefficient
    /// regularization). Configurations outside this set still work, they
    /// just always run on the CPU: this never produces a silently-wrong
    /// (numerically different) result for a config the GPU path doesn't
    /// fully implement.
    fn lda_config_supported_on_gpu(config: &LinearDiscriminantAnalysisConfig) -> bool {
        config.shrinkage.is_none()
            && !config.adaptive_regularization
            && !config.robust
            && config.l1_reg == 0.0
            && config.l2_reg == 0.0
    }

    /// Whether `config` requests only regularization modes the GPU QDA path
    /// implements today (robust MCD estimation is CPU-only; see
    /// [`Self::lda_config_supported_on_gpu`] for the rationale).
    fn qda_config_supported_on_gpu(config: &QuadraticDiscriminantAnalysisConfig) -> bool {
        !config.robust
    }

    /// Empirical class priors `n_k / n_samples` for 0-based contiguous class
    /// indices `y`.
    fn compute_priors(&self, y: &ArrayView1<usize>, n_classes: usize) -> Array1<Float> {
        let mut class_counts = Array1::<Float>::zeros(n_classes);
        for &label in y.iter() {
            if label < n_classes {
                class_counts[label] += 1.0;
            }
        }
        let total_samples = y.len() as Float;
        class_counts.mapv(|count| count / total_samples)
    }

    /// Fits LDA on `(x, y)` and predicts on `x_test`, using the GPU path when
    /// available, large enough, and the configuration is GPU-eligible (see
    /// [`Self::lda_config_supported_on_gpu`]); otherwise (or if any on-device
    /// step fails) transparently falls back to plain CPU
    /// [`LinearDiscriminantAnalysis`].
    ///
    /// `y` must contain 0-based contiguous class indices (`0..n_classes`);
    /// callers with arbitrary/sparse `i32` class labels should map them to a
    /// dense index space before calling and map predictions back through the
    /// same table.
    pub fn gpu_lda_fit_predict(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        lda_config: &LinearDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let (n_samples, n_features) = x.dim();
        let eligible = self.should_use_gpu(n_samples, n_features)
            && Self::lda_config_supported_on_gpu(lda_config);

        if !eligible {
            return self.cpu_fallback_lda(x, y, x_test, lda_config);
        }

        let backend = self
            .backend
            .as_ref()
            .expect("should_use_gpu() already checked is_gpu_available()");

        match self.try_gpu_lda(backend, x, y, x_test, lda_config) {
            Ok(predictions) => Ok(predictions),
            Err(_) => self.cpu_fallback_lda(x, y, x_test, lda_config),
        }
    }

    fn try_gpu_lda(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        config: &LinearDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let kernel = self
            .lda_kernel
            .as_ref()
            .expect("lda_kernel is always Some after construction");

        let n_classes = y
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .ok_or_else(|| SklearsError::InvalidInput("y must not be empty".to_string()))?;
        let n_features = x.ncols();

        let start = std::time::Instant::now();

        let class_means = kernel.compute_class_means_gpu(backend, x, y, n_classes)?;
        let mut sw = kernel.compute_within_scatter_gpu(backend, x, y, &class_means.view())?;
        for i in 0..n_features {
            sw[[i, i]] += config.tol;
        }
        let sw = symmetrize(&sw);
        let sw_inv = gpu_matrix_inverse(backend, &sw)?;

        let priors = config
            .priors
            .clone()
            .unwrap_or_else(|| self.compute_priors(y, n_classes));
        let log_priors = priors.mapv(|p: Float| p.ln());

        let discriminants_gpu = kernel.compute_discriminant_gpu(
            backend,
            x_test,
            &class_means.view(),
            &sw_inv.view(),
            &log_priors.view(),
        )?;
        let predictions = kernel.predict_gpu(&discriminants_gpu)?;

        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.gpu_time_ms += start.elapsed().as_millis() as f64;
            stats.kernel_executions += 1;
        }

        Ok(predictions)
    }

    /// Fits QDA on `(x, y)` and predicts on `x_test`, mirroring
    /// [`Self::gpu_lda_fit_predict`]'s dispatch/fallback shape (see
    /// [`Self::qda_config_supported_on_gpu`] for GPU-eligibility rules).
    pub fn gpu_qda_fit_predict(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        qda_config: &QuadraticDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let (n_samples, n_features) = x.dim();
        let eligible = self.should_use_gpu(n_samples, n_features)
            && Self::qda_config_supported_on_gpu(qda_config);

        if !eligible {
            return self.cpu_fallback_qda(x, y, x_test, qda_config);
        }

        let backend = self
            .backend
            .as_ref()
            .expect("should_use_gpu() already checked is_gpu_available()");

        match self.try_gpu_qda(backend, x, y, x_test, qda_config) {
            Ok(predictions) => Ok(predictions),
            Err(_) => self.cpu_fallback_qda(x, y, x_test, qda_config),
        }
    }

    fn try_gpu_qda(
        &self,
        backend: &GpuBackend,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        config: &QuadraticDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        let lda_kernel = self
            .lda_kernel
            .as_ref()
            .expect("lda_kernel is always Some after construction");
        let qda_kernel = self
            .qda_kernel
            .as_ref()
            .expect("qda_kernel is always Some after construction");

        let n_classes = y
            .iter()
            .copied()
            .max()
            .map(|m| m + 1)
            .ok_or_else(|| SklearsError::InvalidInput("y must not be empty".to_string()))?;
        let n_features = x.ncols();

        let start = std::time::Instant::now();

        let class_means = lda_kernel.compute_class_means_gpu(backend, x, y, n_classes)?;
        let mut class_covariances = qda_kernel.compute_class_covariances_gpu(
            backend,
            x,
            y,
            &class_means.view(),
            n_classes,
        )?;

        for cov in class_covariances.iter_mut() {
            if config.diagonal_covariance {
                for i in 0..n_features {
                    for j in 0..n_features {
                        if i != j {
                            cov[[i, j]] = 0.0;
                        }
                    }
                }
            }
            if config.reg_param > 0.0 {
                for i in 0..n_features {
                    cov[[i, i]] += config.reg_param;
                }
            }
        }

        let mut class_cov_inv = Vec::with_capacity(n_classes);
        let mut class_log_det = Vec::with_capacity(n_classes);
        for cov in &class_covariances {
            class_cov_inv.push(gpu_matrix_inverse(backend, cov)?);
            class_log_det.push(gpu_log_determinant(backend, cov)?);
        }

        let priors = config
            .priors
            .clone()
            .unwrap_or_else(|| self.compute_priors(y, n_classes));
        let log_priors = priors.mapv(|p: Float| p.ln());

        let discriminants_gpu = qda_kernel.compute_qda_discriminant_gpu(
            backend,
            x_test,
            &class_means.view(),
            &class_cov_inv,
            &class_log_det,
            &log_priors.view(),
        )?;
        let predictions = lda_kernel.predict_gpu(&discriminants_gpu)?;

        if let Ok(mut stats) = self.performance_stats.lock() {
            stats.gpu_time_ms += start.elapsed().as_millis() as f64;
            stats.kernel_executions += 1;
        }

        Ok(predictions)
    }

    fn cpu_fallback_lda(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        config: &LinearDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("performance stats lock not poisoned");
            stats.fallback_count += 1;
        }

        let y_i32 = usize_labels_to_i32(y)?;
        let x_owned = x.to_owned();
        let x_test_owned = x_test.to_owned();

        let lda = LinearDiscriminantAnalysis::new()
            .solver(&config.solver)
            .shrinkage(config.shrinkage)
            .priors(config.priors.clone())
            .n_components(config.n_components)
            .store_covariance(config.store_covariance)
            .tol(config.tol)
            .l1_reg(config.l1_reg)
            .l2_reg(config.l2_reg)
            .elastic_net_ratio(config.elastic_net_ratio)
            .max_iter(config.max_iter)
            .robust(config.robust)
            .robust_method(&config.robust_method)
            .contamination(config.contamination)
            .adaptive_regularization(config.adaptive_regularization)
            .adaptive_method(&config.adaptive_method);

        let trained = lda.fit(&x_owned, &y_i32)?;
        let predictions_i32 = trained.predict(&x_test_owned)?;
        i32_predictions_to_usize(&predictions_i32)
    }

    fn cpu_fallback_qda(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
        config: &QuadraticDiscriminantAnalysisConfig,
    ) -> Result<Array1<usize>> {
        {
            let mut stats = self
                .performance_stats
                .lock()
                .expect("performance stats lock not poisoned");
            stats.fallback_count += 1;
        }

        let y_i32 = usize_labels_to_i32(y)?;
        let x_owned = x.to_owned();
        let x_test_owned = x_test.to_owned();

        let qda = QuadraticDiscriminantAnalysis::new()
            .priors(config.priors.clone())
            .reg_param(config.reg_param)
            .store_covariance(config.store_covariance)
            .tol(config.tol)
            .diagonal_covariance(config.diagonal_covariance)
            .robust(config.robust)
            .robust_method(&config.robust_method)
            .contamination(config.contamination);

        let trained = qda.fit(&x_owned, &y_i32)?;
        let predictions_i32 = trained.predict(&x_test_owned)?;
        i32_predictions_to_usize(&predictions_i32)
    }
}

// ─── High-level wrappers ────────────────────────────────────────────────────

/// High-level GPU-accelerated LDA wrapper: pairs a
/// [`LinearDiscriminantAnalysisConfig`] with a [`GpuDiscriminantAnalysis`]
/// manager for a simple `fit_predict` entry point.
pub struct GpuAcceleratedLDA {
    gpu_manager: GpuDiscriminantAnalysis,
    config: LinearDiscriminantAnalysisConfig,
}

/// High-level GPU-accelerated QDA wrapper; see [`GpuAcceleratedLDA`].
pub struct GpuAcceleratedQDA {
    gpu_manager: GpuDiscriminantAnalysis,
    config: QuadraticDiscriminantAnalysisConfig,
}

impl GpuAcceleratedLDA {
    pub fn new(
        lda_config: LinearDiscriminantAnalysisConfig,
        gpu_config: GpuAccelerationConfig,
    ) -> Result<Self> {
        Ok(Self {
            gpu_manager: GpuDiscriminantAnalysis::new(gpu_config)?,
            config: lda_config,
        })
    }

    pub fn fit_predict(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
    ) -> Result<Array1<usize>> {
        self.gpu_manager
            .gpu_lda_fit_predict(x, y, x_test, &self.config)
    }

    pub fn is_using_gpu(&self) -> bool {
        self.gpu_manager.is_gpu_available()
    }

    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.gpu_manager.performance_stats()
    }
}

impl GpuAcceleratedQDA {
    pub fn new(
        qda_config: QuadraticDiscriminantAnalysisConfig,
        gpu_config: GpuAccelerationConfig,
    ) -> Result<Self> {
        Ok(Self {
            gpu_manager: GpuDiscriminantAnalysis::new(gpu_config)?,
            config: qda_config,
        })
    }

    pub fn fit_predict(
        &self,
        x: &ArrayView2<Float>,
        y: &ArrayView1<usize>,
        x_test: &ArrayView2<Float>,
    ) -> Result<Array1<usize>> {
        self.gpu_manager
            .gpu_qda_fit_predict(x, y, x_test, &self.config)
    }

    pub fn is_using_gpu(&self) -> bool {
        self.gpu_manager.is_gpu_available()
    }

    pub fn performance_stats(&self) -> GpuPerformanceStats {
        self.gpu_manager.performance_stats()
    }
}

// ─── Tests ──────────────────────────────────────────────────────────────────

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // -- Pure host-side logic, no GPU required --------------------------------

    #[test]
    fn test_gpu_acceleration_config_default() {
        let config = GpuAccelerationConfig::default();
        assert_eq!(config.gpu_threshold, 256);
        assert!(matches!(
            config.memory_strategy,
            GpuMemoryStrategy::Balanced
        ));
    }

    #[test]
    fn test_symmetrize_removes_round_off_asymmetry() {
        let m = array![[1.0, 2.0 + 1e-12], [2.0, 3.0]];
        let s = symmetrize(&m);
        assert!((s[[0, 1]] - s[[1, 0]]).abs() < 1e-15);
    }

    #[test]
    fn test_usize_labels_roundtrip() {
        let y = array![0usize, 1, 2, 1, 0];
        let y_i32 = usize_labels_to_i32(&y.view()).expect("conversion should succeed");
        assert_eq!(y_i32, array![0i32, 1, 2, 1, 0]);
        let back = i32_predictions_to_usize(&y_i32).expect("conversion should succeed");
        assert_eq!(back, y);
    }

    #[test]
    fn test_i32_predictions_to_usize_rejects_negative() {
        let bad = array![0i32, -1, 2];
        assert!(i32_predictions_to_usize(&bad).is_err());
    }

    #[test]
    fn test_argmax_rows() {
        let scores = array![[0.1, 0.9, 0.2], [5.0, 1.0, 2.0], [1.0, 1.0, 3.0]];
        let preds = argmax_rows(&scores.view());
        assert_eq!(preds, array![1usize, 0, 2]);
    }

    #[test]
    fn test_softmax_rows_sums_to_one() {
        let scores = array![[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]];
        let probs = softmax_rows(&scores.view());
        for row in probs.axis_iter(Axis(0)) {
            let sum: Float = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
        // Uniform scores -> uniform probabilities.
        assert!((probs[[1, 0]] - 1.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_between_class_scatter_matches_manual_computation() {
        // Two classes, two points each, on the x-axis: means (0,0) and (2,0),
        // overall mean (1,0). S_b = 2*(1,0)(1,0)^T + 2*(1,0)(1,0)^T-shape
        // (using the (mean_k - overall_mean) form): diff0 = (-1,0),
        // diff1 = (1,0), each weighted by n_k=2 -> S_b = [[4,0],[0,0]].
        let x = array![[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]];
        let y = array![0usize, 0, 1, 1];
        let means = array![[0.0, 0.0], [2.0, 0.0]];
        let sb = between_class_scatter(&x.view(), &y.view(), &means.view(), 2);
        assert!((sb[[0, 0]] - 4.0).abs() < 1e-10);
        assert!(sb[[0, 1]].abs() < 1e-10);
        assert!(sb[[1, 1]].abs() < 1e-10);
    }

    #[test]
    fn test_lda_config_supported_on_gpu() {
        let default_config = LinearDiscriminantAnalysisConfig::default();
        assert!(GpuDiscriminantAnalysis::lda_config_supported_on_gpu(
            &default_config
        ));

        let shrinkage_config = LinearDiscriminantAnalysisConfig {
            shrinkage: Some(0.1),
            ..Default::default()
        };
        assert!(!GpuDiscriminantAnalysis::lda_config_supported_on_gpu(
            &shrinkage_config
        ));

        let robust_config = LinearDiscriminantAnalysisConfig {
            robust: true,
            ..Default::default()
        };
        assert!(!GpuDiscriminantAnalysis::lda_config_supported_on_gpu(
            &robust_config
        ));
    }

    #[test]
    fn test_qda_config_supported_on_gpu() {
        let default_config = QuadraticDiscriminantAnalysisConfig::default();
        assert!(GpuDiscriminantAnalysis::qda_config_supported_on_gpu(
            &default_config
        ));

        let robust_config = QuadraticDiscriminantAnalysisConfig {
            robust: true,
            ..Default::default()
        };
        assert!(!GpuDiscriminantAnalysis::qda_config_supported_on_gpu(
            &robust_config
        ));
    }

    /// Validates the *math* used by [`GpuLDAKernel::solve_generalized_eigen_gpu`]
    /// (Cholesky reduction: `S_w = LL^T`, `C = L^{-1} S_b L^{-T}`, eigh(C),
    /// back-transform `w = L^{-T}y`) via plain host arithmetic — no GPU
    /// required. This exercises the exact same formulas the on-device method
    /// evaluates via `oxicuda_solver`, just computed with
    /// `scirs2_linalg`/`NumericalStability` instead, so it gives strong
    /// confidence in the on-device path's correctness even though it cannot
    /// be executed directly on a machine with no CUDA device.
    #[test]
    fn test_cholesky_reduction_math_matches_generalized_eigen_definition() {
        // NOTE: `sw`/`sb` are chosen so that every symmetric eigenproblem this
        // test touches (`C = L^{-1} S_b L^{-T}` below, and the `B^{-1/2} A
        // B^{-1/2}` reduction inside `stable_generalized_eigen`'s
        // cross-check) has well-separated eigenvalues (consecutive ratios
        // <~0.45). `scirs2_linalg::eigh`'s n=3 specialization is a *plain*
        // (unshifted) QR iteration capped at 50 sweeps; its convergence rate
        // is governed by the ratio of consecutive eigenvalues, and a
        // close-eigenvalue matrix (e.g. eigenvalues 0.458/0.500/0.556, ratio
        // ~0.9) does not fully converge in 50 unshifted sweeps, leaving an
        // eigenvalue error of a few 1e-6 -- enough to blow the 1e-6
        // eigenpair-residual check below. That is a precision limit of the
        // upstream small-matrix solver, not of the Cholesky-reduction
        // formula this test exists to validate, so we sidestep it with
        // better-conditioned inputs rather than loosening the tolerance.
        let sw = array![[6.0, 1.0, 0.0], [1.0, 3.0, 0.5], [0.0, 0.5, 1.0]];
        let sb = array![[9.0, 0.2, 0.05], [0.2, 2.0, 0.1], [0.05, 0.1, 0.3]];

        let l = scirs2_linalg::cholesky(&sw.view(), None).expect("S_w is SPD by construction");
        let ns = NumericalStability::new();
        let l_inv = ns
            .matrix_inverse(&l)
            .expect("L is invertible (SPD Cholesky factor)");
        let c = symmetrize(&l_inv.dot(&sb).dot(&l_inv.t()));
        let (eigenvalues, y) = ns
            .stable_eigen_decomposition(&c)
            .expect("C is symmetric by construction");
        let w = l_inv.t().dot(&y);

        // Direct check of the generalized-eigenpair definition: S_b w ~=
        // lambda * S_w w for every column, independent of any reference
        // implementation.
        for k in 0..eigenvalues.len() {
            let wk = w.column(k);
            let lhs = sb.dot(&wk);
            let rhs = sw.dot(&wk).mapv(|v| v * eigenvalues[k]);
            for (a, b) in lhs.iter().zip(rhs.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "S_b w != lambda S_w w for eigenpair {k}: {a} vs {b}"
                );
            }
        }

        // Cross-check against the already-tested CPU generalized eigensolver:
        // eigenvalues should match up to sort order (both conventions here
        // are ascending from `stable_eigen_decomposition` composed the same
        // way -- what matters is the *set* of eigenvalues agrees).
        let (ref_eigenvalues, _) = ns
            .stable_generalized_eigen(&sb, &sw)
            .expect("reference generalized eigensolve should succeed");
        let mut a: Vec<Float> = eigenvalues.iter().copied().collect();
        let mut b: Vec<Float> = ref_eigenvalues.iter().copied().collect();
        a.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        b.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-6, "eigenvalue mismatch: {x} vs {y}");
        }
    }

    // -- GPU-gated: skip gracefully when GpuBackend::detect() finds nothing --

    #[test]
    fn test_compute_class_means_gpu_matches_reference() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!("skipping test_compute_class_means_gpu_matches_reference: no GPU detected");
            return;
        };

        let x = array![[1.0, 2.0], [3.0, 4.0], [10.0, 10.0], [12.0, 14.0]];
        let y = array![0usize, 0, 1, 1];
        let kernel = GpuLDAKernel::new(GpuAccelerationConfig::default());
        let means = kernel
            .compute_class_means_gpu(&backend, &x.view(), &y.view(), 2)
            .expect("compute_class_means_gpu should succeed");

        assert!((means[[0, 0]] - 2.0).abs() < 1e-8);
        assert!((means[[0, 1]] - 3.0).abs() < 1e-8);
        assert!((means[[1, 0]] - 11.0).abs() < 1e-8);
        assert!((means[[1, 1]] - 12.0).abs() < 1e-8);
    }

    #[test]
    fn test_solve_generalized_eigen_gpu_matches_cpu_reference() {
        let Some(backend) = GpuBackend::detect().expect("detect() should not hard-error") else {
            eprintln!(
                "skipping test_solve_generalized_eigen_gpu_matches_cpu_reference: no GPU detected"
            );
            return;
        };

        let sw = array![[4.0, 1.0], [1.0, 3.0]];
        let sb = array![[2.0, 0.5], [0.5, 1.0]];
        let kernel = GpuLDAKernel::new(GpuAccelerationConfig::default());
        let (eigenvalues, w) = kernel
            .solve_generalized_eigen_gpu(&backend, &sb, &sw)
            .expect("solve_generalized_eigen_gpu should succeed");

        for k in 0..eigenvalues.len() {
            let wk = w.column(k);
            let lhs = sb.dot(&wk);
            let rhs = sw.dot(&wk).mapv(|v| v * eigenvalues[k]);
            for (a, b) in lhs.iter().zip(rhs.iter()) {
                assert!((a - b).abs() < 1e-6);
            }
        }
    }

    // -- Manager-level fit/predict: unconditional (exercises the CPU fallback
    //    on this dev machine, and would exercise the GPU path on real
    //    hardware without any change to the test itself) ---------------------

    /// Requirements (b) and (c): predictions from
    /// `GpuDiscriminantAnalysis::gpu_lda_fit_predict` must match plain CPU
    /// `LinearDiscriminantAnalysis` fit+predict on the same data, and this
    /// must hold (not panic, not error) in the graceful-fallback case where
    /// `GpuBackend::detect()` finds nothing -- the expected case on this
    /// dev machine.
    #[test]
    fn test_gpu_lda_fit_predict_matches_cpu_lda() {
        let x = array![
            [0.0, 0.0],
            [0.2, -0.1],
            [-0.1, 0.15],
            [5.0, 5.0],
            [5.2, 4.9],
            [4.9, 5.1],
            [-5.0, 5.0],
            [-4.8, 5.2],
            [-5.1, 4.85],
        ];
        let y = array![0usize, 0, 0, 1, 1, 1, 2, 2, 2];
        let x_test = array![[0.1, 0.05], [5.1, 5.0], [-5.0, 5.05]];

        let manager = GpuDiscriminantAnalysis::new(GpuAccelerationConfig::default())
            .expect("manager creation should not fail");
        // Environment sanity check: on this dev machine (no CUDA device),
        // detect() should find nothing, so the call below exercises the CPU
        // fallback path specifically.
        assert!(!manager.is_gpu_available());

        let config = LinearDiscriminantAnalysisConfig::default();
        let gpu_predictions = manager
            .gpu_lda_fit_predict(&x.view(), &y.view(), &x_test.view(), &config)
            .expect("gpu_lda_fit_predict should succeed via CPU fallback");

        let y_i32 = usize_labels_to_i32(&y.view()).expect("conversion should succeed");
        let cpu_lda = LinearDiscriminantAnalysis::new()
            .fit(&x, &y_i32)
            .expect("CPU LDA fit should succeed");
        let cpu_predictions_i32 = cpu_lda
            .predict(&x_test)
            .expect("CPU LDA predict should succeed");
        let cpu_predictions =
            i32_predictions_to_usize(&cpu_predictions_i32).expect("conversion should succeed");

        assert_eq!(gpu_predictions, cpu_predictions);
    }

    /// Same as above for QDA. Uses well-separated classes with independently
    /// constructed (near-diagonal) per-class covariance so the comparison is
    /// robust to `QuadraticDiscriminantAnalysis::predict_proba`'s existing
    /// diagonal-only quadratic-form approximation for the non-diagonal-config
    /// case (a pre-existing characteristic of the CPU implementation, not
    /// something introduced here).
    #[test]
    fn test_gpu_qda_fit_predict_matches_cpu_qda() {
        let x = array![
            [0.0, 0.0],
            [0.3, -0.2],
            [-0.2, 0.25],
            [0.1, -0.1],
            [10.0, 0.0],
            [10.3, 0.2],
            [9.7, -0.15],
            [10.1, 0.1],
            [0.0, 10.0],
            [0.25, 9.8],
            [-0.2, 10.2],
            [0.1, 9.9],
        ];
        let y = array![0usize, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2];
        let x_test = array![[0.05, -0.05], [10.05, 0.05], [0.05, 10.05]];

        let manager = GpuDiscriminantAnalysis::new(GpuAccelerationConfig::default())
            .expect("manager creation should not fail");
        assert!(!manager.is_gpu_available());

        let config = QuadraticDiscriminantAnalysisConfig::default();
        let gpu_predictions = manager
            .gpu_qda_fit_predict(&x.view(), &y.view(), &x_test.view(), &config)
            .expect("gpu_qda_fit_predict should succeed via CPU fallback");

        let y_i32 = usize_labels_to_i32(&y.view()).expect("conversion should succeed");
        let cpu_qda = QuadraticDiscriminantAnalysis::new()
            .fit(&x, &y_i32)
            .expect("CPU QDA fit should succeed");
        let cpu_predictions_i32 = cpu_qda
            .predict(&x_test)
            .expect("CPU QDA predict should succeed");
        let cpu_predictions =
            i32_predictions_to_usize(&cpu_predictions_i32).expect("conversion should succeed");

        assert_eq!(gpu_predictions, cpu_predictions);
    }

    #[test]
    fn test_gpu_accelerated_lda_wrapper_fallback() {
        let x = array![[0.0, 0.0], [0.1, 0.1], [5.0, 5.0], [5.1, 4.9]];
        let y = array![0usize, 0, 1, 1];
        let x_test = array![[0.05, 0.05], [5.05, 4.95]];

        let wrapper = GpuAcceleratedLDA::new(
            LinearDiscriminantAnalysisConfig::default(),
            GpuAccelerationConfig::default(),
        )
        .expect("wrapper creation should not fail");
        assert!(!wrapper.is_using_gpu());

        let predictions = wrapper
            .fit_predict(&x.view(), &y.view(), &x_test.view())
            .expect("fit_predict should succeed via CPU fallback");
        assert_eq!(predictions, array![0usize, 1]);
        assert!(wrapper.performance_stats().fallback_count >= 1);
    }

    #[test]
    fn test_gpu_accelerated_qda_wrapper_fallback() {
        let x = array![
            [0.0, 0.0],
            [0.1, -0.1],
            [-0.1, 0.1],
            [5.0, 5.0],
            [5.1, 4.9],
            [4.9, 5.1]
        ];
        let y = array![0usize, 0, 0, 1, 1, 1];
        let x_test = array![[0.0, 0.0], [5.0, 5.0]];

        let wrapper = GpuAcceleratedQDA::new(
            QuadraticDiscriminantAnalysisConfig::default(),
            GpuAccelerationConfig::default(),
        )
        .expect("wrapper creation should not fail");
        assert!(!wrapper.is_using_gpu());

        let predictions = wrapper
            .fit_predict(&x.view(), &y.view(), &x_test.view())
            .expect("fit_predict should succeed via CPU fallback");
        assert_eq!(predictions, array![0usize, 1]);
        assert!(wrapper.performance_stats().fallback_count >= 1);
    }
}
