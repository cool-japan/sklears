//! GPU acceleration for kernel approximations, backed by `sklears_core::gpu`
//! (a real CUDA context + BLAS handle, via the `oxicuda` crate family).
//!
//! # Honesty pass (2026-07-06)
//!
//! Before this pass, this module shipped in every default build (there was
//! no `gpu` feature gate at all) and every "GPU" code path was fake:
//!
//! - The local `GpuBackend { Cuda, OpenCL, Metal, Cpu }` enum had no real
//!   backend behind `Cuda`/`OpenCL`/`Metal` — `initialize_cuda`/`_opencl`/
//!   `_metal` always returned `SklearsError::NotImplemented`, so the only
//!   backend that could ever actually initialize was `Cpu`.
//! - `generate_features_cuda`/`_opencl`/`_metal` and
//!   `transform_cuda`/`_opencl`/`_metal` were byte-for-byte copies of the
//!   CPU loop, just renamed.
//! - `compute_kernel_cuda`/`_opencl`/`_metal` and
//!   `eigendecomposition_cuda`/`_opencl`/`_metal` called straight through to
//!   the CPU implementation.
//! - Worst of all, `eigendecomposition_cpu` itself was numerically wrong: it
//!   ran power iteration for the leading eigenpair only, then fabricated
//!   every other eigenvalue as a flat `0.1` and every other eigenvector as a
//!   standard basis vector — silently distorting Nyström scaling for every
//!   component past the first, on every build, GPU feature or not.
//!
//! This version:
//!
//! - Gates this whole module behind this crate's own `gpu` feature (see
//!   `Cargo.toml`), off by default, so no GPU-named API ships in default
//!   Pure-Rust builds.
//! - Replaces the fake `GpuBackend` enum with a re-export of the real
//!   [`sklears_core::gpu::GpuBackend`]: [`GpuContext::initialize`] calls
//!   [`GpuBackend::detect`] honestly and stores `None` when no device is
//!   found (this crate's own macOS development machine included) instead of
//!   pretending a CUDA/OpenCL/Metal kernel ran. There is no `OpenCL`/`Metal`
//!   variant to fall back to any more — only "a real CUDA device was found"
//!   or "run on the CPU".
//! - Replaces `eigendecomposition_cpu`'s power-iteration-plus-fabrication
//!   with a real symmetric eigendecomposition
//!   (`scirs2_linalg::compat::eigh`), fixing the correctness bug above. As
//!   of `oxicuda-solver` 0.4.0 the on-device symmetric eigensolver is a
//!   documented exact-CPU host fallback (no true on-device eigendecomposition
//!   exists to route to yet), so this crate does not claim one: eigendecomposition
//!   always runs on the CPU, GPU feature or not.
//! - [`FittedGpuRBFSampler::transform`] and [`GpuNystroem`]'s kernel-matrix
//!   computation now do real on-device GEMM
//!   ([`sklears_core::gpu::GpuArray::matmul`]) for their `O(n·m·d)`
//!   inner-product term whenever a GPU is detected, instead of three
//!   duplicate CPU loops named after CUDA/OpenCL/Metal. The elementwise
//!   transform on top (the cosine random features, or the RBF exponential)
//!   still runs on the host after a single download: neither op has an
//!   on-device primitive wired up in this crate yet, and both are `O(n·m)`
//!   — cheap relative to the `O(n·m·d)` GEMM they follow.

use scirs2_core::ndarray::{Array1, Array2};
use scirs2_core::random::essentials::{Normal as RandNormal, Uniform as RandUniform};
use scirs2_core::random::rngs::StdRng as RealStdRng;
use scirs2_core::random::seq::SliceRandom;
use scirs2_core::random::RngExt;
use scirs2_core::random::{thread_rng, SeedableRng};
use scirs2_linalg::compat::{eigh, UPLO};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use sklears_core::error::{Result, SklearsError};
pub use sklears_core::gpu::GpuBackend;
use sklears_core::gpu::{GpuArray, GpuMatrixOps, GpuUtils};
use sklears_core::traits::{Fit, Transform};

fn numerical_err<E: std::fmt::Display>(e: E) -> SklearsError {
    SklearsError::NumericalError(e.to_string())
}

/// GPU memory management strategy.
///
/// Informational only for now: `sklears_core::gpu`'s `GpuArray` always
/// allocates and transfers eagerly, so this does not yet change how memory
/// is actually moved. It is kept on [`GpuConfig`] as a forward-compatible
/// knob rather than removed outright.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryStrategy {
    /// Page-locked (pinned) host memory for faster transfers.
    Pinned,
    /// Unified memory management.
    Managed,
    /// Manual, explicit memory management.
    Explicit,
}

/// GPU computation precision.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Precision {
    /// f32
    Single,
    /// f64
    Double,
    /// f16 (if supported)
    Half,
}

/// GPU device information.
///
/// Populated from [`GpuUtils::device_properties`] (name, total memory,
/// compute capability) plus a direct `oxicuda_driver::Device::info()` query
/// for the two fields (`multiprocessor_count`, `max_threads_per_block`)
/// `sklears_core::gpu`'s public API does not surface.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuDevice {
    /// Device ordinal.
    pub id: usize,
    /// Human-readable device name.
    pub name: String,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (i32, i32),
    /// Total device memory in bytes.
    pub total_memory: usize,
    /// Number of streaming multiprocessors.
    pub multiprocessor_count: usize,
    /// Maximum threads per block.
    pub max_threads_per_block: usize,
}

impl GpuDevice {
    /// Queries real device properties for `device_id`. Returns `None` if
    /// the driver cannot be initialized, the device does not exist, or any
    /// property query fails — this is a best-effort lookup used only to
    /// populate informational fields, never load-bearing for correctness.
    fn query(device_id: usize) -> Option<Self> {
        let props = GpuUtils::device_properties(device_id).ok()?;
        oxicuda_driver::init().ok()?;
        let ordinal = i32::try_from(device_id).ok()?;
        let device = oxicuda_driver::Device::get(ordinal).ok()?;
        let info = device.info().ok()?;
        Some(Self {
            id: device_id,
            name: props.name,
            compute_capability: props.compute_capability,
            total_memory: props.total_memory,
            multiprocessor_count: info.multiprocessor_count.max(0) as usize,
            max_threads_per_block: info.max_threads_per_block.max(0) as usize,
        })
    }
}

/// GPU configuration for kernel approximations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuConfig {
    /// Which device ordinal to detect/bind to.
    pub device_id: usize,
    /// memory_strategy
    pub memory_strategy: MemoryStrategy,
    /// precision
    pub precision: Precision,
    /// block_size
    pub block_size: usize,
    /// grid_size
    pub grid_size: usize,
    /// enable_async
    pub enable_async: bool,
    /// stream_count
    pub stream_count: usize,
}

impl Default for GpuConfig {
    fn default() -> Self {
        Self {
            device_id: 0,
            memory_strategy: MemoryStrategy::Managed,
            precision: Precision::Double,
            block_size: 256,
            grid_size: 0, // Auto-calculate
            enable_async: true,
            stream_count: 4,
        }
    }
}

/// GPU context manager: an honest wrapper around the real
/// [`sklears_core::gpu::GpuBackend`].
///
/// `backend` is `Some` iff [`GpuBackend::detect`] found a real device;
/// otherwise every operation built on top of this context runs on the CPU.
/// Unlike the pre-migration version, construction never needs to "fail" for
/// an unimplemented backend — there is only "a GPU was found" or "run on the
/// CPU", and both are legitimate, expected outcomes.
#[derive(Clone)]
pub struct GpuContext {
    /// config
    pub config: GpuConfig,
    /// device
    pub device: Option<GpuDevice>,
    backend: Option<GpuBackend>,
    /// is_initialized
    pub is_initialized: bool,
}

impl std::fmt::Debug for GpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GpuContext")
            .field("config", &self.config)
            .field("device", &self.device)
            .field("is_gpu", &self.backend.is_some())
            .field("is_initialized", &self.is_initialized)
            .finish()
    }
}

impl GpuContext {
    /// Builds a context that is not yet initialized; call
    /// [`initialize`](Self::initialize) before using it.
    pub fn new(config: GpuConfig) -> Self {
        Self {
            config,
            device: None,
            backend: None,
            is_initialized: false,
        }
    }

    /// Detects a real GPU via [`GpuBackend::detect`]. Always returns `Ok`:
    /// "no GPU found" is a legitimate, expected outcome (e.g. on this
    /// crate's own macOS development machine), not an error.
    pub fn initialize(&mut self) -> Result<()> {
        self.backend = GpuBackend::detect()?;
        self.device = self
            .backend
            .as_ref()
            .and_then(|_| GpuDevice::query(self.config.device_id));
        self.is_initialized = true;
        Ok(())
    }

    /// `true` iff a real GPU was detected during [`initialize`](Self::initialize).
    pub fn is_gpu(&self) -> bool {
        self.backend.is_some()
    }

    /// The live GPU backend handle, if one was detected.
    pub fn backend(&self) -> Option<&GpuBackend> {
        self.backend.as_ref()
    }

    /// Suggested CUDA block size for a problem of the given size, derived
    /// from real device properties when available.
    pub fn get_optimal_block_size(&self, problem_size: usize) -> usize {
        if let Some(device) = &self.device {
            let max_threads = device.max_threads_per_block.max(1);
            let suggested_size = (problem_size as f64).sqrt() as usize;
            suggested_size.min(max_threads).max(32)
        } else {
            256
        }
    }

    /// Grid size needed to cover `problem_size` with the given block size.
    pub fn get_optimal_grid_size(&self, problem_size: usize, block_size: usize) -> usize {
        problem_size.div_ceil(block_size.max(1))
    }
}

/// GPU-accelerated RBF sampler (random Fourier features).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuRBFSampler {
    /// n_components
    pub n_components: usize,
    /// gamma
    pub gamma: f64,
    /// gpu_config
    pub gpu_config: GpuConfig,
}

impl GpuRBFSampler {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            gamma: 1.0,
            gpu_config: GpuConfig::default(),
        }
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn gpu_config(mut self, config: GpuConfig) -> Self {
        self.gpu_config = config;
        self
    }

    /// Single host-side random-feature generation, used regardless of
    /// backend. There is no on-device RNG wired up here (`cuRAND` would be a
    /// separate integration); the historical `generate_features_cuda`/
    /// `_opencl`/`_metal` variants were byte-for-byte copies of this same
    /// loop and have been removed rather than kept as decoration.
    fn generate_random_features(&self, input_dim: usize) -> Result<Array2<f64>> {
        let mut rng = RealStdRng::from_seed(thread_rng().random());
        let normal = RandNormal::new(0.0, (2.0 * self.gamma).sqrt()).map_err(numerical_err)?;

        let mut weights = Array2::zeros((self.n_components, input_dim));
        for i in 0..self.n_components {
            for j in 0..input_dim {
                weights[[i, j]] = rng.sample(normal);
            }
        }

        Ok(weights)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedGpuRBFSampler {
    /// n_components
    pub n_components: usize,
    /// gamma
    pub gamma: f64,
    /// gpu_config
    pub gpu_config: GpuConfig,
    /// weights
    pub weights: Array2<f64>,
    /// biases
    pub biases: Array1<f64>,
    #[serde(skip)]
    pub gpu_context: Option<Arc<GpuContext>>,
}

impl Fit<Array2<f64>, ()> for GpuRBFSampler {
    type Fitted = FittedGpuRBFSampler;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let input_dim = x.ncols();

        // Detect (honestly) whether a real GPU is available.
        let mut gpu_context = GpuContext::new(self.gpu_config.clone());
        gpu_context.initialize()?;

        // Generate random features (always host-side; see doc comment).
        let weights = self.generate_random_features(input_dim)?;

        // Generate random biases.
        let mut rng = RealStdRng::from_seed(thread_rng().random());
        let uniform = RandUniform::new(0.0, 2.0 * std::f64::consts::PI).map_err(numerical_err)?;
        let biases = Array1::from_vec(
            (0..self.n_components)
                .map(|_| rng.sample(uniform))
                .collect(),
        );

        Ok(FittedGpuRBFSampler {
            n_components: self.n_components,
            gamma: self.gamma,
            gpu_config: self.gpu_config.clone(),
            weights,
            biases,
            gpu_context: Some(Arc::new(gpu_context)),
        })
    }
}

impl FittedGpuRBFSampler {
    /// Real on-device GEMM (`X · Wᵀ`) via [`GpuArray::matmul`], with the
    /// bias add and cosine transform applied on the host after a single
    /// download.
    fn transform_gpu(&self, x: &Array2<f64>, backend: &GpuBackend) -> Result<Array2<f64>> {
        let x_gpu = GpuArray::from_array2(backend, x)?;
        let w_t = self.weights.t().to_owned();
        let w_gpu = GpuArray::from_array2(backend, &w_t)?;
        let projected = x_gpu.matmul(&w_gpu)?;
        let projected = projected.to_array2()?;
        self.apply_cosine_features(&projected)
    }

    fn transform_cpu(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let projected = x.dot(&self.weights.t());
        self.apply_cosine_features(&projected)
    }

    fn apply_cosine_features(&self, projected: &Array2<f64>) -> Result<Array2<f64>> {
        let scale = (2.0 / self.n_components as f64).sqrt();
        let mut result = Array2::zeros(projected.dim());
        for i in 0..projected.nrows() {
            for j in 0..projected.ncols() {
                result[[i, j]] = scale * (projected[[i, j]] + self.biases[j]).cos();
            }
        }
        Ok(result)
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedGpuRBFSampler {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        match self.gpu_context.as_ref().and_then(|ctx| ctx.backend()) {
            Some(backend) => self.transform_gpu(x, backend),
            None => self.transform_cpu(x),
        }
    }
}

/// Computes the kernel matrix `K[i,j] = k(x_i, y_j)` for `"linear"` or
/// `"rbf"` kernels, shared by [`GpuNystroem::fit`] and
/// [`FittedGpuNystroem::transform`].
///
/// When `backend` is `Some`, the `O(n·m·d)` inner-product term (`X·Yᵀ`) runs
/// as a real on-device GEMM via [`GpuArray::matmul`]; the norm expansion and
/// RBF exponential (both `O(n·m)`, cheap relative to the GEMM they follow)
/// still run on the host after a single download.
fn compute_kernel_matrix(
    kernel: &str,
    gamma: f64,
    x: &Array2<f64>,
    y: &Array2<f64>,
    backend: Option<&GpuBackend>,
) -> Result<Array2<f64>> {
    let inner = |x: &Array2<f64>, y: &Array2<f64>| -> Result<Array2<f64>> {
        if let Some(backend) = backend {
            let x_gpu = GpuArray::from_array2(backend, x)?;
            let y_t = y.t().to_owned();
            let y_gpu = GpuArray::from_array2(backend, &y_t)?;
            x_gpu.matmul(&y_gpu)?.to_array2()
        } else {
            Ok(x.dot(&y.t()))
        }
    };

    match kernel {
        "linear" => inner(x, y),
        "rbf" => {
            let inner_products = inner(x, y)?;
            let x_norms: Vec<f64> = x.rows().into_iter().map(|r| r.dot(&r)).collect();
            let y_norms: Vec<f64> = y.rows().into_iter().map(|r| r.dot(&r)).collect();

            let mut kernel_matrix = Array2::zeros((x.nrows(), y.nrows()));
            for i in 0..x.nrows() {
                for j in 0..y.nrows() {
                    // |x_i - y_j|^2 = |x_i|^2 + |y_j|^2 - 2 x_i . y_j
                    let squared_norm = x_norms[i] + y_norms[j] - 2.0 * inner_products[[i, j]];
                    kernel_matrix[[i, j]] = (-gamma * squared_norm).exp();
                }
            }
            Ok(kernel_matrix)
        }
        other => Err(SklearsError::InvalidInput(format!(
            "Unsupported kernel: {other}"
        ))),
    }
}

/// GPU-accelerated Nyström approximation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuNystroem {
    /// n_components
    pub n_components: usize,
    /// kernel
    pub kernel: String,
    /// gamma
    pub gamma: f64,
    /// gpu_config
    pub gpu_config: GpuConfig,
}

impl GpuNystroem {
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            kernel: "rbf".to_string(),
            gamma: 1.0,
            gpu_config: GpuConfig::default(),
        }
    }

    pub fn kernel(mut self, kernel: String) -> Self {
        self.kernel = kernel;
        self
    }

    pub fn gamma(mut self, gamma: f64) -> Self {
        self.gamma = gamma;
        self
    }

    pub fn gpu_config(mut self, config: GpuConfig) -> Self {
        self.gpu_config = config;
        self
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FittedGpuNystroem {
    /// n_components
    pub n_components: usize,
    /// kernel
    pub kernel: String,
    /// gamma
    pub gamma: f64,
    /// gpu_config
    pub gpu_config: GpuConfig,
    /// basis_vectors
    pub basis_vectors: Array2<f64>,
    /// eigenvalues
    pub eigenvalues: Array1<f64>,
    /// eigenvectors
    pub eigenvectors: Array2<f64>,
    #[serde(skip)]
    pub gpu_context: Option<Arc<GpuContext>>,
}

impl Fit<Array2<f64>, ()> for GpuNystroem {
    type Fitted = FittedGpuNystroem;

    fn fit(self, x: &Array2<f64>, _y: &()) -> Result<Self::Fitted> {
        let mut gpu_context = GpuContext::new(self.gpu_config.clone());
        gpu_context.initialize()?;

        let n_samples = x.nrows();
        let n_components = self.n_components.min(n_samples);

        // Random sampling of basis vectors.
        let mut rng = RealStdRng::from_seed(thread_rng().random());
        let mut indices: Vec<usize> = (0..n_samples).collect();
        indices.shuffle(&mut rng);
        indices.truncate(n_components);

        let mut basis_vectors = Array2::zeros((n_components, x.ncols()));
        for (i, &idx) in indices.iter().enumerate() {
            basis_vectors.row_mut(i).assign(&x.row(idx));
        }

        // Compute kernel matrix (real on-device GEMM for the inner-product
        // term when a GPU is present; see `compute_kernel_matrix`).
        let kernel_matrix = compute_kernel_matrix(
            &self.kernel,
            self.gamma,
            &basis_vectors,
            &basis_vectors,
            gpu_context.backend(),
        )?;

        // Real symmetric eigendecomposition — see the module docs for why
        // this always runs on the CPU, GPU feature or not, and for what it
        // replaces (a power-iteration-plus-fabrication placeholder that
        // invented every eigenvalue past the first as a flat `0.1`).
        let (eigenvalues, eigenvectors) = eigh(&kernel_matrix, UPLO::Lower)
            .map_err(|e| SklearsError::NumericalError(format!("eigendecomposition failed: {e}")))?;

        Ok(FittedGpuNystroem {
            n_components,
            kernel: self.kernel.clone(),
            gamma: self.gamma,
            gpu_config: self.gpu_config.clone(),
            basis_vectors,
            eigenvalues,
            eigenvectors,
            gpu_context: Some(Arc::new(gpu_context)),
        })
    }
}

impl Transform<Array2<f64>, Array2<f64>> for FittedGpuNystroem {
    fn transform(&self, x: &Array2<f64>) -> Result<Array2<f64>> {
        let backend = self.gpu_context.as_ref().and_then(|ctx| ctx.backend());
        let kernel_matrix =
            compute_kernel_matrix(&self.kernel, self.gamma, x, &self.basis_vectors, backend)?;

        // Apply Nyström approximation: K(x, basis) @ V @ S^{-1/2}
        let mut result = Array2::zeros((x.nrows(), self.n_components));
        for i in 0..x.nrows() {
            for j in 0..self.n_components {
                let mut sum = 0.0;
                for k in 0..self.n_components {
                    let eigenval_sqrt_inv = if self.eigenvalues[k] > 1e-12 {
                        1.0 / self.eigenvalues[k].sqrt()
                    } else {
                        0.0
                    };
                    sum += kernel_matrix[[i, k]] * self.eigenvectors[[k, j]] * eigenval_sqrt_inv;
                }
                result[[i, j]] = sum;
            }
        }

        Ok(result)
    }
}

/// GPU performance profiler.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GpuProfiler {
    /// enable_profiling
    pub enable_profiling: bool,
    /// kernel_times
    pub kernel_times: Vec<(String, f64)>,
    /// memory_usage
    pub memory_usage: Vec<(String, usize)>,
    /// transfer_times
    pub transfer_times: Vec<(String, f64)>,
}

impl Default for GpuProfiler {
    fn default() -> Self {
        Self::new()
    }
}

impl GpuProfiler {
    pub fn new() -> Self {
        Self {
            enable_profiling: false,
            kernel_times: Vec::new(),
            memory_usage: Vec::new(),
            transfer_times: Vec::new(),
        }
    }

    pub fn enable(mut self) -> Self {
        self.enable_profiling = true;
        self
    }

    pub fn record_kernel_time(&mut self, kernel_name: &str, time_ms: f64) {
        if self.enable_profiling {
            self.kernel_times.push((kernel_name.to_string(), time_ms));
        }
    }

    pub fn record_memory_usage(&mut self, operation: &str, bytes: usize) {
        if self.enable_profiling {
            self.memory_usage.push((operation.to_string(), bytes));
        }
    }

    pub fn record_transfer_time(&mut self, operation: &str, time_ms: f64) {
        if self.enable_profiling {
            self.transfer_times.push((operation.to_string(), time_ms));
        }
    }

    pub fn get_summary(&self) -> String {
        let mut summary = String::new();

        summary.push_str("GPU Performance Summary:\n");
        summary.push_str(&format!(
            "Total kernel executions: {}\n",
            self.kernel_times.len()
        ));

        if !self.kernel_times.is_empty() {
            let total_kernel_time: f64 = self.kernel_times.iter().map(|(_, time)| time).sum();
            summary.push_str(&format!("Total kernel time: {:.2} ms\n", total_kernel_time));
        }

        if !self.memory_usage.is_empty() {
            let max_memory: usize = *self
                .memory_usage
                .iter()
                .map(|(_, bytes)| bytes)
                .max()
                .unwrap_or(&0);
            summary.push_str(&format!("Peak memory usage: {} bytes\n", max_memory));
        }

        if !self.transfer_times.is_empty() {
            let total_transfer_time: f64 = self.transfer_times.iter().map(|(_, time)| time).sum();
            summary.push_str(&format!(
                "Total transfer time: {:.2} ms\n",
                total_transfer_time
            ));
        }

        summary
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::essentials::Normal;

    use scirs2_core::ndarray::{Array, Array2};
    use scirs2_core::random::thread_rng;

    #[test]
    fn test_gpu_context_initialization() {
        let config = GpuConfig::default();
        let mut context = GpuContext::new(config);

        // `initialize()` must never hard-error just because there is no GPU
        // present (e.g. on this crate's own macOS dev machine) — `Ok` with
        // `is_gpu() == false` is the expected, honest outcome there.
        assert!(context.initialize().is_ok());
        assert!(context.is_initialized);
    }

    #[test]
    fn test_gpu_rbf_sampler() {
        let x: Array2<f64> = Array::from_shape_fn((50, 10), |_| {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("valid normal params"))
        });
        let sampler = GpuRBFSampler::new(100).gamma(0.5);

        let fitted = sampler.fit(&x, &()).expect("fit should succeed");
        let transformed = fitted.transform(&x).expect("transform should succeed");

        assert_eq!(transformed.shape(), &[50, 100]);
    }

    #[test]
    fn test_gpu_nystroem() {
        let x: Array2<f64> = Array::from_shape_fn((30, 5), |_| {
            let mut rng = thread_rng();
            rng.sample(Normal::new(0.0, 1.0).expect("valid normal params"))
        });
        let nystroem = GpuNystroem::new(20).gamma(1.0);

        let fitted = nystroem.fit(&x, &()).expect("fit should succeed");
        let transformed = fitted.transform(&x).expect("transform should succeed");

        assert_eq!(transformed.shape()[0], 30);
        assert!(transformed.shape()[1] <= 20);
    }

    /// Regression test for the eigendecomposition correctness bug this pass
    /// fixes: the old power-iteration placeholder fabricated every
    /// eigenvalue past the first as a flat `0.1`, regardless of the input
    /// matrix. A real eigensolver on an (almost) rank-1 kernel matrix must
    /// report the non-leading eigenvalues as close to zero, not `0.1`.
    #[test]
    fn test_nystroem_eigendecomposition_is_not_fabricated() {
        // A symmetric, strongly rank-1-dominant matrix: outer(v, v) scaled
        // up, plus a tiny symmetric perturbation so it is not exactly
        // singular.
        let v = Array1::from_vec(vec![1.0, 2.0, 3.0, 4.0]);
        let n = v.len();
        let mut matrix = Array2::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                matrix[[i, j]] = 10.0 * v[i] * v[j];
            }
        }
        for i in 0..n {
            matrix[[i, i]] += 1e-6 * (i as f64 + 1.0);
        }

        let (eigenvalues, _) =
            eigh(&matrix, UPLO::Lower).expect("eigendecomposition should succeed");

        // Sort ascending-by-magnitude is already what `eigh` gives us; the
        // largest-magnitude eigenvalue should dominate, and every other one
        // should be near zero — not the fabricated `0.1`.
        let max_abs = eigenvalues
            .iter()
            .cloned()
            .fold(0.0_f64, |a, b| a.max(b.abs()));
        let mut others_near_zero = true;
        for &val in eigenvalues.iter() {
            if val.abs() < max_abs - 1e-9 && val.abs() > 1e-3 {
                others_near_zero = false;
            }
        }
        assert!(
            others_near_zero,
            "non-leading eigenvalues should not be fabricated as ~0.1: {eigenvalues:?}"
        );
    }

    #[test]
    fn test_gpu_profiler() {
        let mut profiler = GpuProfiler::new().enable();

        profiler.record_kernel_time("test_kernel", 5.5);
        profiler.record_memory_usage("allocation", 1024);
        profiler.record_transfer_time("host_to_device", 2.1);

        let summary = profiler.get_summary();
        assert!(summary.contains("Total kernel executions: 1"));
        assert!(summary.contains("Total kernel time: 5.50"));
    }
}
