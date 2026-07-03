//! GPU-accelerated methods for manifold learning
//!
//! This module provides GPU-accelerated implementations of core manifold learning
//! operations using oxicuda-backend for cross-platform GPU compute support.
//!
//! # Wave B7
//!
//! Two operations here are genuine host-side algorithms with no GPU-device
//! dependency at all: [`GpuTSNE::fit_transform`] (via
//! `oxicuda_manifold::tsne_fit`) and the HNSW branch of
//! [`GpuAccelerator::knn_search`] (via `oxicuda_manifold::{hnsw_build,
//! hnsw_search}`). Neither checks [`GpuAccelerator::is_gpu_available`]
//! before running, because they are not GPU kernels being conditionally
//! dispatched -- they *are* the CPU implementation, unconditionally. Only
//! `pairwise_distances`/`matrix_operations` have a real GPU code path (an
//! `oxicuda-blas` GEMM via `sklears_core::gpu`) that needs a detected
//! device, with a CPU loop fallback when `GpuBackend::detect` finds none.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps};

#[cfg(feature = "gpu")]
use oxicuda_manifold::{
    hnsw_build, hnsw_search, tsne_fit, HnswConfig, HnswDistance, LcgRng, ManifoldError,
    TsneConfigBuilder,
};

/// Map an `oxicuda_manifold` error onto this crate's [`SklearsError`].
///
/// `oxicuda_manifold`'s algorithms (t-SNE, HNSW, ...) are host-side and
/// self-contained, so their errors are always about the caller's own
/// data/config (bad perplexity, `k` larger than the dataset, a malformed
/// shape, ...) rather than about hardware. This gives the common variants a
/// reasonably-typed home in [`SklearsError`] instead of collapsing
/// everything into a single opaque string.
#[cfg(feature = "gpu")]
fn manifold_err(e: ManifoldError) -> SklearsError {
    match e {
        ManifoldError::ShapeMismatch { expected, got } => SklearsError::ShapeMismatch {
            expected: format!("{expected:?}"),
            actual: format!("{got:?}"),
        },
        ManifoldError::DimensionMismatch { a, b } => SklearsError::DimensionMismatch {
            expected: a,
            actual: b,
        },
        ManifoldError::NotConverged { iter } => {
            SklearsError::ConvergenceError { iterations: iter }
        }
        ManifoldError::EmptyInput => {
            SklearsError::InvalidInput("oxicuda_manifold: empty input".to_string())
        }
        ManifoldError::InvalidParameter { name, reason } => {
            SklearsError::InvalidParameter { name, reason }
        }
        ManifoldError::KNeighborsTooLarge { k, n } => SklearsError::InvalidParameter {
            name: "k".to_string(),
            reason: format!("k={k} exceeds population n={n}"),
        },
        ManifoldError::IndexOutOfBounds { index, len } => SklearsError::InvalidInput(format!(
            "oxicuda_manifold: index {index} out of bounds for length {len}"
        )),
        ManifoldError::InvalidConfiguration(reason) => SklearsError::InvalidConfiguration(reason),
        other => SklearsError::NumericalError(other.to_string()),
    }
}

/// GPU-accelerated distance computation for manifold learning.
///
/// Provides hardware-accelerated distance computations usable by manifold learning
/// algorithms. When GPU is unavailable, falls back to CPU computation.
///
/// # Examples
///
/// ```
/// use sklears_manifold::gpu_acceleration::GpuAccelerator;
/// use scirs2_core::ndarray::array;
///
/// let data = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
///
/// let mut accelerator = GpuAccelerator::new().unwrap();
/// let distances = accelerator.pairwise_distances(&data.view(), "euclidean").unwrap();
/// ```
pub struct GpuAccelerator {
    #[cfg(feature = "gpu")]
    context: Option<GpuContext>,
}

impl GpuAccelerator {
    /// Construct a new accelerator, auto-detecting the best available GPU
    /// device (the one with the most free memory).
    ///
    /// `GpuContext::detect()` -- an alias for
    /// `sklears_core::gpu::GpuBackend::detect` -- returns `Ok(None)` rather
    /// than an `Err` when no GPU/driver is present, which is the expected,
    /// non-error outcome on most development/CI machines. In that case
    /// every operation on this accelerator transparently uses its CPU
    /// implementation; see [`is_gpu_available`](Self::is_gpu_available).
    pub fn new() -> SklResult<Self> {
        #[cfg(feature = "gpu")]
        {
            let context = GpuContext::detect()?;
            Ok(Self { context })
        }
        #[cfg(not(feature = "gpu"))]
        Ok(Self {})
    }

    pub fn is_gpu_available(&self) -> bool {
        #[cfg(feature = "gpu")]
        {
            self.context.is_some()
        }
        #[cfg(not(feature = "gpu"))]
        {
            false
        }
    }

    /// Compute pairwise distances between all points in the dataset.
    pub fn pairwise_distances(
        &mut self,
        data: &ArrayView2<Float>,
        metric: &str,
    ) -> SklResult<Array2<Float>> {
        if self.is_gpu_available() {
            self.gpu_pairwise_distances(data, metric)
        } else {
            self.cpu_pairwise_distances(data, metric)
        }
    }

    /// GPU pairwise distances — Euclidean via GEMM trick, others via CPU.
    fn gpu_pairwise_distances(
        &mut self,
        data: &ArrayView2<Float>,
        metric: &str,
    ) -> SklResult<Array2<Float>> {
        match metric {
            "euclidean" => self.gemm_pairwise_euclidean(data),
            _ => self.cpu_pairwise_distances(data, metric),
        }
    }

    /// Euclidean GEMM trick: D[i,j] = sqrt(||x_i||² + ||y_j||² − 2·x_i·x_j)
    ///
    /// Upload X once, compute G = X·Xᵀ on GPU, derive norms from diagonal.
    #[cfg(feature = "gpu")]
    fn gemm_pairwise_euclidean(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let Some(ctx) = self.context.as_ref() else {
            // Called directly (not only through `pairwise_distances`'s
            // `is_gpu_available()` gate) with no context detected: degrade
            // to the CPU path instead of assuming the invariant holds.
            return self.cpu_pairwise_distances(data, "euclidean");
        };
        let owned = data.to_owned();
        let x_gpu = GpuArray::<Float>::from_array2(ctx, &owned)?;

        // Xᵀ: shape (d, n)
        let (n, d) = data.dim();
        let mut xt_flat = vec![0.0 as Float; n * d];
        for r in 0..n {
            for c in 0..d {
                xt_flat[c * n + r] = data[[r, c]];
            }
        }
        let xt_array = Array2::from_shape_vec((d, n), xt_flat)
            .map_err(|e| SklearsError::InvalidInput(format!("Transpose shape error: {}", e)))?;
        let xt_gpu = GpuArray::<Float>::from_array2(ctx, &xt_array)?;

        // G = X · Xᵀ  [n × n]
        let gram_gpu = x_gpu.matmul(&xt_gpu)?;
        let gram = gram_gpu.to_array2()?;

        let mut distances = Array2::<Float>::zeros((n, n));
        for i in 0..n {
            for j in 0..n {
                let sq = (gram[[i, i]] + gram[[j, j]] - 2.0 * gram[[i, j]]).max(0.0);
                distances[[i, j]] = sq.sqrt();
            }
        }
        Ok(distances)
    }

    #[cfg(not(feature = "gpu"))]
    fn gemm_pairwise_euclidean(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        self.cpu_pairwise_distances(data, "euclidean")
    }

    pub(crate) fn cpu_pairwise_distances(
        &self,
        data: &ArrayView2<Float>,
        metric: &str,
    ) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        for i in 0..n_samples {
            for j in i..n_samples {
                let dist = match metric {
                    "euclidean" => self.euclidean_distance(&data.row(i), &data.row(j)),
                    "manhattan" => self.manhattan_distance(&data.row(i), &data.row(j)),
                    "cosine" => self.cosine_distance(&data.row(i), &data.row(j)),
                    _ => {
                        return Err(SklearsError::InvalidParameter {
                            name: "metric".to_string(),
                            reason: format!("Unknown metric: {}", metric),
                        })
                    }
                };
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    /// Compute k-nearest neighbors.
    ///
    /// For the `"euclidean"`/`"cosine"` metrics on non-degenerate input (at
    /// least 1 requested neighbour, at least 1 feature column), this builds
    /// an `oxicuda_manifold` HNSW graph index and queries it: an
    /// approximate (but, per `oxicuda_manifold`'s own recall tests,
    /// typically >= 0.8-0.9 recall) search that is asymptotically much
    /// cheaper than the O(n^2) sort below for large `n`. Like
    /// `GpuTSNE::fit_transform`, this is a host-side algorithm, so it does
    /// not consult `is_gpu_available()`. Any other metric (e.g.
    /// `"manhattan"`, for which `HnswDistance` has no variant), or a
    /// degenerate input shape, falls back to the exact
    /// pairwise-distance-and-sort path below -- which also remains
    /// available directly via `cpu_knn_search` as an always-exact
    /// comparison baseline (see `GpuBenchmark::benchmark_knn_search`).
    pub fn knn_search(
        &mut self,
        data: &ArrayView2<Float>,
        k: usize,
        metric: &str,
    ) -> SklResult<(Array2<Float>, Array2<usize>)> {
        let n_samples = data.nrows();
        let k_actual = k.min(n_samples.saturating_sub(1));

        #[cfg(feature = "gpu")]
        if k_actual > 0 && data.ncols() > 0 {
            if let Some(hnsw_metric) = Self::hnsw_distance_for_metric(metric) {
                return self.hnsw_knn_search(data, k_actual, hnsw_metric);
            }
        }

        let distances = self.pairwise_distances(data, metric)?;
        let mut knn_distances = Array2::zeros((n_samples, k_actual));
        let mut knn_indices = Array2::zeros((n_samples, k_actual));

        for i in 0..n_samples {
            let mut row: Vec<(Float, usize)> = (0..n_samples)
                .filter(|&j| j != i)
                .map(|j| (distances[[i, j]], j))
                .collect();
            row.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (rank, (d, idx)) in row.into_iter().take(k_actual).enumerate() {
                knn_distances[[i, rank]] = d;
                knn_indices[[i, rank]] = idx;
            }
        }

        Ok((knn_distances, knn_indices))
    }

    /// Metrics for which an `oxicuda_manifold` HNSW index can be built.
    /// Anything else (e.g. `"manhattan"`, which `HnswDistance` has no
    /// variant for) should use the exact brute-force path instead.
    #[cfg(feature = "gpu")]
    fn hnsw_distance_for_metric(metric: &str) -> Option<HnswDistance> {
        match metric {
            "euclidean" => Some(HnswDistance::Euclidean),
            "cosine" => Some(HnswDistance::Cosine),
            _ => None,
        }
    }

    /// Approximate k-nearest-neighbour search via an `oxicuda_manifold` HNSW
    /// graph index (`hnsw_build` + `hnsw_search`).
    ///
    /// `k_actual` must already be `< n_samples` (as `knn_search` ensures):
    /// this asks the index for `k_actual + 1` neighbours per query point and
    /// drops each point's self-match -- querying an HNSW index with the same
    /// points it was built from always finds each point itself as its
    /// closest neighbour, at distance 0 -- to match `cpu_knn_search`'s "k
    /// nearest OTHER points" contract. `HnswDistance::Euclidean` measures
    /// *squared* Euclidean distance, so that case is square-rooted before
    /// being returned, to match this module's other `"euclidean"` distances.
    #[cfg(feature = "gpu")]
    fn hnsw_knn_search(
        &self,
        data: &ArrayView2<Float>,
        k_actual: usize,
        hnsw_metric: HnswDistance,
    ) -> SklResult<(Array2<Float>, Array2<usize>)> {
        let (n_samples, dim) = data.dim();

        let mut flat = vec![0.0_f64; n_samples * dim];
        for i in 0..n_samples {
            for c in 0..dim {
                flat[i * dim + c] = data[[i, c]];
            }
        }

        let default_cfg = HnswConfig::default();
        let config = HnswConfig {
            distance: hnsw_metric,
            ef_search: default_cfg.ef_search.max(2 * (k_actual + 1)),
            ..default_cfg
        };
        let index = hnsw_build(&flat, n_samples, dim, &config).map_err(manifold_err)?;
        let result = hnsw_search(&index, &flat, n_samples, k_actual + 1).map_err(manifold_err)?;

        let mut knn_distances = Array2::<Float>::zeros((n_samples, k_actual));
        let mut knn_indices = Array2::<usize>::zeros((n_samples, k_actual));
        for i in 0..n_samples {
            let mut rank = 0usize;
            for (&idx, &dist) in result.indices[i].iter().zip(result.distances[i].iter()) {
                if idx == i {
                    continue;
                }
                if rank >= k_actual {
                    break;
                }
                knn_indices[[i, rank]] = idx;
                knn_distances[[i, rank]] = match hnsw_metric {
                    HnswDistance::Euclidean => dist.max(0.0).sqrt(),
                    _ => dist,
                };
                rank += 1;
            }
        }

        Ok((knn_distances, knn_indices))
    }

    pub(crate) fn cpu_knn_search(
        &self,
        data: &ArrayView2<Float>,
        k: usize,
        metric: &str,
    ) -> SklResult<(Array2<Float>, Array2<usize>)> {
        let n_samples = data.nrows();
        let k_actual = k.min(n_samples.saturating_sub(1));
        let mut knn_distances = Array2::zeros((n_samples, k_actual));
        let mut knn_indices = Array2::zeros((n_samples, k_actual));

        for i in 0..n_samples {
            let mut distances_with_indices: Vec<(Float, usize)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let dist = match metric {
                        "euclidean" => self.euclidean_distance(&data.row(i), &data.row(j)),
                        "manhattan" => self.manhattan_distance(&data.row(i), &data.row(j)),
                        "cosine" => self.cosine_distance(&data.row(i), &data.row(j)),
                        _ => {
                            return Err(SklearsError::InvalidParameter {
                                name: "metric".to_string(),
                                reason: format!("Unknown metric: {}", metric),
                            })
                        }
                    };
                    distances_with_indices.push((dist, j));
                }
            }
            distances_with_indices
                .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
            for (idx, &(dist, neighbor_idx)) in
                distances_with_indices.iter().take(k_actual).enumerate()
            {
                knn_distances[[i, idx]] = dist;
                knn_indices[[i, idx]] = neighbor_idx;
            }
        }

        Ok((knn_distances, knn_indices))
    }

    /// Accelerated matrix operations.
    pub fn matrix_operations(
        &mut self,
        operation: &str,
        data: &ArrayView2<Float>,
    ) -> SklResult<Array2<Float>> {
        match operation {
            "gram_matrix" => self.compute_gram_matrix(data),
            "laplacian" => self.compute_laplacian(data),
            "normalize" => self.normalize_matrix(data),
            _ => Err(SklearsError::InvalidParameter {
                name: "operation".to_string(),
                reason: format!("Unknown operation: {}", operation),
            }),
        }
    }

    fn compute_gram_matrix(&mut self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        #[cfg(feature = "gpu")]
        if let Some(ctx) = &self.context {
            let owned = data.to_owned();
            let x_gpu = GpuArray::<Float>::from_array2(ctx, &owned)?;
            let (n, d) = data.dim();
            let mut xt_flat = vec![0.0 as Float; n * d];
            for r in 0..n {
                for c in 0..d {
                    xt_flat[c * n + r] = data[[r, c]];
                }
            }
            let xt_array = Array2::from_shape_vec((d, n), xt_flat)
                .map_err(|e| SklearsError::InvalidInput(format!("Transpose shape error: {}", e)))?;
            let xt_gpu = GpuArray::<Float>::from_array2(ctx, &xt_array)?;
            let gram_gpu = x_gpu.matmul(&xt_gpu)?;
            return gram_gpu.to_array2();
        }
        self.cpu_gram_matrix(data)
    }

    fn cpu_gram_matrix(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        Ok(data.dot(&data.t()))
    }

    fn compute_laplacian(&self, adjacency: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let n = adjacency.nrows();
        let mut laplacian = Array2::zeros((n, n));
        let mut degrees = Array1::zeros(n);
        for i in 0..n {
            degrees[i] = adjacency.row(i).sum();
        }
        for i in 0..n {
            laplacian[[i, i]] = degrees[i];
            for j in 0..n {
                if i != j {
                    laplacian[[i, j]] = -adjacency[[i, j]];
                }
            }
        }
        Ok(laplacian)
    }

    fn normalize_matrix(&self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let mut normalized = data.to_owned();
        for mut row in normalized.rows_mut() {
            let norm = row.mapv(|x| x * x).sum().sqrt();
            if norm > 0.0 {
                row /= norm;
            }
        }
        Ok(normalized)
    }

    fn euclidean_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<Float>,
        b: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    fn manhattan_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<Float>,
        b: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    fn cosine_distance(
        &self,
        a: &scirs2_core::ndarray::ArrayView1<Float>,
        b: &scirs2_core::ndarray::ArrayView1<Float>,
    ) -> Float {
        let dot = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<Float>();
        let na = a.iter().map(|x| x * x).sum::<Float>().sqrt();
        let nb = b.iter().map(|x| x * x).sum::<Float>().sqrt();
        if na > 0.0 && nb > 0.0 {
            1.0 - dot / (na * nb)
        } else {
            0.0
        }
    }
}

/// GPU-accelerated t-SNE implementation.
///
/// Despite the name (kept for API continuity with the rest of this module),
/// [`fit_transform`](Self::fit_transform) does not run on a GPU device at
/// all: it calls `oxicuda_manifold::tsne_fit`, a pure-Rust,
/// `#![forbid(unsafe_code)]` host-side implementation of perplexity binary
/// search + early-exaggeration + momentum gradient descent. `accelerator` is
/// therefore not consulted by `fit_transform` -- it is kept only to back
/// [`is_gpu_available`](Self::is_gpu_available), which remains a meaningful
/// (if now purely informational) diagnostic.
pub struct GpuTSNE {
    accelerator: GpuAccelerator,
    n_components: usize,
    perplexity: f64,
    learning_rate: f64,
    n_iter: usize,
    random_state: Option<u64>,
}

impl GpuTSNE {
    pub fn new() -> SklResult<Self> {
        Ok(Self {
            accelerator: GpuAccelerator::new()?,
            n_components: 2,
            perplexity: 30.0,
            learning_rate: 200.0,
            n_iter: 1000,
            random_state: None,
        })
    }

    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    pub fn perplexity(mut self, perplexity: f64) -> Self {
        self.perplexity = perplexity;
        self
    }

    pub fn learning_rate(mut self, learning_rate: f64) -> Self {
        self.learning_rate = learning_rate;
        self
    }

    pub fn n_iter(mut self, n_iter: usize) -> Self {
        self.n_iter = n_iter;
        self
    }

    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit t-SNE and return the low-dimensional embedding.
    ///
    /// This calls the real `oxicuda_manifold::tsne_fit` optimiser
    /// unconditionally. Unlike `pairwise_distances`/`knn_search`, there is
    /// no "is a GPU available?" branch here at all: `tsne_fit` is a
    /// host-side algorithm, so this call *is* the implementation rather than
    /// one of two alternatives, and it never falls back to a randomly
    /// initialised, unoptimised embedding. Given the same input data,
    /// configuration, and `random_state`, it is fully deterministic (see
    /// `test_gpu_tsne_fit_transform_is_deterministic_and_separates_clusters`).
    ///
    /// # Errors
    /// Returns an error if `n_components`/`perplexity`/`learning_rate`/etc.
    /// fail `oxicuda_manifold`'s validation (e.g. `n_components == 0`), or
    /// if `data` is empty.
    #[cfg(feature = "gpu")]
    pub fn fit_transform(&mut self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let (n_samples, n_features) = data.dim();
        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidInput(
                "GpuTSNE::fit_transform requires a non-empty 2-D array".to_string(),
            ));
        }

        let mut flat = vec![0.0_f64; n_samples * n_features];
        for i in 0..n_samples {
            for j in 0..n_features {
                flat[i * n_features + j] = data[[i, j]];
            }
        }

        let cfg = TsneConfigBuilder::new()
            .n_components(self.n_components)
            .perplexity(self.perplexity)
            .learning_rate(self.learning_rate)
            .n_iter(self.n_iter)
            .build()
            .map_err(manifold_err)?;

        let mut rng = match self.random_state {
            Some(seed) => LcgRng::new(seed),
            None => LcgRng::new(thread_rng().random::<u64>()),
        };

        let result =
            tsne_fit(&flat, n_samples, n_features, &cfg, &mut rng).map_err(manifold_err)?;

        let embedding = Array2::from_shape_vec((n_samples, self.n_components), result.embedding)?;
        Ok(embedding)
    }

    /// Without the `gpu` feature, `oxicuda-manifold` (an optional
    /// dependency gated on that feature) is not compiled in at all, so
    /// there is no real t-SNE implementation available here. This
    /// deliberately errors instead of resurrecting the old behaviour of
    /// returning an unfitted random-normal embedding, which is
    /// indistinguishable from a real (if poor) fit unless the caller already
    /// distrusts the result.
    #[cfg(not(feature = "gpu"))]
    pub fn fit_transform(&mut self, _data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        Err(SklearsError::MissingDependency {
            dependency: "oxicuda-manifold".to_string(),
            feature: "gpu".to_string(),
        })
    }

    pub fn is_gpu_available(&self) -> bool {
        self.accelerator.is_gpu_available()
    }
}

/// Performance benchmarking utilities for GPU acceleration.
pub struct GpuBenchmark {
    accelerator: GpuAccelerator,
}

impl GpuBenchmark {
    pub fn new() -> SklResult<Self> {
        Ok(Self {
            accelerator: GpuAccelerator::new()?,
        })
    }

    pub fn benchmark_distance_computation(
        &mut self,
        data: &ArrayView2<Float>,
        metric: &str,
    ) -> SklResult<(f64, f64)> {
        use std::time::Instant;

        let start = Instant::now();
        let _cpu_result = self.accelerator.cpu_pairwise_distances(data, metric)?;
        let cpu_time = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let _gpu_result = self.accelerator.pairwise_distances(data, metric)?;
        let gpu_time = start.elapsed().as_secs_f64();

        Ok((cpu_time, gpu_time))
    }

    pub fn benchmark_knn_search(
        &mut self,
        data: &ArrayView2<Float>,
        k: usize,
        metric: &str,
    ) -> SklResult<(f64, f64)> {
        use std::time::Instant;

        let start = Instant::now();
        let _cpu_result = self.accelerator.cpu_knn_search(data, k, metric)?;
        let cpu_time = start.elapsed().as_secs_f64();

        let start = Instant::now();
        let _gpu_result = self.accelerator.knn_search(data, k, metric)?;
        let gpu_time = start.elapsed().as_secs_f64();

        Ok((cpu_time, gpu_time))
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_gpu_accelerator_creation() {
        let accelerator = GpuAccelerator::new();
        assert!(accelerator.is_ok());
    }

    #[test]
    fn test_pairwise_distances() {
        let mut accelerator = GpuAccelerator::new().expect("operation should succeed");
        let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let distances = accelerator
            .pairwise_distances(&data.view(), "euclidean")
            .expect("operation should succeed");

        assert_eq!(distances.shape(), &[3, 3]);
        assert_abs_diff_eq!(distances[[0, 0]], 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(distances[[0, 1]], distances[[1, 0]], epsilon = 1e-10);
    }

    #[test]
    fn test_knn_search() {
        let mut accelerator = GpuAccelerator::new().expect("operation should succeed");
        let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];

        let (distances, indices) = accelerator
            .knn_search(&data.view(), 2, "euclidean")
            .expect("operation should succeed");

        assert_eq!(distances.shape(), &[4, 2]);
        assert_eq!(indices.shape(), &[4, 2]);
    }

    #[test]
    fn test_knn_search_hnsw_matches_brute_force_on_clustered_data() {
        let mut accelerator = GpuAccelerator::new().expect("operation should succeed");
        // Two well-separated 2-D clusters: nearest neighbours are
        // unambiguous, so the approximate HNSW path (`knn_search`) and the
        // exact brute-force path (`cpu_knn_search`) should fully agree —
        // every neighbour returned for a point should be another point in
        // the same cluster.
        let data = array![
            [0.0_f64, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [50.0, 50.0],
            [50.1, 50.0],
            [50.0, 50.1],
        ];

        let (_, hnsw_indices) = accelerator
            .knn_search(&data.view(), 2, "euclidean")
            .expect("operation should succeed");
        let (_, brute_indices) = accelerator
            .cpu_knn_search(&data.view(), 2, "euclidean")
            .expect("operation should succeed");

        assert_eq!(hnsw_indices.shape(), &[6, 2]);
        assert_eq!(brute_indices.shape(), &[6, 2]);
        for i in 0..6 {
            let own_cluster = if i < 3 { 0..3 } else { 3..6 };
            for j in 0..2 {
                let neighbor = hnsw_indices[[i, j]];
                assert!(
                    own_cluster.contains(&neighbor),
                    "point {i}'s HNSW neighbor {neighbor} (rank {j}) is outside its cluster; \
                     hnsw={:?} brute-force={:?}",
                    hnsw_indices.row(i),
                    brute_indices.row(i)
                );
            }
        }
    }

    #[test]
    fn test_matrix_operations() {
        let mut accelerator = GpuAccelerator::new().expect("operation should succeed");
        let data = array![[1.0_f64, 2.0], [3.0, 4.0]];

        let gram = accelerator
            .matrix_operations("gram_matrix", &data.view())
            .expect("operation should succeed");
        assert_eq!(gram.shape(), &[2, 2]);

        let normalized = accelerator
            .matrix_operations("normalize", &data.view())
            .expect("operation should succeed");
        assert_eq!(normalized.shape(), &[2, 2]);
    }

    #[test]
    fn test_gpu_tsne() {
        let mut tsne = GpuTSNE::new().expect("operation should succeed");
        let data = array![[1.0_f64, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];

        let embedding = tsne
            .fit_transform(&data.view())
            .expect("operation should succeed");
        assert_eq!(embedding.shape(), &[3, 2]);
    }

    /// Two tight, well-separated 3-D "clusters" at fixed (non-random)
    /// coordinates, so the *input* itself is as deterministic as the fit
    /// under test. Six points per cluster mirrors the scale of
    /// `oxicuda_manifold`'s own `tsne_separates_clusters` test.
    fn well_separated_two_cluster_data() -> Array2<Float> {
        let cluster_a: [[Float; 3]; 6] = [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.05],
            [0.0, 0.1, -0.05],
            [-0.1, 0.05, 0.0],
            [0.05, -0.1, 0.05],
            [-0.05, -0.05, -0.05],
        ];
        let cluster_b: [[Float; 3]; 6] = [
            [20.0, 20.0, 20.0],
            [20.1, 20.0, 20.05],
            [20.0, 20.1, 19.95],
            [19.9, 20.05, 20.0],
            [20.05, 19.9, 20.05],
            [19.95, 19.95, 19.95],
        ];
        let flat: Vec<Float> = cluster_a
            .iter()
            .chain(cluster_b.iter())
            .flat_map(|row| row.iter().copied())
            .collect();
        Array2::from_shape_vec((cluster_a.len() + cluster_b.len(), 3), flat)
            .expect("operation should succeed")
    }

    /// The single most important test in Wave B7: `fit_transform` used to
    /// call `cpu_tsne_fallback`, which drew fresh `StandardNormal` noise and
    /// called it an embedding -- never fitting anything, never
    /// deterministic, and never cluster-preserving. This exercises exactly
    /// the two properties that bug could never have satisfied.
    #[test]
    fn test_gpu_tsne_fit_transform_is_deterministic_and_separates_clusters() {
        let data = well_separated_two_cluster_data();
        let n = data.nrows();
        let half = n / 2;

        let run = |seed: u64| -> Array2<Float> {
            GpuTSNE::new()
                .expect("operation should succeed")
                .n_components(2)
                .perplexity(5.0)
                .learning_rate(100.0)
                .n_iter(400)
                .random_state(seed)
                .fit_transform(&data.view())
                .expect("operation should succeed")
        };

        // (a) Deterministic: same seed => (near-)identical embedding, run
        // twice from two independent `GpuTSNE` instances.
        let embedding_a = run(42);
        let embedding_b = run(42);

        assert_eq!(embedding_a.shape(), &[n, 2]);
        assert_eq!(embedding_b.shape(), &[n, 2]);
        assert!(embedding_a.iter().all(|v| v.is_finite()));
        for (a, b) in embedding_a.iter().zip(embedding_b.iter()) {
            assert_abs_diff_eq!(*a, *b, epsilon = 1e-9);
        }

        // Guard against the determinism check above passing vacuously (e.g.
        // if a bug made `random_state` a no-op): a different seed should
        // essentially never reproduce the same embedding.
        let embedding_c = run(1234);
        let same_as_c = embedding_a
            .iter()
            .zip(embedding_c.iter())
            .all(|(a, c)| (a - c).abs() < 1e-9);
        assert!(
            !same_as_c,
            "different random_state seeds produced identical embeddings"
        );

        // (b) Cluster separation: the centre-to-centre distance between the
        // two clusters should exceed either cluster's own within-cluster
        // radius in the embedding.
        let centre = |rows: std::ops::Range<usize>| -> (Float, Float) {
            let count = rows.len() as Float;
            let (mut sx, mut sy) = (0.0, 0.0);
            for i in rows {
                sx += embedding_a[[i, 0]];
                sy += embedding_a[[i, 1]];
            }
            (sx / count, sy / count)
        };
        let (ax, ay) = centre(0..half);
        let (bx, by) = centre(half..n);
        let between = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();

        let max_radius = |rows: std::ops::Range<usize>, cx: Float, cy: Float| -> Float {
            rows.map(|i| {
                ((embedding_a[[i, 0]] - cx).powi(2) + (embedding_a[[i, 1]] - cy).powi(2)).sqrt()
            })
            .fold(0.0_f64, Float::max)
        };
        let radius_a = max_radius(0..half, ax, ay);
        let radius_b = max_radius(half..n, bx, by);
        let max_within = radius_a.max(radius_b);

        assert!(
            between > max_within,
            "clusters not separated in the embedding: centre distance {between} <= \
             max within-cluster radius {max_within}"
        );
    }

    #[test]
    fn test_benchmarking() {
        let mut benchmark = GpuBenchmark::new().expect("operation should succeed");
        let data = array![[1.0_f64, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let (cpu_time, gpu_time) = benchmark
            .benchmark_distance_computation(&data.view(), "euclidean")
            .expect("operation should succeed");
        assert!(cpu_time >= 0.0);
        assert!(gpu_time >= 0.0);
    }
}
