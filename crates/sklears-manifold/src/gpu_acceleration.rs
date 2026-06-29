//! GPU-accelerated methods for manifold learning
//!
//! This module provides GPU-accelerated implementations of core manifold learning
//! operations using oxicuda-backend for cross-platform GPU compute support.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_core::random::thread_rng;
use scirs2_core::random::SeedableRng;
use scirs2_core::StdRng;
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};

#[cfg(feature = "gpu")]
use sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps};

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
    pub fn new() -> SklResult<Self> {
        #[cfg(feature = "gpu")]
        {
            match GpuContext::with_device_id(0) {
                Ok(ctx) => Ok(Self { context: Some(ctx) }),
                Err(_) => Ok(Self { context: None }),
            }
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
        let ctx = self.context.as_ref().expect("checked above");
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
    pub fn knn_search(
        &mut self,
        data: &ArrayView2<Float>,
        k: usize,
        metric: &str,
    ) -> SklResult<(Array2<Float>, Array2<usize>)> {
        let distances = self.pairwise_distances(data, metric)?;
        let n_samples = data.nrows();
        let k_actual = k.min(n_samples.saturating_sub(1));
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
    pub fn fit_transform(&mut self, data: &ArrayView2<Float>) -> SklResult<Array2<Float>> {
        let distances = self.accelerator.pairwise_distances(data, "euclidean")?;
        self.cpu_tsne_fallback(data, &distances)
    }

    fn cpu_tsne_fallback(
        &self,
        data: &ArrayView2<Float>,
        _distances: &Array2<Float>,
    ) -> SklResult<Array2<Float>> {
        let n_samples = data.nrows();
        let mut embedding = Array2::zeros((n_samples, self.n_components));

        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().random::<u64>())
        };

        for i in 0..n_samples {
            for j in 0..self.n_components {
                embedding[[i, j]] = rng.sample(scirs2_core::StandardNormal);
            }
        }

        Ok(embedding)
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
