//! GPU-Accelerated Distance Computations
//!
//! This module provides GPU-accelerated distance computations for clustering algorithms
//! using `oxicuda-backend` (the Pure Rust CUDA replacement) for GPU compute support.
//!
//! # Features
//! - **Euclidean Distance**: Batch computation of Euclidean distances
//! - **Manhattan Distance**: Batch computation of Manhattan distances
//! - **Cosine Distance**: Batch computation of cosine distances
//! - **Distance Matrix**: GPU-accelerated distance matrix computation
//! - **K-Nearest Neighbors**: GPU-accelerated k-NN search
//! - **Memory Management**: Efficient GPU buffer management for large datasets
//!
//! # GPU Acceleration Strategy
//! Euclidean and squared-Euclidean distances above `GpuConfig::gpu_threshold` use the
//! GEMM trick, powered by `oxicuda-backend`/`oxicuda-blas` via
//! `sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps}`:
//! - `D[i,j] = ||x_i||² + ||y_j||² - 2 * x_i · y_j`
//! - The `X · Y^T` term is computed on the GPU through `GpuArray::matmul`
//! - Row norms are computed on the CPU (inexpensive relative to the GEMM)
//! - Manhattan/Cosine distances and datasets below the threshold fall back to CPU
//!
//! # Usage
//! ```rust,ignore
//! use sklears_clustering::gpu_distances::{GpuDistanceComputer, GpuDistanceMetric};
//!
//! // Create GPU distance computer
//! let mut gpu_computer = GpuDistanceComputer::new().await?;
//!
//! // Compute distances between datasets
//! let distances = gpu_computer.compute_pairwise_distances(
//!     &data1, &data2, GpuDistanceMetric::Euclidean
//! ).await?;
//! ```

#[cfg(feature = "gpu")]
/// GPU-accelerated distance computation implementations powered by oxicuda-backend.
///
/// Euclidean and squared-Euclidean distances use the GEMM trick:
/// `D[i,j] = ||x_i||² + ||y_j||² - 2 * x_i · y_j`
pub mod gpu {
    use std::collections::HashMap;

    use scirs2_core::ndarray::Array2;
    use sklears_core::error::{Result, SklearsError};
    use sklears_core::gpu::{GpuArray, GpuContext, GpuMatrixOps};

    /// GPU distance metrics
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GpuDistanceMetric {
        /// Standard L2 (Euclidean) distance.
        Euclidean,
        /// L1 (Manhattan) distance.
        Manhattan,
        /// Cosine distance (1 − cosine similarity).
        Cosine,
        /// Squared Euclidean distance (no sqrt).
        SquaredEuclidean,
    }

    /// Configuration for GPU distance computation
    #[derive(Debug, Clone)]
    pub struct GpuConfig {
        /// Minimum number of points to use GPU acceleration
        pub gpu_threshold: usize,
        /// GPU device ID
        pub device_id: usize,
    }

    impl Default for GpuConfig {
        fn default() -> Self {
            Self {
                gpu_threshold: 64,
                device_id: 0,
            }
        }
    }

    /// GPU-accelerated distance computer.
    pub struct GpuDistanceComputer {
        /// `Some` when a real GPU/driver was detected at construction time;
        /// `None` means construction gracefully found no usable GPU (see
        /// [`GpuContext::with_device_id`]'s `Option`-returning contract) and
        /// every operation below transparently routes through the CPU
        /// fallbacks instead of the GEMM path.
        context: Option<GpuContext>,
        config: GpuConfig,
    }

    impl GpuDistanceComputer {
        /// Create a `GpuDistanceComputer` using default configuration.
        pub async fn new() -> Result<Self> {
            Self::with_config(GpuConfig::default()).await
        }

        /// Create a `GpuDistanceComputer` with a specific configuration.
        ///
        /// This never fails solely because no GPU is present:
        /// `GpuContext::with_device_id` returns `Ok(None)` in that case, which
        /// is stored as `context: None` here rather than surfaced as an error,
        /// so construction always succeeds and later calls to
        /// `compute_pairwise_distances` simply use the CPU path.
        pub async fn with_config(config: GpuConfig) -> Result<Self> {
            let context = GpuContext::with_device_id(config.device_id).map_err(|e| {
                SklearsError::InvalidInput(format!(
                    "Failed to initialize GPU device {}: {}",
                    config.device_id, e
                ))
            })?;
            Ok(Self { context, config })
        }

        /// Compute pairwise distances between rows of `data_a` and `data_b`.
        ///
        /// Euclidean and squared-Euclidean use the GEMM trick when a GPU was
        /// detected at construction time *and* the dataset exceeds
        /// `config.gpu_threshold`; otherwise (no GPU, or below the threshold)
        /// this falls back to CPU.
        pub async fn compute_pairwise_distances(
            &mut self,
            data_a: &Array2<f64>,
            data_b: &Array2<f64>,
            metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            if data_a.ncols() != data_b.ncols() {
                return Err(SklearsError::InvalidInput(
                    "Data dimensions must match".to_string(),
                ));
            }

            let use_gpu = self.context.is_some()
                && data_a.nrows() + data_b.nrows() >= self.config.gpu_threshold;

            match metric {
                GpuDistanceMetric::Euclidean | GpuDistanceMetric::SquaredEuclidean if use_gpu => {
                    self.gemm_euclidean_distances(data_a, data_b, metric)
                }
                GpuDistanceMetric::Euclidean => {
                    Self::cpu_euclidean_distances(data_a, data_b, false)
                }
                GpuDistanceMetric::SquaredEuclidean => {
                    Self::cpu_euclidean_distances(data_a, data_b, true)
                }
                GpuDistanceMetric::Manhattan => Self::cpu_manhattan_distances(data_a, data_b),
                GpuDistanceMetric::Cosine => Self::cpu_cosine_distances(data_a, data_b),
            }
        }

        /// Compute distance matrix for a single dataset (all pairwise distances).
        pub async fn compute_distance_matrix(
            &mut self,
            data: &Array2<f64>,
            metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            self.compute_pairwise_distances(data, data, metric).await
        }

        /// k-nearest-neighbours search: returns indices and distances for each query point.
        pub async fn k_nearest_neighbors(
            &mut self,
            query: &Array2<f64>,
            data: &Array2<f64>,
            k: usize,
        ) -> Result<(Array2<usize>, Array2<f64>)> {
            let distances = self
                .compute_pairwise_distances(query, data, GpuDistanceMetric::Euclidean)
                .await?;

            let n_query = query.nrows();
            let n_data = data.nrows();
            let k_actual = k.min(n_data);

            let mut indices = Array2::<usize>::zeros((n_query, k_actual));
            let mut dists = Array2::<f64>::zeros((n_query, k_actual));

            for i in 0..n_query {
                let mut row: Vec<(usize, f64)> =
                    (0..n_data).map(|j| (j, distances[[i, j]])).collect();
                row.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                for (rank, (idx, d)) in row.into_iter().take(k_actual).enumerate() {
                    indices[[i, rank]] = idx;
                    dists[[i, rank]] = d;
                }
            }

            Ok((indices, dists))
        }

        /// Return a map of backend/device diagnostic strings.
        pub fn device_info(&self) -> HashMap<String, String> {
            let mut info = HashMap::new();
            if let Some(context) = &self.context {
                if let Ok(mem) = context.memory_info() {
                    info.insert("total_memory_bytes".to_string(), mem.total.to_string());
                    info.insert("free_memory_bytes".to_string(), mem.free.to_string());
                }
            }
            info.insert("backend".to_string(), "oxicuda-backend".to_string());
            info.insert(
                "gpu_available".to_string(),
                self.context.is_some().to_string(),
            );
            info
        }

        // ── Internal helpers ────────────────────────────────────────────────────

        /// Euclidean GEMM trick: D[i,j] = sqrt(||x_i||² + ||y_j||² - 2*x_i·y_j)
        ///
        /// The only caller, `compute_pairwise_distances`, checks `use_gpu`
        /// (which requires `self.context.is_some()`) before reaching here, but
        /// the context is still resolved via `ok_or_else` rather than indexed
        /// unconditionally so a future call-site mistake surfaces as an
        /// ordinary `Result::Err` instead of a panic.
        fn gemm_euclidean_distances(
            &self,
            x: &Array2<f64>,
            y: &Array2<f64>,
            metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            let context = self.context.as_ref().ok_or_else(|| {
                SklearsError::InvalidOperation(
                    "gemm_euclidean_distances requires a detected GPU context".to_string(),
                )
            })?;
            let (m, _d) = x.dim();
            let (n, _) = y.dim();

            // Upload X and Y^T to GPU
            let x_gpu = GpuArray::<f64>::from_array2(context, x)?;
            let yt: Vec<f64> = {
                let (nr, nc) = y.dim();
                let mut t = vec![0.0f64; nr * nc];
                for r in 0..nr {
                    for c in 0..nc {
                        t[c * nr + r] = y[[r, c]];
                    }
                }
                t
            };
            let yt_array = Array2::from_shape_vec((y.ncols(), y.nrows()), yt)
                .map_err(|e| SklearsError::InvalidInput(format!("Transpose shape error: {}", e)))?;
            let yt_gpu = GpuArray::<f64>::from_array2(context, &yt_array)?;

            // inner = X · Y^T  [m × n]
            let inner_gpu = x_gpu.matmul(&yt_gpu)?;
            let inner = inner_gpu.to_array2()?;

            // Row norms computed CPU-side (small)
            let x_norms: Vec<f64> = x
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f64>())
                .collect();
            let y_norms: Vec<f64> = y
                .rows()
                .into_iter()
                .map(|row| row.iter().map(|v| v * v).sum::<f64>())
                .collect();

            let mut distances = Array2::<f64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let sq = (x_norms[i] + y_norms[j] - 2.0 * inner[[i, j]]).max(0.0);
                    distances[[i, j]] = if matches!(metric, GpuDistanceMetric::SquaredEuclidean) {
                        sq
                    } else {
                        sq.sqrt()
                    };
                }
            }
            Ok(distances)
        }

        fn cpu_euclidean_distances(
            x: &Array2<f64>,
            y: &Array2<f64>,
            squared: bool,
        ) -> Result<Array2<f64>> {
            let (m, d) = x.dim();
            let (n, d2) = y.dim();
            if d != d2 {
                return Err(SklearsError::InvalidInput("Dimension mismatch".to_string()));
            }
            let mut out = Array2::<f64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let sq: f64 = (0..d).map(|k| (x[[i, k]] - y[[j, k]]).powi(2)).sum();
                    out[[i, j]] = if squared { sq } else { sq.sqrt() };
                }
            }
            Ok(out)
        }

        fn cpu_manhattan_distances(x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>> {
            let (m, d) = x.dim();
            let (n, d2) = y.dim();
            if d != d2 {
                return Err(SklearsError::InvalidInput("Dimension mismatch".to_string()));
            }
            let mut out = Array2::<f64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut s = 0.0f64;
                    for k in 0..d {
                        s += (x[[i, k]] - y[[j, k]]).abs();
                    }
                    out[[i, j]] = s;
                }
            }
            Ok(out)
        }

        fn cpu_cosine_distances(x: &Array2<f64>, y: &Array2<f64>) -> Result<Array2<f64>> {
            let (m, d) = x.dim();
            let (n, d2) = y.dim();
            if d != d2 {
                return Err(SklearsError::InvalidInput("Dimension mismatch".to_string()));
            }
            let mut out = Array2::<f64>::zeros((m, n));
            for i in 0..m {
                for j in 0..n {
                    let mut dot = 0.0f64;
                    let mut na = 0.0f64;
                    let mut nb = 0.0f64;
                    for k in 0..d {
                        dot += x[[i, k]] * y[[j, k]];
                        na += x[[i, k]] * x[[i, k]];
                        nb += y[[j, k]] * y[[j, k]];
                    }
                    let denom = na.sqrt() * nb.sqrt();
                    out[[i, j]] = if denom < f64::EPSILON {
                        1.0
                    } else {
                        1.0 - dot / denom
                    };
                }
            }
            Ok(out)
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn test_euclidean_distances() {
            pollster::block_on(async {
                let mut computer = GpuDistanceComputer::new()
                    .await
                    .expect("Should create GPU computer");

                let data_a =
                    Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 1.0, 0.0]).expect("shape");
                let data_b =
                    Array2::from_shape_vec((2, 2), vec![3.0, 4.0, 0.0, 0.0]).expect("shape");

                let distances = computer
                    .compute_pairwise_distances(&data_a, &data_b, GpuDistanceMetric::Euclidean)
                    .await
                    .expect("Should compute distances");

                // data_a[0]=(0,0), data_b[0]=(3,4): sqrt(9+16)=5
                assert!((distances[[0, 0]] - 5.0).abs() < 1e-5);
                // data_a[1]=(1,0), data_b[0]=(3,4): sqrt(4+16)=sqrt(20)
                assert!((distances[[1, 0]] - 20.0_f64.sqrt()).abs() < 1e-5);
                // data_a[0]=(0,0), data_b[1]=(0,0): 0
                assert!(distances[[0, 1]].abs() < 1e-5);
                // data_a[1]=(1,0), data_b[1]=(0,0): 1
                assert!((distances[[1, 1]] - 1.0).abs() < 1e-5);
            });
        }

        #[test]
        fn test_manhattan_distances() {
            pollster::block_on(async {
                let mut computer = GpuDistanceComputer::new()
                    .await
                    .expect("Should create GPU computer");

                let data_a = Array2::from_shape_vec((1, 2), vec![1.0, 2.0]).expect("shape");
                let data_b = Array2::from_shape_vec((1, 2), vec![3.0, 4.0]).expect("shape");

                let distances = computer
                    .compute_pairwise_distances(&data_a, &data_b, GpuDistanceMetric::Manhattan)
                    .await
                    .expect("Should compute distances");

                assert!((distances[[0, 0]] - 4.0).abs() < 1e-6);
            });
        }
    }
}

// Re-export GPU module when feature is enabled
#[cfg(feature = "gpu")]
pub use gpu::*;

// Provide stub implementations when GPU feature is not enabled
#[cfg(not(feature = "gpu"))]
pub mod stub {
    use sklears_core::error::{Result, SklearsError};

    /// Stub GPU distance metric (no-op when GPU feature disabled)
    #[derive(Debug, Clone, Copy, PartialEq)]
    pub enum GpuDistanceMetric {
        Euclidean,
        Manhattan,
        Cosine,
        SquaredEuclidean,
    }

    /// Stub GPU distance computer (returns error when GPU feature disabled)
    pub struct GpuDistanceComputer;

    impl GpuDistanceComputer {
        pub async fn new() -> Result<Self> {
            Err(SklearsError::InvalidInput(
                "GPU feature not enabled. Enable with --features gpu".to_string(),
            ))
        }

        pub async fn compute_pairwise_distances(
            &mut self,
            _data_a: &Array2<f64>,
            _data_b: &Array2<f64>,
            _metric: GpuDistanceMetric,
        ) -> Result<Array2<f64>> {
            Err(SklearsError::InvalidInput(
                "GPU feature not enabled".to_string(),
            ))
        }
    }
}

#[cfg(not(feature = "gpu"))]
pub use stub::*;
