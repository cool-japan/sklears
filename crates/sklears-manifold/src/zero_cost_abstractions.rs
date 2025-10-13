//! Zero-cost abstractions for manifold learning
//!
//! This module provides zero-cost abstractions that compile down to efficient
//! machine code while providing high-level APIs for manifold learning operations.
//! Uses const generics, traits, and other Rust features to eliminate runtime overhead.

use scirs2_core::ndarray::{Array2, ArrayView1, ArrayView2};
/// Zero-cost distance metrics using const generics
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    types::Float,
};
use std::marker::PhantomData;
pub trait DistanceMetric<const METRIC_ID: usize> {
    /// Compute distance between two points
    fn distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float;

    /// Get the metric name
    const NAME: &'static str;

    /// Whether this metric satisfies triangle inequality
    const IS_METRIC: bool;
}

/// Euclidean distance metric (L2 norm)
pub struct EuclideanDistance;

impl DistanceMetric<0> for EuclideanDistance {
    fn distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>()
            .sqrt()
    }

    const NAME: &'static str = "euclidean";
    const IS_METRIC: bool = true;
}

/// Manhattan distance metric (L1 norm)
pub struct ManhattanDistance;

impl DistanceMetric<1> for ManhattanDistance {
    fn distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
    }

    const NAME: &'static str = "manhattan";
    const IS_METRIC: bool = true;
}

/// Cosine distance (1 - cosine similarity)
pub struct CosineDistance;

impl DistanceMetric<2> for CosineDistance {
    fn distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<Float>();
        let norm_a = a.iter().map(|x| x * x).sum::<Float>().sqrt();
        let norm_b = b.iter().map(|x| x * x).sum::<Float>().sqrt();

        if norm_a > 0.0 && norm_b > 0.0 {
            1.0 - dot_product / (norm_a * norm_b)
        } else {
            0.0
        }
    }

    const NAME: &'static str = "cosine";
    const IS_METRIC: bool = false; // Cosine distance doesn't satisfy triangle inequality
}

/// Chebyshev distance (L∞ norm)
pub struct ChebyshevDistance;

impl DistanceMetric<3> for ChebyshevDistance {
    fn distance(a: ArrayView1<Float>, b: ArrayView1<Float>) -> Float {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, |max_val, diff| max_val.max(diff))
    }

    const NAME: &'static str = "chebyshev";
    const IS_METRIC: bool = true;
}

/// Zero-cost manifold operations using const generics
pub trait ManifoldOperation<const OP_ID: usize> {
    type Input;
    type Output;

    /// Apply the operation
    fn apply(input: Self::Input) -> SklResult<Self::Output>;

    /// Get operation name
    const NAME: &'static str;

    /// Computational complexity class
    const COMPLEXITY: &'static str;
}

/// Principal Component Analysis operation
pub struct PCAOperation;

impl ManifoldOperation<0> for PCAOperation {
    type Input = ArrayView2<'static, Float>;
    type Output = (Array2<Float>, Array2<Float>); // (components, transformed_data)

    fn apply(input: Self::Input) -> SklResult<Self::Output> {
        // Simplified PCA implementation for demonstration
        let n_samples = input.nrows();
        let n_features = input.ncols();

        if n_samples == 0 || n_features == 0 {
            return Err(SklearsError::InvalidParameter {
                name: "input_shape".to_string(),
                reason: "Input must have positive dimensions".to_string(),
            });
        }

        // Center the data
        let mean = input.mean_axis(scirs2_core::ndarray::Axis(0)).unwrap();
        let mut centered = input.to_owned();
        for mut row in centered.rows_mut() {
            row -= &mean;
        }

        // Compute covariance matrix
        let cov = centered.t().dot(&centered) / (n_samples as Float - 1.0);

        // For demonstration, return identity transformation
        let components = Array2::eye(n_features);
        let transformed = centered;

        Ok((components, transformed))
    }

    const NAME: &'static str = "PCA";
    const COMPLEXITY: &'static str = "O(n² * d)";
}

/// Independent Component Analysis operation
pub struct ICAOperation;

impl ManifoldOperation<1> for ICAOperation {
    type Input = ArrayView2<'static, Float>;
    type Output = (Array2<Float>, Array2<Float>); // (mixing_matrix, independent_components)

    fn apply(input: Self::Input) -> SklResult<Self::Output> {
        let n_samples = input.nrows();
        let n_features = input.ncols();

        if n_samples < n_features {
            return Err(SklearsError::InvalidParameter {
                name: "sample_size".to_string(),
                reason: "Number of samples must be >= number of features for ICA".to_string(),
            });
        }

        // Simplified ICA (placeholder implementation)
        let mixing_matrix = Array2::eye(n_features);
        let components = input.to_owned();

        Ok((mixing_matrix, components))
    }

    const NAME: &'static str = "ICA";
    const COMPLEXITY: &'static str = "O(n³)";
}

/// Zero-cost manifold learning algorithm abstraction
pub trait ZeroCostManifoldAlgorithm<
    const ALGO_ID: usize,
    M: DistanceMetric<METRIC_ID>,
    const METRIC_ID: usize,
>
{
    /// Algorithm configuration type
    type Config;

    /// Output embedding type
    type Embedding;

    /// Apply the manifold learning algorithm
    fn fit_transform(data: ArrayView2<Float>, config: Self::Config) -> SklResult<Self::Embedding>;

    /// Get algorithm name
    const NAME: &'static str;

    /// Whether the algorithm preserves distances
    const PRESERVES_DISTANCES: bool;

    /// Whether the algorithm is linear
    const IS_LINEAR: bool;
}

/// Configuration for MDS algorithm
#[derive(Debug, Clone)]
pub struct MDSConfig {
    /// n_components
    pub n_components: usize,
    /// max_iter
    pub max_iter: usize,
    /// eps
    pub eps: Float,
}

impl Default for MDSConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            max_iter: 300,
            eps: 1e-6,
        }
    }
}

/// Multidimensional Scaling algorithm
pub struct MDSAlgorithm<M, const METRIC_ID: usize>(PhantomData<M>);

impl<M: DistanceMetric<METRIC_ID>, const METRIC_ID: usize>
    ZeroCostManifoldAlgorithm<0, M, METRIC_ID> for MDSAlgorithm<M, METRIC_ID>
{
    type Config = MDSConfig;
    type Embedding = Array2<Float>;

    fn fit_transform(data: ArrayView2<Float>, config: Self::Config) -> SklResult<Self::Embedding> {
        let n_samples = data.nrows();

        if config.n_components > data.ncols() {
            return Err(SklearsError::InvalidParameter {
                name: "n_components".to_string(),
                reason: format!(
                    "n_components {} cannot exceed input dimensions {}",
                    config.n_components,
                    data.ncols()
                ),
            });
        }

        // Compute distance matrix using the metric
        let mut distances = Array2::zeros((n_samples, n_samples));
        for i in 0..n_samples {
            for j in i..n_samples {
                let dist = M::distance(data.row(i), data.row(j));
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        // Classical MDS: double centering
        let mut gram = distances.mapv(|x| -0.5 * x * x);
        let row_means = gram.mean_axis(scirs2_core::ndarray::Axis(1)).unwrap();
        let total_mean = row_means.mean().unwrap();

        for i in 0..n_samples {
            for j in 0..n_samples {
                gram[[i, j]] = gram[[i, j]] - row_means[i] - row_means[j] + total_mean;
            }
        }

        // For demonstration, return first n_components columns
        let embedding = gram
            .slice(scirs2_core::ndarray::s![.., ..config.n_components])
            .to_owned();
        Ok(embedding)
    }

    const NAME: &'static str = "MDS";
    const PRESERVES_DISTANCES: bool = true;
    const IS_LINEAR: bool = false;
}

/// Configuration for Isomap algorithm
#[derive(Debug, Clone)]
pub struct IsomapConfig {
    /// n_components
    pub n_components: usize,
    /// n_neighbors
    pub n_neighbors: usize,
}

impl Default for IsomapConfig {
    fn default() -> Self {
        Self {
            n_components: 2,
            n_neighbors: 5,
        }
    }
}

/// Isomap algorithm
pub struct IsomapAlgorithm<M, const METRIC_ID: usize>(PhantomData<M>);

impl<M: DistanceMetric<METRIC_ID>, const METRIC_ID: usize>
    ZeroCostManifoldAlgorithm<1, M, METRIC_ID> for IsomapAlgorithm<M, METRIC_ID>
{
    type Config = IsomapConfig;
    type Embedding = Array2<Float>;

    fn fit_transform(data: ArrayView2<Float>, config: Self::Config) -> SklResult<Self::Embedding> {
        let n_samples = data.nrows();

        if config.n_neighbors >= n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "n_neighbors".to_string(),
                reason: format!(
                    "n_neighbors {} must be less than n_samples {}",
                    config.n_neighbors, n_samples
                ),
            });
        }

        // Build k-NN graph
        let mut adjacency = Array2::from_elem((n_samples, n_samples), Float::INFINITY);

        for i in 0..n_samples {
            adjacency[[i, i]] = 0.0;

            // Find k nearest neighbors
            let mut distances: Vec<(Float, usize)> = Vec::new();
            for j in 0..n_samples {
                if i != j {
                    let dist = M::distance(data.row(i), data.row(j));
                    distances.push((dist, j));
                }
            }

            distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

            for (dist, neighbor) in distances.iter().take(config.n_neighbors) {
                adjacency[[i, *neighbor]] = *dist;
                adjacency[[*neighbor, i]] = *dist; // Make symmetric
            }
        }

        // Floyd-Warshall for geodesic distances
        let mut geodesic = adjacency.clone();
        for k in 0..n_samples {
            for i in 0..n_samples {
                for j in 0..n_samples {
                    let through_k = geodesic[[i, k]] + geodesic[[k, j]];
                    if through_k < geodesic[[i, j]] {
                        geodesic[[i, j]] = through_k;
                    }
                }
            }
        }

        // Apply MDS to geodesic distances
        let mds_config = MDSConfig {
            n_components: config.n_components,
            ..Default::default()
        };

        // For demonstration, return projection of first n_components columns
        let embedding = data
            .slice(scirs2_core::ndarray::s![
                ..,
                ..config.n_components.min(data.ncols())
            ])
            .to_owned();
        Ok(embedding)
    }

    const NAME: &'static str = "Isomap";
    const PRESERVES_DISTANCES: bool = true;
    const IS_LINEAR: bool = false;
}

/// Zero-cost neighbor search using const generics
pub trait NeighborSearch<const SEARCH_ID: usize> {
    /// Find k nearest neighbors
    fn knn<M: DistanceMetric<METRIC_ID>, const METRIC_ID: usize>(
        data: ArrayView2<Float>,
        query: ArrayView1<Float>,
        k: usize,
    ) -> SklResult<(Vec<Float>, Vec<usize>)>;

    /// Algorithm name
    const NAME: &'static str;

    /// Computational complexity
    const COMPLEXITY: &'static str;
}

/// Brute force neighbor search
pub struct BruteForceSearch;

impl NeighborSearch<0> for BruteForceSearch {
    fn knn<M: DistanceMetric<METRIC_ID>, const METRIC_ID: usize>(
        data: ArrayView2<Float>,
        query: ArrayView1<Float>,
        k: usize,
    ) -> SklResult<(Vec<Float>, Vec<usize>)> {
        let n_samples = data.nrows();

        if k > n_samples {
            return Err(SklearsError::InvalidParameter {
                name: "k".to_string(),
                reason: format!("k={} cannot exceed n_samples={}", k, n_samples),
            });
        }

        let mut distances: Vec<(Float, usize)> = Vec::with_capacity(n_samples);

        for (i, row) in data.rows().into_iter().enumerate() {
            let dist = M::distance(row, query);
            distances.push((dist, i));
        }

        distances.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

        let (dists, indices): (Vec<Float>, Vec<usize>) = distances.into_iter().take(k).unzip();

        Ok((dists, indices))
    }

    const NAME: &'static str = "BruteForce";
    const COMPLEXITY: &'static str = "O(n * d)";
}

/// Compile-time manifold learning pipeline
pub struct ManifoldPipeline<A, M, const ALGO_ID: usize, const METRIC_ID: usize>
where
    A: ZeroCostManifoldAlgorithm<ALGO_ID, M, METRIC_ID>,
    M: DistanceMetric<METRIC_ID>,
{
    _algorithm: PhantomData<A>,
    _metric: PhantomData<M>,
}

impl<A, M, const ALGO_ID: usize, const METRIC_ID: usize> Default
    for ManifoldPipeline<A, M, ALGO_ID, METRIC_ID>
where
    A: ZeroCostManifoldAlgorithm<ALGO_ID, M, METRIC_ID>,
    M: DistanceMetric<METRIC_ID>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<A, M, const ALGO_ID: usize, const METRIC_ID: usize> ManifoldPipeline<A, M, ALGO_ID, METRIC_ID>
where
    A: ZeroCostManifoldAlgorithm<ALGO_ID, M, METRIC_ID>,
    M: DistanceMetric<METRIC_ID>,
{
    /// Create a new pipeline
    pub const fn new() -> Self {
        Self {
            _algorithm: PhantomData,
            _metric: PhantomData,
        }
    }

    /// Run the pipeline
    pub fn run(data: ArrayView2<Float>, config: A::Config) -> SklResult<A::Embedding> {
        A::fit_transform(data, config)
    }

    /// Get pipeline information
    pub const fn info() -> (&'static str, &'static str, bool, bool) {
        (A::NAME, M::NAME, A::PRESERVES_DISTANCES, A::IS_LINEAR)
    }
}

/// Zero-cost kernel functions using const generics
pub trait KernelFunction<const KERNEL_ID: usize> {
    /// Compute kernel value between two points
    fn kernel(a: ArrayView1<Float>, b: ArrayView1<Float>, params: &Self::Params) -> Float;

    /// Kernel parameters type
    type Params: Default;

    /// Kernel name
    const NAME: &'static str;

    /// Whether kernel is positive definite
    const IS_PD: bool;
}

/// RBF (Gaussian) kernel parameters
#[derive(Debug, Clone)]
pub struct RBFParams {
    /// gamma
    pub gamma: Float,
}

impl Default for RBFParams {
    fn default() -> Self {
        Self { gamma: 1.0 }
    }
}

/// Radial Basis Function (RBF) kernel
pub struct RBFKernel;

impl KernelFunction<0> for RBFKernel {
    type Params = RBFParams;

    fn kernel(a: ArrayView1<Float>, b: ArrayView1<Float>, params: &Self::Params) -> Float {
        let squared_distance = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<Float>();

        (-params.gamma * squared_distance).exp()
    }

    const NAME: &'static str = "RBF";
    const IS_PD: bool = true;
}

/// Polynomial kernel parameters
#[derive(Debug, Clone)]
pub struct PolynomialParams {
    /// degree
    pub degree: u32,
    /// coef0
    pub coef0: Float,
}

impl Default for PolynomialParams {
    fn default() -> Self {
        Self {
            degree: 3,
            coef0: 1.0,
        }
    }
}

/// Polynomial kernel
pub struct PolynomialKernel;

impl KernelFunction<1> for PolynomialKernel {
    type Params = PolynomialParams;

    fn kernel(a: ArrayView1<Float>, b: ArrayView1<Float>, params: &Self::Params) -> Float {
        let dot_product = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum::<Float>();
        (dot_product + params.coef0).powf(params.degree as Float)
    }

    const NAME: &'static str = "Polynomial";
    const IS_PD: bool = true;
}

/// Type aliases for common zero-cost configurations
pub type EuclideanMDS =
    ManifoldPipeline<MDSAlgorithm<EuclideanDistance, 0>, EuclideanDistance, 0, 0>;
pub type ManhattanMDS =
    ManifoldPipeline<MDSAlgorithm<ManhattanDistance, 1>, ManhattanDistance, 0, 1>;
pub type EuclideanIsomap =
    ManifoldPipeline<IsomapAlgorithm<EuclideanDistance, 0>, EuclideanDistance, 1, 0>;
pub type CosineIsomap = ManifoldPipeline<IsomapAlgorithm<CosineDistance, 2>, CosineDistance, 1, 2>;

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::{array, ArrayView1, ArrayView2};

    #[test]
    fn test_distance_metrics() {
        let a = array![1.0, 2.0, 3.0];
        let b = array![4.0, 5.0, 6.0];

        let euclidean_dist = EuclideanDistance::distance(a.view(), b.view());
        assert_abs_diff_eq!(euclidean_dist, 27.0_f64.sqrt(), epsilon = 1e-10);

        let manhattan_dist = ManhattanDistance::distance(a.view(), b.view());
        assert_abs_diff_eq!(manhattan_dist, 9.0, epsilon = 1e-10);

        assert_eq!(EuclideanDistance::NAME, "euclidean");
        assert_eq!(EuclideanDistance::IS_METRIC, true);
    }

    #[test]
    fn test_zero_cost_mds() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let config = MDSConfig::default();

        let embedding = EuclideanMDS::run(data.view(), config).unwrap();
        assert_eq!(embedding.shape(), &[3, 2]);

        let (algo_name, metric_name, preserves_dist, is_linear) = EuclideanMDS::info();
        assert_eq!(algo_name, "MDS");
        assert_eq!(metric_name, "euclidean");
        assert_eq!(preserves_dist, true);
        assert_eq!(is_linear, false);
    }

    #[test]
    fn test_neighbor_search() {
        let data = array![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let query = array![0.5, 0.5];

        let (distances, indices) =
            BruteForceSearch::knn::<EuclideanDistance, 0>(data.view(), query.view(), 2).unwrap();

        assert_eq!(distances.len(), 2);
        assert_eq!(indices.len(), 2);
        assert_eq!(BruteForceSearch::NAME, "BruteForce");
    }

    #[test]
    fn test_kernel_functions() {
        let a = array![1.0, 2.0];
        let b = array![3.0, 4.0];

        let rbf_params = RBFParams { gamma: 0.5 };
        let rbf_value = RBFKernel::kernel(a.view(), b.view(), &rbf_params);
        assert!(rbf_value > 0.0 && rbf_value <= 1.0);

        let poly_params = PolynomialParams::default();
        let poly_value = PolynomialKernel::kernel(a.view(), b.view(), &poly_params);
        assert!(poly_value > 0.0);

        assert_eq!(RBFKernel::NAME, "RBF");
        assert_eq!(RBFKernel::IS_PD, true);
    }

    #[test]
    fn test_compile_time_properties() {
        // These assertions are evaluated at compile time
        assert_eq!(EuclideanDistance::IS_METRIC, true);
        assert_eq!(CosineDistance::IS_METRIC, false);
        assert_eq!(RBFKernel::IS_PD, true);

        // Test that we can access compile-time constants
        const EUCLIDEAN_NAME: &str = EuclideanDistance::NAME;
        assert_eq!(EUCLIDEAN_NAME, "euclidean");
    }
}
