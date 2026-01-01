//! Isomap (Isometric Mapping) implementation
//!
//! This module provides Isomap for non-linear dimensionality reduction through isometric mapping.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Isomap embedding
///
/// Non-linear dimensionality reduction through Isometric Mapping.
/// Isomap seeks a lower-dimensional embedding which maintains geodesic
/// distances between all points.
///
/// # Parameters
///
/// * `n_neighbors` - Number of neighbors to consider for each point
/// * `n_components` - Number of coordinates for the manifold
/// * `eigen_solver` - The eigensolver to use
/// * `tol` - Convergence tolerance passed to arpack or lobpcg
/// * `max_iter` - Maximum number of iterations for the arpack solver
/// * `path_method` - Method to use in finding shortest path
/// * `neighbors_algorithm` - Algorithm to use for nearest neighbors search
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::Isomap;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let isomap = Isomap::new()
///     .n_neighbors(2)
///     .n_components(2);
/// let fitted = isomap.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct Isomap<S = Untrained> {
    state: S,
    n_neighbors: usize,
    n_components: usize,
    eigen_solver: String,
    tol: f64,
    max_iter: Option<usize>,
    path_method: String,
    neighbors_algorithm: String,
    n_jobs: Option<i32>,
}

/// Trained state for Isomap
#[derive(Debug, Clone)]
pub struct IsomapTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// Matrix of geodesic distances between all points
    pub geodesic_distances: Array2<f64>,
    /// Reconstruction error from the embedding
    pub reconstruction_error: f64,
}

impl Isomap<Untrained> {
    /// Create a new Isomap instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_neighbors: 5,
            n_components: 2,
            eigen_solver: "auto".to_string(),
            tol: 1e-6,
            max_iter: None,
            path_method: "auto".to_string(),
            neighbors_algorithm: "auto".to_string(),
            n_jobs: None,
        }
    }

    /// Set the number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set the eigen solver
    pub fn eigen_solver(mut self, eigen_solver: &str) -> Self {
        self.eigen_solver = eigen_solver.to_string();
        self
    }

    /// Set the tolerance
    pub fn tol(mut self, tol: f64) -> Self {
        self.tol = tol;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: Option<usize>) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set the path method
    pub fn path_method(mut self, path_method: &str) -> Self {
        self.path_method = path_method.to_string();
        self
    }

    /// Set the neighbors algorithm
    pub fn neighbors_algorithm(mut self, neighbors_algorithm: &str) -> Self {
        self.neighbors_algorithm = neighbors_algorithm.to_string();
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for Isomap<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for Isomap<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for Isomap<Untrained> {
    type Fitted = Isomap<IsomapTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _) = x.dim();

        if n_samples <= self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        // Compute pairwise distances
        let distances = self.compute_pairwise_distances(&x)?;

        // Build k-nearest neighbors graph
        let graph = self.build_knn_graph(&distances)?;

        // Compute shortest path distances
        let geodesic_distances = self.compute_shortest_paths(&graph)?;

        // Apply classical MDS
        let embedding = self.classical_mds(&geodesic_distances)?;

        // Compute reconstruction error
        let reconstruction_error =
            self.compute_reconstruction_error(&geodesic_distances, &embedding);

        Ok(Isomap {
            state: IsomapTrained {
                embedding,
                geodesic_distances,
                reconstruction_error,
            },
            n_neighbors: self.n_neighbors,
            n_components: self.n_components,
            eigen_solver: self.eigen_solver,
            tol: self.tol,
            max_iter: self.max_iter,
            path_method: self.path_method,
            neighbors_algorithm: self.neighbors_algorithm,
            n_jobs: self.n_jobs,
        })
    }
}

impl Isomap<Untrained> {
    fn compute_pairwise_distances(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = x.nrows();
        let mut distances = Array2::zeros((n, n));

        for i in 0..n {
            for j in i + 1..n {
                let diff = &x.row(i) - &x.row(j);
                let dist = diff.mapv(|x| x * x).sum().sqrt();
                distances[[i, j]] = dist;
                distances[[j, i]] = dist;
            }
        }

        Ok(distances)
    }

    fn build_knn_graph(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();
        let mut graph = Array2::from_elem((n, n), f64::INFINITY);

        // Set diagonal to 0
        for i in 0..n {
            graph[[i, i]] = 0.0;
        }

        // For each point, connect to k nearest neighbors
        for i in 0..n {
            let mut neighbors: Vec<(usize, f64)> = (0..n).map(|j| (j, distances[[i, j]])).collect();
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

            for &(neighbor_idx, distance) in
                neighbors.iter().skip(1).take(self.n_neighbors.min(n - 1))
            {
                // Skip self (index 0)
                graph[[i, neighbor_idx]] = distance;
                graph[[neighbor_idx, i]] = distance; // Make symmetric
            }
        }

        Ok(graph)
    }

    fn compute_shortest_paths(&self, graph: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = graph.nrows();
        let mut distances = graph.clone();

        // Floyd-Warshall algorithm for all-pairs shortest paths
        for k in 0..n {
            for i in 0..n {
                for j in 0..n {
                    if distances[[i, k]] != f64::INFINITY && distances[[k, j]] != f64::INFINITY {
                        let new_dist = distances[[i, k]] + distances[[k, j]];
                        if new_dist < distances[[i, j]] {
                            distances[[i, j]] = new_dist;
                        }
                    }
                }
            }
        }

        // Check for disconnected components
        for i in 0..n {
            for j in 0..n {
                if distances[[i, j]] == f64::INFINITY && i != j {
                    return Err(SklearsError::InvalidInput(
                        "Graph is not connected. Consider increasing n_neighbors or using a different neighborhood method.".to_string(),
                    ));
                }
            }
        }

        Ok(distances)
    }

    fn compute_reconstruction_error(
        &self,
        original_distances: &Array2<f64>,
        embedding: &Array2<f64>,
    ) -> f64 {
        let n = embedding.nrows();
        let mut total_error = 0.0;
        let mut count = 0;

        // Compute embedding distances
        for i in 0..n {
            for j in i + 1..n {
                let embedded_diff = &embedding.row(i) - &embedding.row(j);
                let embedded_dist = embedded_diff.mapv(|x| x * x).sum().sqrt();
                let original_dist = original_distances[[i, j]];

                let error = (embedded_dist - original_dist).powi(2);
                total_error += error;
                count += 1;
            }
        }

        if count > 0 {
            total_error / count as f64
        } else {
            0.0
        }
    }

    fn classical_mds(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();

        // Center the squared distance matrix using double centering
        let mut d_squared = distances.mapv(|x| x * x);
        let row_means = d_squared.mean_axis(Axis(1)).unwrap();
        let col_means = d_squared.mean_axis(Axis(0)).unwrap();
        let grand_mean = d_squared.mean().unwrap();

        // Double centering: B = -1/2 * J * D^2 * J where J = I - (1/n) * 1 * 1^T
        for i in 0..n {
            for j in 0..n {
                d_squared[[i, j]] =
                    -0.5 * (d_squared[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // Symmetrize the matrix to ensure numerical stability for eigendecomposition
        let symmetric_matrix = (&d_squared + &d_squared.t()) / 2.0;

        // Eigendecomposition of the centered matrix
        let (eigenvals, eigenvecs) = symmetric_matrix
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {e}")))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take the largest n_components eigenvalues
        let mut embedding = Array2::zeros((n, self.n_components));
        for (comp_idx, &(eigenval, eigen_idx)) in
            eigen_pairs.iter().take(self.n_components).enumerate()
        {
            if eigenval > 1e-12 {
                // Only use positive eigenvalues
                let sqrt_eigenval = eigenval.sqrt();
                for i in 0..n {
                    embedding[[i, comp_idx]] = eigenvecs[[i, eigen_idx]] * sqrt_eigenval;
                }
            }
        }

        Ok(embedding)
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for Isomap<IsomapTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // Isomap doesn't support transforming new data in this implementation
        Ok(self.state.embedding.clone())
    }
}

impl Isomap<IsomapTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the geodesic distances
    pub fn geodesic_distances(&self) -> &Array2<f64> {
        &self.state.geodesic_distances
    }

    /// Get the reconstruction error
    pub fn reconstruction_error(&self) -> f64 {
        self.state.reconstruction_error
    }
}
