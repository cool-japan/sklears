//! Isomap (Isometric Mapping) implementation
//!
//! This module provides Isomap for non-linear dimensionality reduction through isometric mapping.

use scirs2_core::ndarray::{Array1, Array2, ArrayView2, Axis};
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
    // The training data used at fit time (needed to measure distances to new points).
    training_data: Array2<f64>,
    // Top `n_components` eigenvectors of the double-centered matrix, one per column,
    // in the same order used to build `embedding`.
    eigenvectors: Array2<f64>,
    // Corresponding top `n_components` eigenvalues (0.0 for any that were <= 1e-12 and
    // thus unused, mirroring how `embedding` zero-fills those columns).
    eigenvalues: Array1<f64>,
    // Row means of the training D^2 matrix (== column means, since D^2 is symmetric);
    // used to center new points consistently with training.
    row_means: Array1<f64>,
    // Grand mean of the training D^2 matrix.
    grand_mean: f64,
}

/// Intermediate products of classical MDS retained so that out-of-sample points can
/// later be projected onto the training embedding via Gower's supplementary formula.
struct ClassicalMdsResult {
    embedding: Array2<f64>,
    eigenvectors: Array2<f64>,
    eigenvalues: Array1<f64>,
    row_means: Array1<f64>,
    grand_mean: f64,
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

        // Apply classical MDS, retaining the eigenbasis and centering statistics so
        // that new points can be projected onto this embedding later.
        let mds = self.classical_mds(&geodesic_distances)?;

        // Compute reconstruction error
        let reconstruction_error =
            self.compute_reconstruction_error(&geodesic_distances, &mds.embedding);

        Ok(Isomap {
            state: IsomapTrained {
                embedding: mds.embedding,
                geodesic_distances,
                reconstruction_error,
                training_data: x,
                eigenvectors: mds.eigenvectors,
                eigenvalues: mds.eigenvalues,
                row_means: mds.row_means,
                grand_mean: mds.grand_mean,
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
            neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).expect("operation should succeed"));

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

    fn classical_mds(&self, distances: &Array2<f64>) -> SklResult<ClassicalMdsResult> {
        let n = distances.nrows();

        // Center the squared distance matrix using double centering
        let mut d_squared = distances.mapv(|x| x * x);
        let row_means = d_squared
            .mean_axis(Axis(1))
            .expect("operation should succeed");
        let col_means = d_squared
            .mean_axis(Axis(0))
            .expect("operation should succeed");
        let grand_mean = d_squared.mean().expect("operation should succeed");

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
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).expect("operation should succeed"));

        // Take the largest n_components eigenvalues, retaining the eigenbasis so that
        // out-of-sample points can later be projected via Gower's formula.
        let mut embedding = Array2::zeros((n, self.n_components));
        let mut eigenvectors: Array2<f64> = Array2::zeros((n, self.n_components));
        let mut eigenvalues: Array1<f64> = Array1::zeros(self.n_components);
        for (comp_idx, &(eigenval, eigen_idx)) in
            eigen_pairs.iter().take(self.n_components).enumerate()
        {
            // Always retain the eigenvector column in embedding order; transform only
            // consults the columns whose eigenvalue is strictly positive.
            for i in 0..n {
                eigenvectors[[i, comp_idx]] = eigenvecs[[i, eigen_idx]];
            }
            if eigenval > 1e-12 {
                // Only use positive eigenvalues; the rest stay zero-filled in both
                // `embedding` and `eigenvalues`.
                eigenvalues[comp_idx] = eigenval;
                let sqrt_eigenval = eigenval.sqrt();
                for i in 0..n {
                    embedding[[i, comp_idx]] = eigenvecs[[i, eigen_idx]] * sqrt_eigenval;
                }
            }
        }

        Ok(ClassicalMdsResult {
            embedding,
            eigenvectors,
            eigenvalues,
            row_means,
            grand_mean,
        })
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for Isomap<IsomapTrained> {
    /// Project new (out-of-sample) points onto the trained Isomap embedding.
    ///
    /// This implements the standard Isomap out-of-sample extension (Gower's formula for
    /// supplementary points in classical MDS, with geodesic distances substituted for
    /// Euclidean ones). For each new point `p` we:
    ///
    /// 1. measure its direct Euclidean distance to every training point;
    /// 2. take its `k` nearest training points (`k = n_neighbors`, clamped to the
    ///    training size);
    /// 3. approximate its geodesic distance to each training point `j` by entering the
    ///    training graph through one of those neighbours:
    ///    `g[j] = min_nb (direct(p, nb) + geodesic(nb, j))`. Because the training
    ///    geodesic matrix already satisfies the triangle inequality, this is a valid
    ///    shortest-path approximation;
    /// 4. double-center the squared approximate distances against the stored training
    ///    statistics; and
    /// 5. project the result onto the retained eigenbasis.
    ///
    /// Feeding the exact training data back through this method reproduces the training
    /// embedding (up to floating-point error), because a training point is its own
    /// nearest neighbour at distance zero and, by the triangle inequality, `g` collapses
    /// to the exact training geodesic row.
    fn transform(&self, x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        let n_features = self.state.training_data.ncols();
        if x.ncols() != n_features {
            return Err(SklearsError::FeatureMismatch {
                expected: n_features,
                actual: x.ncols(),
            });
        }

        let n_train = self.state.training_data.nrows();
        let n_new = x.nrows();
        let n_components = self.n_components;
        // Number of training neighbours used to re-enter the geodesic graph.
        let k = self.n_neighbors.min(n_train).max(1);

        let mut embedding_new = Array2::zeros((n_new, n_components));

        for p in 0..n_new {
            let point = x.row(p);

            // Step 1: direct Euclidean distance to every training point.
            let mut direct = vec![0.0_f64; n_train];
            for (t, d) in direct.iter_mut().enumerate() {
                let diff = &point - &self.state.training_data.row(t);
                *d = diff.mapv(|v| v * v).sum().sqrt();
            }

            // Step 2: indices of the k nearest training points (by direct distance).
            let mut order: Vec<usize> = (0..n_train).collect();
            order.sort_by(|&a, &b| direct[a].total_cmp(&direct[b]));
            let neighbors = &order[..k];

            // Steps 3-4: approximate squared geodesic distance to every training point by
            // entering the graph through the new point's nearest neighbours, then take
            // this new point's own row mean of those squared distances.
            let mut d_p = vec![0.0_f64; n_train];
            for (j, dp) in d_p.iter_mut().enumerate() {
                let mut g = f64::INFINITY;
                for &nb in neighbors {
                    let candidate = direct[nb] + self.state.geodesic_distances[[nb, j]];
                    if candidate < g {
                        g = candidate;
                    }
                }
                *dp = g * g;
            }
            let m_p = d_p.iter().sum::<f64>() / n_train as f64;

            // Steps 5-6: double-center against the training statistics and project onto
            // the stored eigenbasis:
            //   b_p[j]          = -0.5 (d_p[j] - row_means[j] - m_p + grand_mean)
            //   embedding[p, c] = (sum_j b_p[j] * eigenvectors[j, c]) / sqrt(eigenvalues[c]).
            for c in 0..n_components {
                let eigenval = self.state.eigenvalues[c];
                if eigenval > 1e-12 {
                    let mut acc = 0.0;
                    for (j, &d_p_j) in d_p.iter().enumerate() {
                        let b_p =
                            -0.5 * (d_p_j - self.state.row_means[j] - m_p + self.state.grand_mean);
                        acc += b_p * self.state.eigenvectors[[j, c]];
                    }
                    embedding_new[[p, c]] = acc / eigenval.sqrt();
                }
                // eigenvalue <= 1e-12: leave this coordinate at 0.0, matching `embedding`.
            }
        }

        Ok(embedding_new)
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

#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    // A 3x2 grid of points. Consecutive rows and columns are close enough that a
    // k-NN graph with k = 3 is connected, which Isomap requires, and the data spans two
    // dimensions so both embedding components are meaningful.
    fn training_data() -> Array2<f64> {
        array![
            [0.0, 0.0],
            [1.0, 0.0],
            [2.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [2.0, 1.0],
        ]
    }

    fn fitted_isomap() -> Isomap<IsomapTrained> {
        let x = training_data();
        Isomap::new()
            .n_neighbors(3)
            .n_components(2)
            .fit(&x.view(), &())
            .expect("fit should succeed on a connected k-NN graph")
    }

    #[test]
    fn fit_produces_embedding_of_expected_shape() {
        let fitted = fitted_isomap();
        assert_eq!(fitted.embedding().dim(), (6, 2));
    }

    #[test]
    fn transform_shape_matches_new_data() {
        let fitted = fitted_isomap();
        let new = array![[0.5, 0.5], [1.5, 0.25], [0.25, 0.75]];
        let out = fitted
            .transform(&new.view())
            .expect("transform should succeed");
        assert_eq!(out.dim(), (new.nrows(), 2));
    }

    #[test]
    fn transform_is_a_function_of_the_input() {
        let fitted = fitted_isomap();

        // Two genuinely different out-of-sample sets must yield different embeddings.
        let a = array![[0.5, 0.5], [1.5, 0.5]];
        let b = array![[0.1, 0.9], [1.9, 0.1]];
        let out_a = fitted.transform(&a.view()).expect("transform a");
        let out_b = fitted.transform(&b.view()).expect("transform b");

        let mut max_diff = 0.0_f64;
        for (va, vb) in out_a.iter().zip(out_b.iter()) {
            max_diff = max_diff.max((va - vb).abs());
        }
        assert!(
            max_diff > 1e-6,
            "different inputs must map to different outputs (max_diff = {max_diff})"
        );

        // The transform of new data must NOT simply replay the training embedding
        // (this is the exact bug being fixed).
        let emb = fitted.embedding();
        let mut replays_training = true;
        for r in 0..out_a.nrows() {
            for c in 0..2 {
                if (out_a[[r, c]] - emb[[r, c]]).abs() > 1e-9 {
                    replays_training = false;
                }
            }
        }
        assert!(
            !replays_training,
            "transform of new data must not return the stale training embedding"
        );
    }

    #[test]
    fn self_transform_reproduces_training_embedding() {
        let fitted = fitted_isomap();
        let x = training_data();

        let round_trip = fitted
            .transform(&x.view())
            .expect("self-transform should succeed");
        let emb = fitted.embedding();
        assert_eq!(round_trip.dim(), emb.dim());
        for (got, want) in round_trip.iter().zip(emb.iter()) {
            assert!(
                (got - want).abs() < 1e-6,
                "self-transform must reproduce the training embedding: got {got}, want {want}"
            );
        }
    }

    #[test]
    fn near_duplicate_maps_close_to_original_embedding() {
        let fitted = fitted_isomap();
        let emb = fitted.embedding();

        // Perturb training point 0 = [0, 0] by a tiny amount along each coordinate.
        let perturbed = array![[1e-4, 1e-4]];
        let out = fitted
            .transform(&perturbed.view())
            .expect("transform should succeed");

        let dist_to_own =
            ((out[[0, 0]] - emb[[0, 0]]).powi(2) + (out[[0, 1]] - emb[[0, 1]]).powi(2)).sqrt();

        // It must land closer to its own original embedding row than to any other row.
        for i in 1..emb.nrows() {
            let dist_to_other =
                ((out[[0, 0]] - emb[[i, 0]]).powi(2) + (out[[0, 1]] - emb[[i, 1]]).powi(2)).sqrt();
            assert!(
                dist_to_own < dist_to_other,
                "perturbed point 0 should map nearest to embedding row 0, but row {i} \
                 was closer (own = {dist_to_own}, other = {dist_to_other})"
            );
        }

        // And in absolute terms it should stay very close to the original.
        assert!(
            dist_to_own < 1e-2,
            "a tiny perturbation should stay near the original embedding (dist = {dist_to_own})"
        );
    }

    #[test]
    fn transform_rejects_feature_mismatch() {
        let fitted = fitted_isomap();

        // Three features where the model was trained on two.
        let bad = array![[0.0, 1.0, 2.0]];
        let result = fitted.transform(&bad.view());
        assert!(
            matches!(
                result,
                Err(SklearsError::FeatureMismatch {
                    expected: 2,
                    actual: 3
                })
            ),
            "feature-count mismatch must be rejected with FeatureMismatch"
        );
    }
}
