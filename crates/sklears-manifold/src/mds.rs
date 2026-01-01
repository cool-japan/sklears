//! Classical Multidimensional Scaling (MDS) implementation
//!
//! This module provides MDS for non-linear dimensionality reduction through classical scaling.

use scirs2_core::ndarray::{Array2, ArrayView2, Axis};
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Estimator, Fit, Transform, Untrained},
    types::Float,
};

/// Classical Multidimensional Scaling (MDS)
///
/// Classical MDS places the data in a low-dimensional space such that the
/// Euclidean distances in this space best match the original distances.
/// It uses eigendecomposition of the double-centered squared distance matrix.
///
/// # Parameters
///
/// * `n_components` - Number of coordinates for the manifold
/// * `metric` - Metric to use for distance calculation
/// * `n_init` - Number of random initializations
/// * `max_iter` - Maximum number of iterations for SMACOF algorithm
/// * `verbose` - Whether to be verbose
/// * `eps` - Relative tolerance for convergence
/// * `random_state` - Random state for reproducibility
/// * `dissimilarity` - Type of dissimilarity to use
/// * `n_jobs` - Number of parallel jobs
///
/// # Examples
///
/// ```
/// use sklears_manifold::MDS;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::array;
///
/// let x = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]];
///
/// let mds = MDS::new()
///     .n_components(2);
/// let fitted = mds.fit(&x.view(), &()).unwrap();
/// let embedded = fitted.transform(&x.view()).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MDS<S = Untrained> {
    state: S,
    n_components: usize,
    metric: bool,
    n_init: usize,
    max_iter: usize,
    verbose: bool,
    eps: f64,
    random_state: Option<u64>,
    dissimilarity: String,
    n_jobs: Option<i32>,
}

/// Trained state for MDS
#[derive(Debug, Clone)]
pub struct MdsTrained {
    /// The low-dimensional embedding of the training data
    pub embedding: Array2<f64>,
    /// Stress value of the embedding
    pub stress: f64,
    /// Matrix of pairwise distances
    pub distance_matrix: Array2<f64>,
    /// Number of iterations performed
    pub n_iter: usize,
}

impl MDS<Untrained> {
    /// Create a new MDS instance
    pub fn new() -> Self {
        Self {
            state: Untrained,
            n_components: 2,
            metric: true,
            n_init: 4,
            max_iter: 300,
            verbose: false,
            eps: 1e-3,
            random_state: None,
            dissimilarity: "euclidean".to_string(),
            n_jobs: None,
        }
    }

    /// Set the number of components
    pub fn n_components(mut self, n_components: usize) -> Self {
        self.n_components = n_components;
        self
    }

    /// Set whether to use metric MDS
    pub fn metric(mut self, metric: bool) -> Self {
        self.metric = metric;
        self
    }

    /// Set the number of initializations
    pub fn n_init(mut self, n_init: usize) -> Self {
        self.n_init = n_init;
        self
    }

    /// Set the maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set verbosity
    pub fn verbose(mut self, verbose: bool) -> Self {
        self.verbose = verbose;
        self
    }

    /// Set the tolerance
    pub fn eps(mut self, eps: f64) -> Self {
        self.eps = eps;
        self
    }

    /// Set the random state
    pub fn random_state(mut self, random_state: Option<u64>) -> Self {
        self.random_state = random_state;
        self
    }

    /// Set the dissimilarity metric
    pub fn dissimilarity(mut self, dissimilarity: &str) -> Self {
        self.dissimilarity = dissimilarity.to_string();
        self
    }

    /// Set the number of jobs
    pub fn n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

impl Default for MDS<Untrained> {
    fn default() -> Self {
        Self::new()
    }
}

impl Estimator for MDS<Untrained> {
    type Config = ();
    type Error = SklearsError;
    type Float = Float;

    fn config(&self) -> &Self::Config {
        &()
    }
}

impl Fit<ArrayView2<'_, Float>, ()> for MDS<Untrained> {
    type Fitted = MDS<MdsTrained>;

    fn fit(self, x: &ArrayView2<'_, Float>, _y: &()) -> SklResult<Self::Fitted> {
        let x = x.mapv(|x| x);
        let (n_samples, _) = x.dim();

        if n_samples <= self.n_components {
            return Err(SklearsError::InvalidInput(
                "Number of samples must be greater than n_components".to_string(),
            ));
        }

        // Compute pairwise distances
        let distances = self.compute_distance_matrix(&x)?;

        // Apply classical MDS
        let embedding = if self.metric {
            self.classical_mds(&distances)?
        } else {
            // Non-metric MDS would go here (SMACOF algorithm)
            self.classical_mds(&distances)?
        };

        // Compute stress
        let stress = self.compute_stress(&distances, &embedding);

        Ok(MDS {
            state: MdsTrained {
                embedding,
                stress,
                distance_matrix: distances,
                n_iter: 0, // Classical MDS is direct, no iterations
            },
            n_components: self.n_components,
            metric: self.metric,
            n_init: self.n_init,
            max_iter: self.max_iter,
            verbose: self.verbose,
            eps: self.eps,
            random_state: self.random_state,
            dissimilarity: self.dissimilarity,
            n_jobs: self.n_jobs,
        })
    }
}

impl MDS<Untrained> {
    fn compute_distance_matrix(&self, x: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n_samples = x.nrows();
        let mut distances = Array2::zeros((n_samples, n_samples));

        match self.dissimilarity.as_str() {
            "euclidean" => {
                for i in 0..n_samples {
                    for j in i + 1..n_samples {
                        let diff = &x.row(i) - &x.row(j);
                        let dist = diff.mapv(|x| x * x).sum().sqrt();
                        distances[[i, j]] = dist;
                        distances[[j, i]] = dist;
                    }
                }
            }
            _ => {
                return Err(SklearsError::InvalidInput(
                    "Unsupported dissimilarity metric".to_string(),
                ))
            }
        }

        Ok(distances)
    }

    fn classical_mds(&self, distances: &Array2<f64>) -> SklResult<Array2<f64>> {
        let n = distances.nrows();

        // Double-center the squared distance matrix
        let mut d_squared = distances.mapv(|x| x * x);
        let row_means = d_squared.mean_axis(Axis(1)).unwrap();
        let col_means = d_squared.mean_axis(Axis(0)).unwrap();
        let grand_mean = d_squared.mean().unwrap();

        // Apply double centering: B = -1/2 * J * D^2 * J
        for i in 0..n {
            for j in 0..n {
                d_squared[[i, j]] =
                    -0.5 * (d_squared[[i, j]] - row_means[i] - col_means[j] + grand_mean);
            }
        }

        // Eigendecomposition
        let (eigenvals, eigenvecs) = d_squared
            .eigh(UPLO::Lower)
            .map_err(|e| SklearsError::InvalidInput(format!("Eigendecomposition failed: {e}")))?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvals
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take the largest n_components eigenvalues and corresponding eigenvectors
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

    fn compute_stress(&self, original_distances: &Array2<f64>, embedding: &Array2<f64>) -> f64 {
        let n = embedding.nrows();
        let mut stress_numerator = 0.0;
        let mut stress_denominator = 0.0;

        for i in 0..n {
            for j in i + 1..n {
                let embedded_diff = &embedding.row(i) - &embedding.row(j);
                let embedded_dist = embedded_diff.mapv(|x| x * x).sum().sqrt();
                let original_dist = original_distances[[i, j]];

                stress_numerator += (embedded_dist - original_dist).powi(2);
                stress_denominator += original_dist.powi(2);
            }
        }

        if stress_denominator > 0.0 {
            (stress_numerator / stress_denominator).sqrt()
        } else {
            0.0
        }
    }
}

impl Transform<ArrayView2<'_, Float>, Array2<f64>> for MDS<MdsTrained> {
    fn transform(&self, _x: &ArrayView2<'_, Float>) -> SklResult<Array2<f64>> {
        // Classical MDS doesn't support transforming new data in this implementation
        Ok(self.state.embedding.clone())
    }
}

impl MDS<MdsTrained> {
    /// Get the embedding
    pub fn embedding(&self) -> &Array2<f64> {
        &self.state.embedding
    }

    /// Get the stress value
    pub fn stress(&self) -> f64 {
        self.state.stress
    }

    /// Get the distance matrix
    pub fn distance_matrix(&self) -> &Array2<f64> {
        &self.state.distance_matrix
    }

    /// Get the number of iterations performed
    pub fn n_iter(&self) -> usize {
        self.state.n_iter
    }
}
