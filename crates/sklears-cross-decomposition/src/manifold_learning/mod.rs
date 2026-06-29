//! Manifold Learning Integration for Cross-Decomposition Methods
//!
//! This module provides comprehensive manifold learning techniques for nonlinear analysis
//! in cross-decomposition algorithms. It implements various manifold learning
//! methods that can capture nonlinear relationships between variables.
//!
//! ## Supported Methods
//! - Locally Linear Embedding (LLE)
//! - Isomap (Isometric Mapping)
//! - Laplacian Eigenmaps
//! - t-SNE (t-Distributed Stochastic Neighbor Embedding)
//! - UMAP (Uniform Manifold Approximation and Projection)
//! - Diffusion Maps
//! - Kernel Principal Component Analysis (Kernel PCA)
//! - Advanced Manifold Learning with comprehensive algorithms
//!
//! ## Applications
//! - Nonlinear dimensionality reduction before CCA/PLS
//! - Manifold-aware canonical correlation analysis
//! - Nonlinear feature extraction for cross-modal analysis

pub mod advanced_manifold;

// Re-export from the original manifold_learning.rs file content
use scirs2_core::ndarray::{Array1, Array2, ArrayView2};
use scirs2_linalg::compat::{eigh, svd, UPLO};
use scirs2_linalg::LinalgError;
use sklears_core::types::Float;

// Re-export the advanced manifold learning components
pub use advanced_manifold::{
    AdvancedManifoldLearning, ConvergenceInfo, CrossModalAlignment, DistanceMetric, EigenSolver,
    FittedManifoldCCA, GeodesicMethod, ManifoldCCA, ManifoldError,
    ManifoldMethod as AdvancedManifoldMethod, ManifoldProperties, ManifoldRegularization,
    ManifoldResults, OptimizationParams, PathMethod,
};

/// Manifold learning method types (original simple interface)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ManifoldMethod {
    /// Locally Linear Embedding
    LLE,
    /// Isometric Mapping
    Isomap,
    /// Laplacian Eigenmaps
    LaplacianEigenmaps,
    /// t-Distributed Stochastic Neighbor Embedding
    TSNE,
    /// Uniform Manifold Approximation and Projection
    UMAP,
    /// Diffusion Maps
    DiffusionMaps,
    /// Kernel Principal Component Analysis
    KernelPCA,
}

/// Simple manifold learning configuration
#[derive(Debug, Clone)]
pub struct ManifoldLearning {
    /// Method type
    method: ManifoldMethod,
    /// Number of components for embedding
    n_components: usize,
    /// Number of neighbors for local computations
    n_neighbors: usize,
    /// Random state for reproducibility
    random_state: Option<u64>,
}

/// Results from manifold learning
#[derive(Debug, Clone)]
pub struct ManifoldLearningResult {
    /// Low-dimensional embedding
    pub embedding: Array2<Float>,
    /// Explained variance ratio (if applicable)
    pub explained_variance_ratio: Option<Array1<Float>>,
    /// Reconstruction error
    pub reconstruction_error: Float,
}

/// Manifold-aware CCA implementation
#[derive(Debug, Clone)]
pub struct ManifoldAwareCCA {
    /// Number of canonical components
    n_components: usize,
    /// Manifold learning configuration for X
    manifold_x: ManifoldLearning,
    /// Manifold learning configuration for Y
    manifold_y: ManifoldLearning,
    /// Regularization parameter
    regularization: Float,
}

/// Fitted manifold-aware CCA model
#[derive(Debug, Clone)]
pub struct FittedManifoldAwareCCA {
    /// Manifold embedding for X
    x_embedding: Array2<Float>,
    /// Manifold embedding for Y
    y_embedding: Array2<Float>,
    /// Canonical vectors in embedding space
    x_canonical: Array2<Float>,
    y_canonical: Array2<Float>,
    /// Canonical correlations
    canonical_correlations: Array1<Float>,
}

impl ManifoldLearning {
    /// Create a new manifold learning configuration
    pub fn new(method: ManifoldMethod, n_components: usize) -> Self {
        Self {
            method,
            n_components,
            n_neighbors: 10,
            random_state: None,
        }
    }

    /// Set number of neighbors
    pub fn n_neighbors(mut self, n_neighbors: usize) -> Self {
        self.n_neighbors = n_neighbors;
        self
    }

    /// Set random state
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }

    /// Fit and transform data using manifold learning
    pub fn fit_transform(
        &self,
        data: ArrayView2<Float>,
    ) -> Result<ManifoldLearningResult, ManifoldError> {
        // The simple interface is a thin wrapper that maps the user-facing method
        // selection onto the comprehensive `AdvancedManifoldLearning` engine, which
        // performs the actual embedding computation in `advanced_manifold`.
        let advanced_method = match self.method {
            ManifoldMethod::LLE => AdvancedManifoldMethod::LocallyLinearEmbedding {
                n_neighbors: self.n_neighbors,
                reg_parameter: 0.001,
                eigen_solver: EigenSolver::Standard,
            },
            ManifoldMethod::Isomap => AdvancedManifoldMethod::Isomap {
                n_neighbors: self.n_neighbors,
                geodesic_method: GeodesicMethod::Dijkstra,
                path_method: PathMethod::Shortest,
            },
            ManifoldMethod::LaplacianEigenmaps => AdvancedManifoldMethod::LaplacianEigenmaps {
                sigma: 1.0,
                reg_parameter: 0.01,
                use_normalized_laplacian: true,
            },
            ManifoldMethod::TSNE => AdvancedManifoldMethod::TSNE {
                perplexity: 30.0,
                early_exaggeration: 12.0,
                learning_rate: 200.0,
                n_iter: 1000,
                min_grad_norm: 1e-7,
            },
            ManifoldMethod::UMAP => AdvancedManifoldMethod::UMAP {
                n_neighbors: self.n_neighbors,
                min_dist: 0.1,
                spread: 1.0,
                repulsion_strength: 1.0,
                n_epochs: 200,
            },
            ManifoldMethod::DiffusionMaps => AdvancedManifoldMethod::DiffusionMaps {
                n_neighbors: self.n_neighbors,
                alpha: 0.5,
                diffusion_time: 1,
                epsilon: 1.0,
            },
            ManifoldMethod::KernelPCA => {
                // Kernel PCA is exposed in the simple `ManifoldMethod` enum but the
                // advanced engine (`AdvancedManifoldMethod`) has no Kernel PCA backend.
                // Silently substituting a different algorithm (e.g. LLE) would return
                // an embedding that misrepresents what was requested, so we surface an
                // honest error instead of fabricating a result.
                return Err(ManifoldError::InvalidParameters(
                    "Kernel PCA is not implemented for the manifold-learning interface; \
                     no Kernel PCA backend exists in AdvancedManifoldLearning. Select \
                     another method (LLE, Isomap, LaplacianEigenmaps, TSNE, UMAP, or \
                     DiffusionMaps)."
                        .to_string(),
                ));
            }
        };

        // Use the requested embedding dimension (bounded by the data's feature
        // count) as the intrinsic-dimension hint for the advanced engine.
        let intrinsic_dim = self.n_components.min(data.ncols());

        let advanced_manifold = AdvancedManifoldLearning::new(intrinsic_dim, self.n_components)
            .method(advanced_method)
            .n_neighbors(self.n_neighbors);

        let advanced_result = advanced_manifold.fit_transform(data)?;

        Ok(ManifoldLearningResult {
            embedding: advanced_result.embedding,
            explained_variance_ratio: None, // Not applicable to all methods
            reconstruction_error: advanced_result.reconstruction_error,
        })
    }
}

impl ManifoldAwareCCA {
    /// Create a new manifold-aware CCA
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            manifold_x: ManifoldLearning::new(ManifoldMethod::UMAP, n_components),
            manifold_y: ManifoldLearning::new(ManifoldMethod::UMAP, n_components),
            regularization: 0.01,
        }
    }

    /// Set manifold learning method for X
    pub fn manifold_x(mut self, manifold: ManifoldLearning) -> Self {
        self.manifold_x = manifold;
        self
    }

    /// Set manifold learning method for Y
    pub fn manifold_y(mut self, manifold: ManifoldLearning) -> Self {
        self.manifold_y = manifold;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Fit manifold-aware CCA
    pub fn fit(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
    ) -> Result<FittedManifoldAwareCCA, ManifoldError> {
        // Apply manifold learning to both datasets
        let x_result = self.manifold_x.fit_transform(x)?;
        let y_result = self.manifold_y.fit_transform(y)?;

        // Perform CCA on the manifold embeddings
        let (x_canonical, y_canonical, correlations) =
            self.compute_canonical_correlation(&x_result.embedding, &y_result.embedding)?;

        Ok(FittedManifoldAwareCCA {
            x_embedding: x_result.embedding,
            y_embedding: y_result.embedding,
            x_canonical,
            y_canonical,
            canonical_correlations: correlations,
        })
    }

    #[allow(clippy::type_complexity)] // returns (x_weights, y_weights, correlations) triple
    fn compute_canonical_correlation(
        &self,
        x_embedding: &Array2<Float>,
        y_embedding: &Array2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>, Array1<Float>), ManifoldError> {
        let n_samples = x_embedding.nrows();
        let n_x = x_embedding.ncols();
        let n_y = y_embedding.ncols();
        let n_components = self.n_components.min(n_x).min(n_y);

        // Center the data
        let x_centered = self.center_data(x_embedding)?;
        let y_centered = self.center_data(y_embedding)?;

        // Compute cross-covariance matrix
        let mut cxy = Array2::zeros((n_x, n_y));
        for i in 0..n_x {
            for j in 0..n_y {
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += x_centered[[k, i]] * y_centered[[k, j]];
                }
                cxy[[i, j]] = cov / (n_samples - 1) as Float;
            }
        }

        // Compute auto-covariance matrices with regularization
        let cxx = self.compute_regularized_covariance(&x_centered)?;
        let cyy = self.compute_regularized_covariance(&y_centered)?;

        // Solve the CCA generalized eigenvalue problem on the embedding-space
        // covariances via whitening + SVD of the cross-covariance.
        let (correlations, x_canonical, y_canonical) =
            self.solve_cca_eigenvalue_problem(&cxx, &cyy, &cxy, n_components)?;

        Ok((x_canonical, y_canonical, correlations))
    }

    fn center_data(&self, data: &Array2<Float>) -> Result<Array2<Float>, ManifoldError> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut centered = data.clone();

        // Compute column means
        for j in 0..n_features {
            let mean = data.column(j).mean().unwrap_or(0.0);
            for i in 0..n_samples {
                centered[[i, j]] -= mean;
            }
        }

        Ok(centered)
    }

    fn compute_regularized_covariance(
        &self,
        data: &Array2<Float>,
    ) -> Result<Array2<Float>, ManifoldError> {
        let n_samples = data.nrows();
        let n_features = data.ncols();
        let mut covariance = Array2::zeros((n_features, n_features));

        // Compute covariance matrix
        for i in 0..n_features {
            for j in 0..n_features {
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += data[[k, i]] * data[[k, j]];
                }
                covariance[[i, j]] = cov / (n_samples - 1) as Float;
            }
        }

        // Add regularization
        for i in 0..n_features {
            covariance[[i, i]] += self.regularization;
        }

        Ok(covariance)
    }

    #[allow(clippy::type_complexity)] // returns (eigenvalues, x_eigenvectors, y_eigenvectors) triple
    fn solve_cca_eigenvalue_problem(
        &self,
        cxx: &Array2<Float>,
        cyy: &Array2<Float>,
        cxy: &Array2<Float>,
        n_components: usize,
    ) -> Result<(Array1<Float>, Array2<Float>, Array2<Float>), ManifoldError> {
        // Canonical Correlation Analysis via the whitened cross-covariance.
        //
        // With Cxx, Cyy regularized (hence symmetric positive definite) and the
        // cross-covariance Cxy, define the whitening transforms Cxx^(-1/2) and
        // Cyy^(-1/2). The matrix
        //     M = Cxx^(-1/2) * Cxy * Cyy^(-1/2)
        // has singular values equal to the canonical correlations, and its left and
        // right singular vectors map back to the canonical weight vectors:
        //     Wx = Cxx^(-1/2) * U,    Wy = Cyy^(-1/2) * V.
        let n_x = cxx.nrows();
        let n_y = cyy.nrows();

        // Validate that the cross-covariance is consistent with the two
        // auto-covariance blocks before attempting the decomposition.
        if cxy.nrows() != n_x || cxy.ncols() != n_y {
            return Err(ManifoldError::DimensionMismatch(format!(
                "cross-covariance shape {:?} is incompatible with Cxx ({}x{}) and Cyy ({}x{})",
                cxy.dim(),
                n_x,
                n_x,
                n_y,
                n_y
            )));
        }

        let cxx_inv_sqrt = Self::symmetric_inverse_sqrt(cxx)?;
        let cyy_inv_sqrt = Self::symmetric_inverse_sqrt(cyy)?;

        // M = Cxx^(-1/2) * Cxy * Cyy^(-1/2)  (shape n_x x n_y)
        let m = cxx_inv_sqrt.dot(cxy).dot(&cyy_inv_sqrt);

        // SVD: M = U * diag(s) * Vt. Singular values are the canonical correlations.
        let (u, s, vt) = svd(&m, false)
            .map_err(|e: LinalgError| ManifoldError::NumericalInstability(e.to_string()))?;

        let max_rank = s.len().min(u.ncols()).min(vt.nrows());
        let k = n_components.min(max_rank);

        // Canonical correlations are singular values clamped to the valid [0, 1] range.
        let correlations = Array1::from_iter(s.iter().take(k).map(|&sv| sv.clamp(0.0, 1.0)));

        // Wx = Cxx^(-1/2) * U[:, :k]  (n_x x k)
        let u_k = u.slice(scirs2_core::ndarray::s![.., ..k]).to_owned();
        let x_canonical = cxx_inv_sqrt.dot(&u_k);

        // Wy = Cyy^(-1/2) * V[:, :k]. SVD returns Vt, so V[:, :k] = Vt[:k, :]^T.
        let v_k = vt.slice(scirs2_core::ndarray::s![..k, ..]).t().to_owned();
        let y_canonical = cyy_inv_sqrt.dot(&v_k);

        Ok((correlations, x_canonical, y_canonical))
    }

    /// Compute the symmetric inverse square root of a symmetric positive
    /// (semi-)definite matrix via its eigendecomposition: for `A = U Λ Uᵀ`,
    /// returns `U Λ^(-1/2) Uᵀ`. Eigenvalues below a small floor are clamped to
    /// avoid division by (near-)zero, keeping the result finite for the
    /// regularized covariance matrices used in CCA.
    fn symmetric_inverse_sqrt(matrix: &Array2<Float>) -> Result<Array2<Float>, ManifoldError> {
        let (eigenvalues, eigenvectors) = eigh(matrix, UPLO::Upper)
            .map_err(|e: LinalgError| ManifoldError::NumericalInstability(e.to_string()))?;

        let floor = 1e-12;
        let inv_sqrt_values: Array1<Float> = eigenvalues
            .iter()
            .map(|&lambda| {
                let safe = if lambda > floor { lambda } else { floor };
                1.0 / safe.sqrt()
            })
            .collect();

        // Reconstruct U * diag(inv_sqrt_values) * Uᵀ.
        let mut scaled = Array2::zeros(eigenvectors.raw_dim());
        for ((i, j), value) in scaled.indexed_iter_mut() {
            *value = eigenvectors[[i, j]] * inv_sqrt_values[j];
        }
        let result = scaled.dot(&eigenvectors.t());

        Ok(result)
    }
}

impl FittedManifoldAwareCCA {
    /// Get canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<Float> {
        &self.canonical_correlations
    }

    /// Transform new (out-of-sample) data using the fitted model.
    ///
    /// # Errors
    ///
    /// Projecting unseen samples requires an out-of-sample extension of the
    /// underlying manifold embedding (e.g. a Nyström extension or a learned
    /// parametric mapping). The fitted model only retains the training
    /// embeddings and the canonical vectors in embedding space; it does not
    /// store the original training data or a parametric forward map, so new
    /// samples cannot be embedded. Rather than return fabricated zero matrices,
    /// this method reports an honest error. Use [`Self::manifold_embeddings`] to
    /// retrieve the embeddings computed for the training data.
    pub fn transform(
        &self,
        x: ArrayView2<Float>,
        y: ArrayView2<Float>,
    ) -> Result<(Array2<Float>, Array2<Float>), ManifoldError> {
        let _ = (x, y);
        Err(ManifoldError::InvalidParameters(
            "Out-of-sample transform is not supported by ManifoldAwareCCA: the fitted \
             model does not retain the training data or a parametric mapping required \
             to embed new samples into the manifold. Re-fit including the new samples, \
             or use manifold_embeddings() to access the training-set embeddings."
                .to_string(),
        ))
    }

    /// Get manifold embeddings
    pub fn manifold_embeddings(&self) -> (&Array2<Float>, &Array2<Float>) {
        (&self.x_embedding, &self.y_embedding)
    }

    /// Get canonical vectors in embedding space
    pub fn canonical_vectors(&self) -> (&Array2<Float>, &Array2<Float>) {
        (&self.x_canonical, &self.y_canonical)
    }
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_manifold_learning_creation() {
        let manifold = ManifoldLearning::new(ManifoldMethod::UMAP, 2);
        assert_eq!(manifold.n_components, 2);
        assert_eq!(manifold.n_neighbors, 10);
    }

    #[test]
    fn test_manifold_learning_configuration() {
        let manifold = ManifoldLearning::new(ManifoldMethod::TSNE, 3)
            .n_neighbors(15)
            .random_state(42);

        assert_eq!(manifold.n_components, 3);
        assert_eq!(manifold.n_neighbors, 15);
        assert_eq!(manifold.random_state, Some(42));
    }

    #[test]
    fn test_manifold_learning_fit_transform() {
        let data = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0]
        ];

        let manifold = ManifoldLearning::new(ManifoldMethod::UMAP, 2);
        let result = manifold.fit_transform(data.view());

        assert!(result.is_ok());
        let manifold_result = result.expect("operation should succeed");
        assert_eq!(manifold_result.embedding.nrows(), 5);
        assert_eq!(manifold_result.embedding.ncols(), 2);
        assert!(manifold_result.reconstruction_error >= 0.0);
    }

    #[test]
    fn test_manifold_aware_cca_creation() {
        let cca = ManifoldAwareCCA::new(2);
        assert_eq!(cca.n_components, 2);
        assert!((cca.regularization - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_manifold_aware_cca_configuration() {
        let manifold_x = ManifoldLearning::new(ManifoldMethod::TSNE, 2);
        let manifold_y = ManifoldLearning::new(ManifoldMethod::UMAP, 2);

        let cca = ManifoldAwareCCA::new(2)
            .manifold_x(manifold_x)
            .manifold_y(manifold_y)
            .regularization(0.1);

        assert!((cca.regularization - 0.1).abs() < 1e-10);
    }

    #[test]
    fn test_manifold_aware_cca_fit() {
        let x = array![
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [2.0, 3.0, 4.0]
        ];

        let y = array![[2.0, 3.0], [5.0, 6.0], [8.0, 9.0], [3.0, 4.0]];

        let cca = ManifoldAwareCCA::new(2);
        let result = cca.fit(x.view(), y.view());

        assert!(result.is_ok());
        let fitted = result.expect("operation should succeed");
        assert_eq!(fitted.canonical_correlations().len(), 2);
        assert_eq!(fitted.x_embedding.nrows(), 4);
        assert_eq!(fitted.y_embedding.nrows(), 4);
    }

    #[test]
    fn test_different_manifold_methods() {
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let methods = [
            ManifoldMethod::LLE,
            ManifoldMethod::Isomap,
            ManifoldMethod::LaplacianEigenmaps,
            ManifoldMethod::TSNE,
            ManifoldMethod::UMAP,
            ManifoldMethod::DiffusionMaps,
        ];

        for method in &methods {
            let manifold = ManifoldLearning::new(*method, 2);
            let result = manifold.fit_transform(data.view());
            assert!(result.is_ok(), "Method {:?} failed", method);
        }
    }

    #[test]
    fn test_manifold_aware_cca_transform_returns_honest_error() {
        // Out-of-sample transform cannot be performed with only the stored
        // embeddings and canonical vectors, so the method must return an honest
        // error rather than fabricated zero matrices.
        let x = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let y = array![[2.0, 3.0], [4.0, 5.0], [6.0, 7.0]];

        let cca = ManifoldAwareCCA::new(2);
        let fitted = cca.fit(x.view(), y.view()).expect("fit should succeed");

        let new_x = array![[7.0, 8.0], [9.0, 10.0]];
        let new_y = array![[8.0, 9.0], [10.0, 11.0]];

        let result = fitted.transform(new_x.view(), new_y.view());
        assert!(result.is_err());
        match result {
            Err(ManifoldError::InvalidParameters(msg)) => {
                assert!(msg.contains("Out-of-sample transform is not supported"));
            }
            other => panic!(
                "expected an honest InvalidParameters error, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_kernel_pca_returns_honest_error() {
        // Kernel PCA has no backend in AdvancedManifoldLearning; selecting it must
        // produce an honest error instead of silently running a different method.
        let data = array![[1.0, 2.0], [2.0, 3.0], [3.0, 4.0], [4.0, 5.0]];

        let manifold = ManifoldLearning::new(ManifoldMethod::KernelPCA, 2);
        let result = manifold.fit_transform(data.view());

        assert!(result.is_err());
        match result {
            Err(ManifoldError::InvalidParameters(msg)) => {
                assert!(msg.contains("Kernel PCA is not implemented"));
            }
            other => panic!(
                "expected an honest InvalidParameters error, got {:?}",
                other
            ),
        }
    }

    #[test]
    fn test_solve_cca_eigenvalue_problem_real_correlation() {
        // A real CCA solve must recover a meaningful canonical correlation from
        // correlated data, NOT the previously fabricated constant 1.0 - i*0.1.
        // Construct data where X and Y share a single strong latent direction.
        let cca = ManifoldAwareCCA::new(1).regularization(1e-6);

        // X = [latent, noise], Y = [latent (sign-flipped + scaled), noise]
        // The first canonical correlation should be close to 1.
        let x = array![
            [1.0, 0.2],
            [2.0, -0.1],
            [3.0, 0.05],
            [4.0, 0.3],
            [5.0, -0.2],
            [6.0, 0.1],
        ];
        let y = array![
            [-2.0, 0.1],
            [-4.0, 0.4],
            [-6.0, -0.2],
            [-8.0, 0.15],
            [-10.0, -0.05],
            [-12.0, 0.25],
        ];

        let x_centered = cca.center_data(&x).expect("center x");
        let y_centered = cca.center_data(&y).expect("center y");

        let n_samples = x_centered.nrows();
        let n_x = x_centered.ncols();
        let n_y = y_centered.ncols();
        let mut cxy = Array2::zeros((n_x, n_y));
        for i in 0..n_x {
            for j in 0..n_y {
                let mut cov = 0.0;
                for k in 0..n_samples {
                    cov += x_centered[[k, i]] * y_centered[[k, j]];
                }
                cxy[[i, j]] = cov / (n_samples - 1) as Float;
            }
        }
        let cxx = cca
            .compute_regularized_covariance(&x_centered)
            .expect("cxx");
        let cyy = cca
            .compute_regularized_covariance(&y_centered)
            .expect("cyy");

        let (correlations, x_canonical, y_canonical) = cca
            .solve_cca_eigenvalue_problem(&cxx, &cyy, &cxy, 1)
            .expect("cca solve should succeed");

        assert_eq!(correlations.len(), 1);
        // Correlation must be a valid value in [0, 1] ...
        assert!((0.0..=1.0 + 1e-9).contains(&correlations[0]));
        // ... and, because X and Y share a near-perfect linear relationship, it
        // must be high. This would be impossible with fabricated identity output.
        assert!(
            correlations[0] > 0.9,
            "expected strong canonical correlation, got {}",
            correlations[0]
        );

        // Canonical weight matrices must have the right shapes and contain finite,
        // non-degenerate values (not the old identity pattern).
        assert_eq!(x_canonical.dim(), (n_x, 1));
        assert_eq!(y_canonical.dim(), (n_y, 1));
        assert!(x_canonical.iter().all(|v| v.is_finite()));
        assert!(y_canonical.iter().all(|v| v.is_finite()));
        assert!(x_canonical.iter().any(|&v| v.abs() > 1e-8));
        assert!(y_canonical.iter().any(|&v| v.abs() > 1e-8));
    }

    #[test]
    fn test_center_data() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let cca = ManifoldAwareCCA::new(2);
        let centered = cca.center_data(&data).expect("operation should succeed");

        // Check that columns have zero mean (approximately)
        let col1_mean = centered.column(0).mean().expect("operation should succeed");
        let col2_mean = centered.column(1).mean().expect("operation should succeed");

        assert!((col1_mean).abs() < 1e-10);
        assert!((col2_mean).abs() < 1e-10);
    }

    #[test]
    fn test_regularized_covariance() {
        let data = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];

        let cca = ManifoldAwareCCA::new(2).regularization(0.1);
        let cov = cca
            .compute_regularized_covariance(&data)
            .expect("operation should succeed");

        // Check that regularization was added to diagonal
        assert!(cov[[0, 0]] >= 0.1);
        assert!(cov[[1, 1]] >= 0.1);
        assert_eq!(cov.nrows(), 2);
        assert_eq!(cov.ncols(), 2);
    }
}
