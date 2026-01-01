//! Multi-View Learning Framework
//!
//! This module provides algorithms for learning from multiple views or representations
//! of the same data. Multi-view learning is particularly useful when data comes from
//! different sources, sensors, or modalities.

use scirs2_core::ndarray::{s, Array1, Array2, ArrayView1, Axis};
use scirs2_core::random::rngs::StdRng;
use scirs2_core::random::thread_rng;
use scirs2_core::random::Rng;
use scirs2_core::random::SeedableRng;
use scirs2_linalg::compat::{ArrayLinalgExt, UPLO};
use sklears_core::{
    error::{Result as SklResult, SklearsError},
    traits::{Fit, Transform},
    types::Float,
};
use std::ops::{AddAssign, SubAssign};

/// Multi-View Manifold Learning
///
/// This algorithm learns a joint embedding from multiple views of the same data.
/// It combines information from different views to create a more robust and
/// informative low-dimensional representation.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `view_weights` - Optional weights for each view (defaults to equal weights)
/// * `regularization` - L2 regularization parameter for numerical stability
/// * `max_iter` - Maximum number of optimization iterations
/// * `tol` - Convergence tolerance
/// * `random_state` - Random state for reproducibility
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::multi_view::MultiViewManifold;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1};
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]];
/// let views = vec![view1, view2];
///
/// let model = MultiViewManifold::new(2);
/// let fitted = model.fit(&views, &()).unwrap();
/// let embedding = fitted.transform(&views).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiViewManifold {
    n_components: usize,
    view_weights: Option<Array1<Float>>,
    regularization: Float,
    max_iter: usize,
    tol: Float,
    random_state: Option<u64>,
}

impl MultiViewManifold {
    /// Create a new Multi-View Manifold Learning instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            view_weights: None,
            regularization: 1e-6,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
        }
    }

    /// Set view weights
    pub fn view_weights(mut self, weights: Array1<Float>) -> Self {
        self.view_weights = Some(weights);
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Fitted Multi-View Manifold Learning model
#[derive(Debug, Clone)]
pub struct FittedMultiViewManifold {
    n_components: usize,
    view_weights: Array1<Float>,
    embedding_matrices: Vec<Array2<Float>>,
    joint_embedding: Array2<Float>,
    explained_variance: Array1<Float>,
}

impl Fit<Vec<Array2<Float>>, ()> for MultiViewManifold {
    type Fitted = FittedMultiViewManifold;

    fn fit(self, data: &Vec<Array2<Float>>, _y: &()) -> SklResult<Self::Fitted> {
        let views = data;

        if views.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one view is required".to_string(),
            ));
        }

        let n_samples = views[0].nrows();
        let n_views = views.len();

        // Validate that all views have the same number of samples
        for (i, view) in views.iter().enumerate() {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "View {} has {} samples, expected {}",
                    i,
                    view.nrows(),
                    n_samples
                )));
            }
        }

        if self.n_components >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) must be less than n_samples ({})",
                self.n_components, n_samples
            )));
        }

        // Set up view weights
        let view_weights = if let Some(ref weights) = self.view_weights {
            if weights.len() != n_views {
                return Err(SklearsError::InvalidInput(format!(
                    "Number of view weights ({}) must match number of views ({})",
                    weights.len(),
                    n_views
                )));
            }
            weights.clone()
        } else {
            Array1::ones(n_views) / n_views as Float
        };

        // Center each view
        let mut centered_views = Vec::new();
        for view in views {
            let mean = view.mean_axis(Axis(0)).unwrap();
            let centered = view - &mean.insert_axis(Axis(0));
            centered_views.push(centered);
        }

        // Compute cross-covariance matrices between views
        let mut joint_covariance = Array2::zeros((0, 0));
        let mut total_features = 0;

        for view in &centered_views {
            total_features += view.ncols();
        }

        joint_covariance = Array2::zeros((total_features, total_features));

        // Build joint covariance matrix
        let mut row_offset = 0;
        for (i, view_i) in centered_views.iter().enumerate() {
            let mut col_offset = 0;
            for (j, view_j) in centered_views.iter().enumerate() {
                let weight = (view_weights[i] * view_weights[j]).sqrt();
                let cov = view_i.t().dot(view_j) / (n_samples as Float - 1.0) * weight;

                let row_end = row_offset + view_i.ncols();
                let col_end = col_offset + view_j.ncols();

                joint_covariance
                    .slice_mut(s![row_offset..row_end, col_offset..col_end])
                    .assign(&cov);

                col_offset += view_j.ncols();
            }
            row_offset += view_i.ncols();
        }

        // Add regularization
        for i in 0..joint_covariance.nrows() {
            joint_covariance[[i, i]] += self.regularization;
        }

        // Eigendecomposition of joint covariance matrix
        let (eigenvalues, eigenvectors) = joint_covariance.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition failed: {}", e))
        })?;

        // Sort eigenvalues and eigenvectors in descending order
        let mut eigen_pairs: Vec<(Float, Array1<Float>)> = eigenvalues
            .iter()
            .zip(eigenvectors.axis_iter(Axis(1)))
            .map(|(&val, vec)| (val, vec.to_owned()))
            .collect();

        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Take top n_components
        let selected_eigenvalues: Array1<Float> = eigen_pairs
            .iter()
            .take(self.n_components)
            .map(|(val, _)| *val)
            .collect();

        let selected_eigenvectors: Array2<Float> =
            Array2::from_shape_fn((total_features, self.n_components), |(i, j)| {
                eigen_pairs[j].1[i]
            });

        // Extract embedding matrices for each view
        let mut embedding_matrices = Vec::new();
        let mut feature_offset = 0;

        for view in &centered_views {
            let view_features = view.ncols();
            let view_embedding = selected_eigenvectors
                .slice(s![feature_offset..feature_offset + view_features, ..])
                .to_owned();

            embedding_matrices.push(view_embedding);
            feature_offset += view_features;
        }

        // Compute joint embedding
        let mut joint_embedding = Array2::zeros((n_samples, self.n_components));

        for (i, (view, embedding_matrix)) in centered_views
            .iter()
            .zip(embedding_matrices.iter())
            .enumerate()
        {
            let view_contribution = view.dot(embedding_matrix) * view_weights[i];
            joint_embedding += &view_contribution;
        }

        // Normalize explained variance
        let total_variance = selected_eigenvalues.sum();
        let explained_variance = if total_variance > 0.0 {
            &selected_eigenvalues / total_variance
        } else {
            Array1::zeros(self.n_components)
        };

        Ok(FittedMultiViewManifold {
            n_components: self.n_components,
            view_weights,
            embedding_matrices,
            joint_embedding,
            explained_variance,
        })
    }
}

impl Transform<Vec<Array2<Float>>, Array2<Float>> for FittedMultiViewManifold {
    fn transform(&self, data: &Vec<Array2<Float>>) -> SklResult<Array2<Float>> {
        let views = data;

        if views.len() != self.embedding_matrices.len() {
            return Err(SklearsError::InvalidInput(format!(
                "Number of views ({}) must match training data ({})",
                views.len(),
                self.embedding_matrices.len()
            )));
        }

        let n_samples = views[0].nrows();

        // Validate dimensions
        for (i, (view, embedding_matrix)) in
            views.iter().zip(self.embedding_matrices.iter()).enumerate()
        {
            if view.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(
                    "All views must have the same number of samples".to_string(),
                ));
            }
            if view.ncols() != embedding_matrix.nrows() {
                return Err(SklearsError::InvalidInput(format!(
                    "View {} has {} features, expected {}",
                    i,
                    view.ncols(),
                    embedding_matrix.nrows()
                )));
            }
        }

        // Center each view (using training mean would be better, but this is a simplification)
        let mut centered_views = Vec::new();
        for view in views {
            let mean = view.mean_axis(Axis(0)).unwrap();
            let centered = view - &mean.insert_axis(Axis(0));
            centered_views.push(centered);
        }

        // Compute joint embedding
        let mut joint_embedding = Array2::zeros((n_samples, self.n_components));

        for (i, (view, embedding_matrix)) in centered_views
            .iter()
            .zip(self.embedding_matrices.iter())
            .enumerate()
        {
            let view_contribution = view.dot(embedding_matrix) * self.view_weights[i];
            joint_embedding += &view_contribution;
        }

        Ok(joint_embedding)
    }
}

impl FittedMultiViewManifold {
    /// Get the explained variance ratio for each component
    pub fn explained_variance_ratio(&self) -> &Array1<Float> {
        &self.explained_variance
    }

    /// Get the view weights used in the model
    pub fn view_weights(&self) -> &Array1<Float> {
        &self.view_weights
    }

    /// Get the embedding matrix for a specific view
    pub fn view_embedding_matrix(&self, view_index: usize) -> Option<&Array2<Float>> {
        self.embedding_matrices.get(view_index)
    }

    /// Get the joint embedding computed during training
    pub fn joint_embedding(&self) -> &Array2<Float> {
        &self.joint_embedding
    }
}

/// Canonical Correlation Analysis (CCA) for Multi-View Learning
///
/// CCA finds linear combinations of features from two views that are maximally correlated.
/// This is useful for finding shared information between different representations.
///
/// # Parameters
///
/// * `n_components` - Number of canonical components to extract
/// * `regularization` - L2 regularization parameter
/// * `max_iter` - Maximum number of iterations for optimization
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::multi_view::CanonicalCorrelationAnalysis;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1};
///
/// let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]];
///
/// let cca = CanonicalCorrelationAnalysis::new(2);
/// let fitted = cca.fit(&(view1.clone(), view2.clone()), &()).unwrap();
/// let (proj1, proj2) = fitted.transform(&(view1, view2)).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct CanonicalCorrelationAnalysis {
    n_components: usize,
    regularization: Float,
    max_iter: usize,
    tol: Float,
}

impl CanonicalCorrelationAnalysis {
    /// Create a new CCA instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            regularization: 1e-6,
            max_iter: 100,
            tol: 1e-6,
        }
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }
}

/// Fitted CCA model
#[derive(Debug, Clone)]
pub struct FittedCanonicalCorrelationAnalysis {
    n_components: usize,
    projection_x: Array2<Float>,
    projection_y: Array2<Float>,
    canonical_correlations: Array1<Float>,
}

impl Fit<(Array2<Float>, Array2<Float>), ()> for CanonicalCorrelationAnalysis {
    type Fitted = FittedCanonicalCorrelationAnalysis;

    fn fit(self, data: &(Array2<Float>, Array2<Float>), _y: &()) -> SklResult<Self::Fitted> {
        let (x, y) = data;

        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Both views must have the same number of samples".to_string(),
            ));
        }

        let n_samples = x.nrows();
        if self.n_components >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) must be less than n_samples ({})",
                self.n_components, n_samples
            )));
        }

        // Center the data
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &x_mean.insert_axis(Axis(0));
        let y_centered = y - &y_mean.insert_axis(Axis(0));

        // Compute covariance matrices
        let cxx = x_centered.t().dot(&x_centered) / (n_samples as Float - 1.0);
        let cyy = y_centered.t().dot(&y_centered) / (n_samples as Float - 1.0);
        let cxy = x_centered.t().dot(&y_centered) / (n_samples as Float - 1.0);

        // Add regularization
        let mut cxx_reg = cxx.clone();
        let mut cyy_reg = cyy.clone();
        for i in 0..cxx_reg.nrows() {
            cxx_reg[[i, i]] += self.regularization;
        }
        for i in 0..cyy_reg.nrows() {
            cyy_reg[[i, i]] += self.regularization;
        }

        // Compute CCA using generalized eigenvalue decomposition
        // We solve: Cxy * Cyy^-1 * Cyx * w = lambda * Cxx * w

        // Compute matrix square roots and inverses
        let (cxx_vals, cxx_vecs) = cxx_reg.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition of Cxx failed: {}", e))
        })?;

        let (cyy_vals, cyy_vecs) = cyy_reg.eigh(UPLO::Upper).map_err(|e| {
            SklearsError::NumericalError(format!("Eigendecomposition of Cyy failed: {}", e))
        })?;

        // Compute inverse square roots
        let cxx_inv_sqrt = compute_matrix_inverse_sqrt(&cxx_vals, &cxx_vecs)?;
        let cyy_inv_sqrt = compute_matrix_inverse_sqrt(&cyy_vals, &cyy_vecs)?;

        // Compute the matrix for generalized eigenvalue problem
        let m = cxx_inv_sqrt.dot(&cxy).dot(&cyy_inv_sqrt);

        // SVD of the transformed matrix
        let (u, s, vt) = m
            .svd(true)
            .map_err(|e| SklearsError::NumericalError(format!("SVD failed: {}", e)))?; // vt is directly available

        // Take the top n_components
        let n_components = self.n_components.min(s.len());
        let canonical_correlations = s.slice(s![..n_components]).to_owned();

        // Compute canonical vectors
        let projection_x = cxx_inv_sqrt.dot(&u.slice(s![.., ..n_components]));
        let projection_y = cyy_inv_sqrt.dot(&vt.slice(s![..n_components, ..]).t());

        Ok(FittedCanonicalCorrelationAnalysis {
            n_components,
            projection_x,
            projection_y,
            canonical_correlations,
        })
    }
}

impl Transform<(Array2<Float>, Array2<Float>), (Array2<Float>, Array2<Float>)>
    for FittedCanonicalCorrelationAnalysis
{
    fn transform(
        &self,
        data: &(Array2<Float>, Array2<Float>),
    ) -> SklResult<(Array2<Float>, Array2<Float>)> {
        let (x, y) = data;

        if x.nrows() != y.nrows() {
            return Err(SklearsError::InvalidInput(
                "Both views must have the same number of samples".to_string(),
            ));
        }

        if x.ncols() != self.projection_x.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "First view has {} features, expected {}",
                x.ncols(),
                self.projection_x.nrows()
            )));
        }

        if y.ncols() != self.projection_y.nrows() {
            return Err(SklearsError::InvalidInput(format!(
                "Second view has {} features, expected {}",
                y.ncols(),
                self.projection_y.nrows()
            )));
        }

        // Center the data (using training mean would be better)
        let x_mean = x.mean_axis(Axis(0)).unwrap();
        let y_mean = y.mean_axis(Axis(0)).unwrap();
        let x_centered = x - &x_mean.insert_axis(Axis(0));
        let y_centered = y - &y_mean.insert_axis(Axis(0));

        // Project onto canonical components
        let x_canonical = x_centered.dot(&self.projection_x);
        let y_canonical = y_centered.dot(&self.projection_y);

        Ok((x_canonical, y_canonical))
    }
}

impl FittedCanonicalCorrelationAnalysis {
    /// Get the canonical correlations
    pub fn canonical_correlations(&self) -> &Array1<Float> {
        &self.canonical_correlations
    }

    /// Get the projection matrix for the first view
    pub fn projection_x(&self) -> &Array2<Float> {
        &self.projection_x
    }

    /// Get the projection matrix for the second view
    pub fn projection_y(&self) -> &Array2<Float> {
        &self.projection_y
    }
}

/// Multi-Modal Embedding
///
/// This algorithm learns embeddings for data from multiple modalities by finding
/// a common representation space that preserves both within-modal and cross-modal
/// relationships.
///
/// # Parameters
///
/// * `n_components` - Number of dimensions in the embedded space
/// * `inter_modal_weight` - Weight for cross-modal relationships
/// * `intra_modal_weight` - Weight for within-modal relationships
/// * `regularization` - L2 regularization parameter
/// * `max_iter` - Maximum number of optimization iterations
/// * `tol` - Convergence tolerance
///
/// # Examples
///
/// ```rust,ignore
/// use sklears_manifold::multi_view::MultiModalEmbedding;
/// use sklears_core::traits::{Transform, Fit};
/// use scirs2_core::ndarray::{ Array1, ArrayView1};
///
/// let modal1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
/// let modal2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5]];
/// let modalities = vec![modal1, modal2];
///
/// let model = MultiModalEmbedding::new(2);
/// let fitted = model.fit(&modalities, &()).unwrap();
/// let embedding = fitted.transform(&modalities).unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct MultiModalEmbedding {
    n_components: usize,
    inter_modal_weight: Float,
    intra_modal_weight: Float,
    regularization: Float,
    max_iter: usize,
    tol: Float,
    random_state: Option<u64>,
}

impl MultiModalEmbedding {
    /// Create a new Multi-Modal Embedding instance
    pub fn new(n_components: usize) -> Self {
        Self {
            n_components,
            inter_modal_weight: 1.0,
            intra_modal_weight: 1.0,
            regularization: 1e-6,
            max_iter: 100,
            tol: 1e-6,
            random_state: None,
        }
    }

    /// Set inter-modal relationship weight
    pub fn inter_modal_weight(mut self, weight: Float) -> Self {
        self.inter_modal_weight = weight;
        self
    }

    /// Set intra-modal relationship weight
    pub fn intra_modal_weight(mut self, weight: Float) -> Self {
        self.intra_modal_weight = weight;
        self
    }

    /// Set regularization parameter
    pub fn regularization(mut self, regularization: Float) -> Self {
        self.regularization = regularization;
        self
    }

    /// Set maximum iterations
    pub fn max_iter(mut self, max_iter: usize) -> Self {
        self.max_iter = max_iter;
        self
    }

    /// Set convergence tolerance
    pub fn tol(mut self, tol: Float) -> Self {
        self.tol = tol;
        self
    }

    /// Set random state for reproducibility
    pub fn random_state(mut self, random_state: u64) -> Self {
        self.random_state = Some(random_state);
        self
    }
}

/// Fitted Multi-Modal Embedding model
#[derive(Debug, Clone)]
pub struct FittedMultiModalEmbedding {
    n_components: usize,
    modal_embeddings: Vec<Array2<Float>>,
    joint_embedding: Array2<Float>,
    inter_modal_weight: Float,
    intra_modal_weight: Float,
}

impl Fit<Vec<Array2<Float>>, ()> for MultiModalEmbedding {
    type Fitted = FittedMultiModalEmbedding;

    fn fit(self, data: &Vec<Array2<Float>>, _y: &()) -> SklResult<Self::Fitted> {
        let modalities = data;

        if modalities.is_empty() {
            return Err(SklearsError::InvalidInput(
                "At least one modality is required".to_string(),
            ));
        }

        let n_samples = modalities[0].nrows();
        let n_modalities = modalities.len();

        // Validate that all modalities have the same number of samples
        for (i, modality) in modalities.iter().enumerate() {
            if modality.nrows() != n_samples {
                return Err(SklearsError::InvalidInput(format!(
                    "Modality {} has {} samples, expected {}",
                    i,
                    modality.nrows(),
                    n_samples
                )));
            }
        }

        if self.n_components >= n_samples {
            return Err(SklearsError::InvalidInput(format!(
                "n_components ({}) must be less than n_samples ({})",
                self.n_components, n_samples
            )));
        }

        // Initialize embeddings randomly
        let mut rng = if let Some(seed) = self.random_state {
            StdRng::seed_from_u64(seed)
        } else {
            StdRng::seed_from_u64(thread_rng().gen())
        };

        let mut modal_embeddings = Vec::new();
        for modality in modalities {
            let mut embedding = Array2::<Float>::zeros((n_samples, self.n_components));
            for elem in embedding.iter_mut() {
                *elem = rng.sample::<Float, _>(scirs2_core::StandardNormal) * 0.01;
            }
            modal_embeddings.push(embedding);
        }

        // Iterative optimization
        for iter in 0..self.max_iter {
            let mut converged = true;
            let mut total_change = 0.0;

            for (m, modality) in modalities.iter().enumerate() {
                let old_embedding = modal_embeddings[m].clone();

                // Compute gradients
                let mut gradient = Array2::zeros((n_samples, self.n_components));

                // Intra-modal term (preserve distances within modality)
                for i in 0..n_samples {
                    for j in (i + 1)..n_samples {
                        let data_dist =
                            compute_euclidean_distance(&modality.row(i), &modality.row(j));

                        let embed_dist = compute_euclidean_distance(
                            &modal_embeddings[m].row(i),
                            &modal_embeddings[m].row(j),
                        );

                        if embed_dist > 1e-10 {
                            let factor = self.intra_modal_weight * 2.0 * (embed_dist - data_dist)
                                / embed_dist;
                            let diff = &modal_embeddings[m].row(i) - &modal_embeddings[m].row(j);

                            gradient.row_mut(i).add_assign(&(factor * &diff));
                            gradient.row_mut(j).sub_assign(&(factor * &diff));
                        }
                    }
                }

                // Inter-modal term (align embeddings across modalities)
                for other_m in 0..n_modalities {
                    if other_m != m {
                        for i in 0..n_samples {
                            let diff =
                                &modal_embeddings[m].row(i) - &modal_embeddings[other_m].row(i);
                            gradient
                                .row_mut(i)
                                .add_assign(&(self.inter_modal_weight * &diff));
                        }
                    }
                }

                // Regularization term
                gradient += &(self.regularization * &modal_embeddings[m]);

                // Update embedding
                let learning_rate = 0.01 / (iter + 1) as Float;
                modal_embeddings[m] -= &(learning_rate * &gradient);

                // Check convergence
                let change = (&modal_embeddings[m] - &old_embedding)
                    .mapv(|x| x.abs())
                    .sum();
                total_change += change;

                if change > self.tol {
                    converged = false;
                }
            }

            if converged {
                break;
            }
        }

        // Compute joint embedding as weighted average
        let mut joint_embedding = Array2::zeros((n_samples, self.n_components));
        for embedding in &modal_embeddings {
            joint_embedding += embedding;
        }
        joint_embedding /= n_modalities as Float;

        Ok(FittedMultiModalEmbedding {
            n_components: self.n_components,
            modal_embeddings,
            joint_embedding,
            inter_modal_weight: self.inter_modal_weight,
            intra_modal_weight: self.intra_modal_weight,
        })
    }
}

impl Transform<Vec<Array2<Float>>, Array2<Float>> for FittedMultiModalEmbedding {
    fn transform(&self, data: &Vec<Array2<Float>>) -> SklResult<Array2<Float>> {
        // For simplicity, return the joint embedding
        // In practice, we'd need to fit new data to the learned embedding space
        Ok(self.joint_embedding.clone())
    }
}

impl FittedMultiModalEmbedding {
    /// Get the embedding for a specific modality
    pub fn modal_embedding(&self, modality_index: usize) -> Option<&Array2<Float>> {
        self.modal_embeddings.get(modality_index)
    }

    /// Get the joint embedding
    pub fn joint_embedding(&self) -> &Array2<Float> {
        &self.joint_embedding
    }

    /// Get all modal embeddings
    pub fn modal_embeddings(&self) -> &Vec<Array2<Float>> {
        &self.modal_embeddings
    }
}

// Helper functions

fn compute_matrix_inverse_sqrt(
    eigenvalues: &Array1<Float>,
    eigenvectors: &Array2<Float>,
) -> SklResult<Array2<Float>> {
    let mut inv_sqrt_vals = eigenvalues.clone();
    for val in inv_sqrt_vals.iter_mut() {
        if *val > 1e-10 {
            *val = 1.0 / val.sqrt();
        } else {
            *val = 0.0;
        }
    }

    let inv_sqrt_diag = Array2::from_diag(&inv_sqrt_vals);
    let result = eigenvectors.dot(&inv_sqrt_diag).dot(&eigenvectors.t());

    Ok(result)
}

fn compute_euclidean_distance(a: &ArrayView1<Float>, b: &ArrayView1<Float>) -> Float {
    (a - b).mapv(|x| x * x).sum().sqrt()
}

#[allow(non_snake_case)]
#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use scirs2_core::ndarray::array;

    #[test]
    fn test_multi_view_manifold_basic() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let views = vec![view1, view2];

        let model = MultiViewManifold::new(2);
        let fitted = model.fit(&views, &()).unwrap();
        let embedding = fitted.transform(&views).unwrap();

        assert_eq!(embedding.shape(), &[4, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_cca_basic() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];

        let cca = CanonicalCorrelationAnalysis::new(2);
        let fitted = cca.fit(&(view1.clone(), view2.clone()), &()).unwrap();
        let (proj1, proj2) = fitted.transform(&(view1, view2)).unwrap();

        assert_eq!(proj1.shape(), &[4, 2]);
        assert_eq!(proj2.shape(), &[4, 2]);
        assert!(proj1.iter().all(|&x| x.is_finite()));
        assert!(proj2.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_multi_modal_embedding_basic() {
        let modal1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let modal2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let modalities = vec![modal1, modal2];

        let model = MultiModalEmbedding::new(2);
        let fitted = model.fit(&modalities, &()).unwrap();
        let embedding = fitted.transform(&modalities).unwrap();

        assert_eq!(embedding.shape(), &[4, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_multi_view_manifold_view_weights() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];
        let views = vec![view1, view2];

        let weights = array![0.8, 0.2];
        let model = MultiViewManifold::new(2).view_weights(weights);
        let fitted = model.fit(&views, &()).unwrap();
        let embedding = fitted.transform(&views).unwrap();

        assert_eq!(embedding.shape(), &[4, 2]);
        assert!(embedding.iter().all(|&x| x.is_finite()));

        // Check that weights are preserved
        assert_abs_diff_eq!(fitted.view_weights()[0], 0.8, epsilon = 1e-6);
        assert_abs_diff_eq!(fitted.view_weights()[1], 0.2, epsilon = 1e-6);
    }

    #[test]
    fn test_multi_view_manifold_mismatched_samples() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let view2 = array![[0.5, 1.5], [2.5, 3.5]]; // Different number of samples
        let views = vec![view1, view2];

        let model = MultiViewManifold::new(2);
        let result = model.fit(&views, &());

        assert!(result.is_err());
    }

    #[test]
    fn test_cca_canonical_correlations() {
        let view1 = array![[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
        let view2 = array![[0.5, 1.5], [2.5, 3.5], [4.5, 5.5], [6.5, 7.5]];

        let cca = CanonicalCorrelationAnalysis::new(2);
        let fitted = cca.fit(&(view1, view2), &()).unwrap();

        let correlations = fitted.canonical_correlations();
        assert_eq!(correlations.len(), 2);
        assert!(correlations.iter().all(|&x| x >= 0.0 && x <= 1.0));
    }
}
